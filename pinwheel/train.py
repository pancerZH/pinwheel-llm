"""Pretraining loop. Every TrainConfig field is also a command-line flag:

    uv run python -m pinwheel.train --data_dir data/tok8192/tinystories --max_tokens 5e8 --note "v0 tinystories"

Writes runs/<run>/{config.json,log.jsonl,samples.md}, ckpt/<run>.pt and a line in runs/results.tsv.
"""

import argparse
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch

from pinwheel.bench import append_bench_row, run_benchmarks
from pinwheel.data import TokenData
from pinwheel.eval import evaluate_bpb
from pinwheel.model import GPT, GPTConfig
from pinwheel.sample import complete
from pinwheel.tokenizer import load_tokenizer, token_byte_lengths


@dataclass
class TrainConfig:
    data_dir: str = "data/tok8192/tinystories"
    run: str = ""  # run name; defaults to a timestamp
    note: str = ""  # one line for results.tsv: what this run changes
    seed: int = 0
    # model (the defaults are v0, about 50.7M parameters)
    n_layer: int = 10
    n_head: int = 10
    n_kv_head: int = 5
    d_model: int = 640
    d_ff: int = 1728
    seq_len: int = 1024
    # optimization
    max_tokens: float = 1e9
    batch_tokens: int = 524288  # tokens per optimizer step, reached by gradient accumulation
    micro_batch: int = 32  # sequences per forward pass
    lr: float = 2e-3  # best of 1e-3 / 2e-3 / 4e-3 for v0 at 1B tokens (stage 1e)
    min_lr_frac: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    compile: bool = True
    overfit_batch: bool = False  # sanity check: train on one fixed batch, loss should fall to ~0
    # evaluation and logging
    eval_every: int = 250
    eval_tokens: int = 2097152
    sample_every: int = 1000
    log_every: int = 10
    peak_tflops: float = 96.0  # measured: 16k x 16k BF16 matmul on the RTX 5070 Ti; only used for MFU


def parse_args():
    p = argparse.ArgumentParser()
    for f in fields(TrainConfig):
        kind = (lambda s: s.lower() in ("1", "true", "yes")) if f.type is bool else f.type
        p.add_argument(f"--{f.name}", type=kind, default=f.default)
    return TrainConfig(**vars(p.parse_args()))


def lr_at(step, cfg, max_steps):
    """Linear warmup, then cosine decay down to min_lr_frac * lr."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def git_sha():
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    sha = git("rev-parse", "--short", "HEAD") or "none"
    return sha + ("+dirty" if git("status", "--porcelain", "--untracked-files=no") else "")


def write_samples(path, step, model, tok, prompts):
    g = torch.Generator(device="cuda").manual_seed(0)  # same seed each time, so steps compare
    with open(path, "a") as f:
        f.write(f"## step {step}\n\n")
        for prompt in prompts:
            text = complete(model, tok, prompt, max_new_tokens=100, generator=g)
            f.write(f"- **{prompt}**{text}\n")
        f.write("\n")


def append_result(cfg, run, n_params, tokens, hours, val_bpb, hellaswag):
    path = Path("runs/results.tsv")
    if not path.exists():
        path.write_text("run_id\tdate\tgit_sha\tchange\tparams\ttokens\twall_h\tval_bpb\thellaswag\tnotes\n")
    row = [run, time.strftime("%Y-%m-%d"), git_sha(), cfg.note, f"{n_params / 1e6:.1f}M",
           f"{tokens / 1e6:.0f}M", f"{hours:.2f}", f"{val_bpb:.4f}", f"{hellaswag:.4f}", ""]
    with open(path, "a") as f:
        f.write("\t".join(row) + "\n")


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    device = "cuda"

    data = TokenData(cfg.data_dir)
    tok = load_tokenizer(data.tokenizer_path)
    token_bytes = token_byte_lengths(tok).to(device)
    mcfg = GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=cfg.n_layer, n_head=cfg.n_head,
                     n_kv_head=cfg.n_kv_head, d_model=cfg.d_model, d_ff=cfg.d_ff,
                     max_seq_len=cfg.seq_len)
    model = GPT(mcfg).to(device)
    train_model = torch.compile(model) if cfg.compile else model
    n_params, flops_per_token = model.num_params(), model.flops_per_token(cfg.seq_len)

    # Weight decay only on matrices; norms (1-D) are left alone.
    params = list(model.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
         {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), fused=True,
    )

    accum, rest = divmod(cfg.batch_tokens, cfg.micro_batch * cfg.seq_len)
    assert accum > 0 and rest == 0, "batch_tokens must be a multiple of micro_batch * seq_len"
    max_steps = int(cfg.max_tokens // cfg.batch_tokens)

    run = cfg.run or time.strftime("%Y%m%d-%H%M%S")
    out = Path("runs") / run
    out.mkdir(parents=True, exist_ok=False)
    (out / "config.json").write_text(json.dumps(
        {"train": asdict(cfg), "model": asdict(mcfg), "params": n_params, "git": git_sha()}, indent=2))
    log = open(out / "log.jsonl", "a")
    prompts = Path("prompts.txt").read_text().splitlines()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    print(f"run {run}: {n_params / 1e6:.1f}M params, {max_steps} steps x {cfg.batch_tokens} tokens "
          f"(grad accum {accum})")

    def record(**r):
        log.write(json.dumps(r) + "\n")
        log.flush()
        print("  ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))

    fixed = data.random_batch(cfg.micro_batch, cfg.seq_len, device) if cfg.overfit_batch else None
    t_start, val_bpb = time.time(), float("nan")
    for step in range(max_steps + 1):
        last = step == max_steps
        if step % cfg.eval_every == 0 or last:
            with autocast:
                val_bpb = evaluate_bpb(train_model, data, token_bytes, cfg.micro_batch, cfg.seq_len,
                                       cfg.eval_tokens)
            record(step=step, tokens=step * cfg.batch_tokens, val_bpb=val_bpb)
        if (step > 0 and step % cfg.sample_every == 0) or last:
            with autocast:
                write_samples(out / "samples.md", step, model, tok, prompts)
        if last:
            break

        lr = lr_at(step, cfg, max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        if step % cfg.log_every == 0:
            torch.cuda.synchronize()  # the CPU runs ahead of the GPU: drain earlier steps so only this one is timed
        t0 = time.time()
        loss_sum = torch.zeros((), device=device)
        for _ in range(accum):
            x, y = fixed or data.random_batch(cfg.micro_batch, cfg.seq_len, device)
            with autocast:
                loss = train_model(x, y)
            (loss / accum).backward()
            loss_sum += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % cfg.log_every == 0:
            torch.cuda.synchronize()
            tok_per_s = cfg.batch_tokens / (time.time() - t0)
            record(step=step, tokens=(step + 1) * cfg.batch_tokens, loss=(loss_sum / accum).item(),
                   lr=lr, grad_norm=grad_norm.item(), tok_per_s=round(tok_per_s),
                   mfu=tok_per_s * flops_per_token / (cfg.peak_tflops * 1e12))

    hours = (time.time() - t_start) / 3600
    Path("ckpt").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": asdict(mcfg),
                "train_config": asdict(cfg), "step": max_steps}, Path("ckpt") / f"{run}.pt")
    bench = run_benchmarks(model, tok, save_items_path=Path("runs/bench") / f"{run}.npz")  # about 30 s for v0
    append_bench_row(run, bench)
    hellaswag = bench["hellaswag"]["acc_norm"]
    append_result(cfg, run, n_params, max_steps * cfg.batch_tokens, hours, val_bpb, hellaswag)
    print(f"done in {hours:.2f} h, val_bpb {val_bpb:.4f}, hellaswag {hellaswag:.4f}")


if __name__ == "__main__":
    main()
