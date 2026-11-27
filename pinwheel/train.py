"""Pretraining loop. Every TrainConfig field is also a command-line flag:

    uv run python -m pinwheel.train --data_dir data/tok8192/tinystories --max_tokens 5e8 --note "v0 tinystories"

Writes runs/<run>/{config.json,log.jsonl,samples.md}, ckpt/<run>.pt and a line in runs/results.tsv.
"""

import argparse
import json
import math
import random
import subprocess
import time
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import torch

from pinwheel.bench import append_bench_row, run_benchmarks
from pinwheel.data import TokenData
from pinwheel.eval import evaluate_bpb
from pinwheel.model import GPT, GPTConfig
from pinwheel.muon import Muon
from pinwheel.sample import complete
from pinwheel.tokenizer import load_tokenizer, token_byte_lengths


@dataclass
class TrainConfig:
    data_dir: str = "data/tok8192/tinystories"
    run: str = ""  # run name; defaults to a timestamp
    note: str = ""  # one line for results.tsv: what this run changes
    seed: int = 0
    data_seed: int = -1  # seed of the training-window order; -1 uses seed (seed alone sets the init then)
    # model (the defaults are v0, about 50.7M parameters)
    n_layer: int = 10
    n_head: int = 10
    n_kv_head: int = 5
    d_model: int = 640
    d_ff: int = 1728
    arch: str = "llama"  # "llama" or "gpt2" (LayerNorm, learned positions, GELU MLP)
    # stage 2e, adopted together: 1B tokens, two seeds, 1.0619 vs 1.0712 with Muon alone
    qk_norm: bool = True  # RMSNorm on every head's queries and keys
    zero_init: bool = True  # wo and w_down start at 0
    emb_std: float = 0.002  # init std of the tied embedding (0.02 before); with zero_init, 0.02 makes the model copy its input at step 0
    # stage 3 (see pinwheel/model.py)
    n_loop: int = 1  # passes through the middle layers, which share their weights
    n_prelude: int = 1  # with n_loop > 1: layers before the loop, run once
    n_coda: int = 1  # with n_loop > 1: layers after the loop, run once
    loop_min: int = 0  # > 0: every micro-batch runs a random number of passes in [loop_min, n_loop]; eval uses n_loop
    mtp: int = 0  # multi-token prediction modules, used only in training
    mtp_weight: float = 0.3  # weight of the mean MTP loss (DeepSeek-V3: 0.3, later 0.1)
    seq_len: int = 1024
    # optimization
    max_tokens: float = 1e9
    batch_tokens: int = 131072  # tokens per optimizer step, reached by gradient accumulation (stage 2a)
    micro_batch: int = 32  # sequences per forward pass
    lr: float = 2e-3  # v0, batch 128k, WSD, 1B tokens: 1.4e-3 / 2e-3 / 2.8e-3 give 1.0915 / 1.0850 / 1.0849
    optimizer: str = "muon"  # Muon for the 2-D weights in the blocks, AdamW for the rest (stage 2e); or "adamw"
    muon_lr: float = 2e-3  # on AdamW's scale (the update RMS is matched), follows the same schedule as lr
    min_lr_frac: float = 0.0  # final LR as a fraction of lr (stage 1 cosine runs used 0.1)
    warmup_steps: int = 100
    schedule: str = "wsd"  # "wsd" (warmup + constant + (1-sqrt) cooldown, stage 2b) or "cosine"
    cooldown_frac: float = 0.2  # WSD: fraction of steps spent in cooldown
    save_stable_at: float = 0.0  # optional token count to save ckpt/<run>-stable.pt (with optimizer)
    cooldown_from: str = ""  # load model + optimizer from a stable ckpt and run a pure cooldown
    init_from: str = ""  # start from a ckpt's weights (fresh optimizer, usual schedule); new loop_emb / mtp stay at init
    # Weight averages kept over the last cooldown_frac of the steps (stage 2f), comma-separated: EMA decays
    # such as 0.998, or "uniform" for a plain mean. Each is scored and saved as ckpt/<run>-<name>.pt.
    tail_avg: str = ""
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
    """Cosine or WSD (warmup + constant + (1-sqrt) cooldown). cooldown_from runs a pure cooldown."""
    if cfg.cooldown_from:
        t = min(1.0, step / max(1, max_steps))
        return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * (1 - math.sqrt(t)))
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if cfg.schedule == "wsd":
        decay_steps = max(1, round(max_steps * cfg.cooldown_frac))
        decay_start = max(cfg.warmup_steps, max_steps - decay_steps)
        if step < decay_start:
            return cfg.lr
        t = min(1.0, (step - decay_start) / max(1, max_steps - decay_start))
        return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * (1 - math.sqrt(t)))
    if cfg.schedule != "cosine":
        raise ValueError(f"unknown schedule: {cfg.schedule}")
    progress = (step - cfg.warmup_steps) / max(1, max_steps - cfg.warmup_steps)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * progress)))


def make_optimizers(model, cfg):
    """AdamW for everything, or with optimizer="muon", Muon for the 2-D weights inside the blocks.

    AdamW decays only matrices (the embedding, and the blocks' weights when they are on AdamW);
    norm gains are left alone. Every optimizer group remembers its peak lr as base_lr.
    """
    if cfg.optimizer not in ("adamw", "muon"):
        raise ValueError(f"unknown optimizer: {cfg.optimizer}")
    muon = [p for n, p in model.named_parameters()
            if cfg.optimizer == "muon" and n.startswith(("blocks.", "mtp.")) and p.dim() == 2]
    rest = [p for p in model.parameters() if all(p is not q for q in muon)]
    optimizers = [torch.optim.AdamW(
        [{"params": [p for p in rest if p.dim() >= 2], "weight_decay": cfg.weight_decay},
         {"params": [p for p in rest if p.dim() < 2], "weight_decay": 0.0}],
        lr=cfg.lr, betas=(0.9, 0.95), fused=True)]
    if muon:
        optimizers.append(Muon(muon, lr=cfg.muon_lr, weight_decay=cfg.weight_decay))
    for opt in optimizers:
        for group in opt.param_groups:
            group["base_lr"] = group["lr"]
    return optimizers


def tail_averages(spec):
    """'0.99,uniform' -> {"ema0.99": 0.99, "uniform": None}: an EMA decay, or None for a plain mean."""
    names = [s.strip() for s in spec.split(",") if s.strip()]
    return {("uniform" if s == "uniform" else f"ema{s}"): (None if s == "uniform" else float(s)) for s in names}


@torch.no_grad()
def update_averages(averages, params, n):
    """Fold the current weights into every average; n counts the weights folded in so far, these included."""
    for decay, avg in averages.values():
        if n == 1:
            torch._foreach_copy_(avg, params)
        else:
            torch._foreach_lerp_(avg, params, 1.0 / n if decay is None else 1.0 - decay)


def load_weights(model, path):
    """Load a checkpoint's weights into model; only parameters the checkpoint predates may be missing."""
    state = torch.load(path, map_location=next(model.parameters()).device, weights_only=True)["model"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    new = [k for k in missing if k == "loop_emb" or k.startswith("mtp.")]
    if unexpected or len(new) < len(missing):
        raise ValueError(f"{path} does not fit this model: missing {missing}, unexpected {unexpected}")
    return new


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


def append_result(cfg, run, n_params, tokens, hours, val_bpb, val_bpb_clean, hellaswag):
    path = Path("runs/results.tsv")
    if not path.exists():
        path.write_text("run_id\tdate\tgit_sha\tchange\tparams\ttokens\twall_h\tval_bpb\tval_bpb_clean\t"
                        "hellaswag\tnotes\n")
    clean = f"{val_bpb_clean:.4f}" if val_bpb_clean is not None else ""
    row = [run, time.strftime("%Y-%m-%d"), git_sha(), cfg.note, f"{n_params / 1e6:.1f}M",
           f"{tokens / 1e6:.0f}M", f"{hours:.2f}", f"{val_bpb:.4f}", clean, f"{hellaswag:.4f}", ""]
    with open(path, "a") as f:
        f.write("\t".join(row) + "\n")


def main():
    cfg = parse_args()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed if cfg.data_seed < 0 else cfg.data_seed)
    loop_rng = random.Random(cfg.seed)  # its own generator, so the data order does not depend on loop_min
    torch.set_float32_matmul_precision("high")
    device = "cuda"

    data = TokenData(cfg.data_dir)
    tok = load_tokenizer(data.tokenizer_path)
    token_bytes = token_byte_lengths(tok).to(device)
    mcfg = GPTConfig(vocab_size=tok.get_vocab_size(), n_layer=cfg.n_layer, n_head=cfg.n_head,
                     n_kv_head=cfg.n_kv_head, d_model=cfg.d_model, d_ff=cfg.d_ff,
                     max_seq_len=cfg.seq_len, arch=cfg.arch, qk_norm=cfg.qk_norm,
                     zero_init=cfg.zero_init, emb_std=cfg.emb_std, n_loop=cfg.n_loop,
                     n_prelude=cfg.n_prelude, n_coda=cfg.n_coda, mtp=cfg.mtp)
    model = GPT(mcfg).to(device)
    n_params, flops_per_token = model.num_params(), model.flops_per_token(cfg.seq_len)
    if cfg.loop_min:  # the mean over the pass counts drawn in training
        loops = range(cfg.loop_min, cfg.n_loop + 1)
        flops_per_token = sum(model.flops_per_token(cfg.seq_len, r) for r in loops) / len(loops)

    if cfg.init_from:
        assert not cfg.cooldown_from, "init_from and cooldown_from do not combine"
        new = load_weights(model, cfg.init_from)
        print(f"init_from {cfg.init_from}" + (f" (new: {', '.join(new)})" if new else ""))
    params = list(model.parameters())
    optimizers = make_optimizers(model, cfg)
    if cfg.cooldown_from:
        ckpt = torch.load(cfg.cooldown_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            states = ckpt["optimizer"]
            for opt, state in zip(optimizers, [states] if isinstance(states, dict) else states, strict=True):
                opt.load_state_dict(state)
        print(f"cooldown_from {cfg.cooldown_from} (step {ckpt.get('step', '?')}, tokens {ckpt.get('tokens', '?')})")
    train_model = torch.compile(model) if cfg.compile else model
    averages = {name: (decay, [torch.empty_like(p) for p in params])
                for name, decay in tail_averages(cfg.tail_avg).items()}

    accum, rest = divmod(cfg.batch_tokens, cfg.micro_batch * cfg.seq_len)
    assert accum > 0 and rest == 0, "batch_tokens must be a multiple of micro_batch * seq_len"
    max_steps = int(cfg.max_tokens // cfg.batch_tokens)
    tail_start = 0 if cfg.cooldown_from else int(max_steps * (1 - cfg.cooldown_frac))

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
    t_start, val_bpb, val_bpb_clean, saved_stable = time.time(), float("nan"), None, False
    for step in range(max_steps + 1):
        last = step == max_steps
        if step % cfg.eval_every == 0 or last:
            with autocast:
                val_bpb, val_bpb_clean = evaluate_bpb(train_model, data, token_bytes, cfg.micro_batch,
                                                      cfg.seq_len, cfg.eval_tokens)
            clean = {} if val_bpb_clean is None else {"val_bpb_clean": val_bpb_clean}
            record(step=step, tokens=step * cfg.batch_tokens, val_bpb=val_bpb, **clean)
        if (step > 0 and step % cfg.sample_every == 0) or last:
            with autocast:
                write_samples(out / "samples.md", step, model, tok, prompts)
        if last:
            break

        lr = lr_at(step, cfg, max_steps)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["base_lr"] * lr / cfg.lr
        if step % cfg.log_every == 0:
            torch.cuda.synchronize()  # the CPU runs ahead of the GPU: drain earlier steps so only this one is timed
        t0 = time.time()
        loss_sum, mtp_sum = torch.zeros((), device=device), torch.zeros((), device=device)
        for _ in range(accum):
            x, y = fixed or data.random_batch(cfg.micro_batch, cfg.seq_len, device)
            loop = {"n_loop": loop_rng.randint(cfg.loop_min, cfg.n_loop)} if cfg.loop_min else {}
            with autocast:
                if cfg.mtp:
                    loss, *mtp_losses = train_model(x, y, mtp=True, **loop)
                    mtp_loss = sum(mtp_losses) / cfg.mtp
                    total = loss + cfg.mtp_weight * mtp_loss
                    mtp_sum += mtp_loss.detach()
                else:
                    loss = total = train_model(x, y, **loop)
            (total / accum).backward()
            loss_sum += loss.detach()  # the next-token loss alone, comparable with runs without MTP
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        for opt in optimizers:
            opt.step()
            opt.zero_grad(set_to_none=True)
        if averages and step >= tail_start:
            update_averages(averages, params, step - tail_start + 1)
        done_tokens = (step + 1) * cfg.batch_tokens
        if not saved_stable and cfg.save_stable_at > 0 and done_tokens >= cfg.save_stable_at:
            Path("ckpt").mkdir(exist_ok=True)
            stable_path = Path("ckpt") / f"{run}-stable.pt"
            torch.save({"model": model.state_dict(), "optimizer": [opt.state_dict() for opt in optimizers],
                        "model_config": asdict(mcfg), "train_config": asdict(cfg),
                        "step": step + 1, "tokens": done_tokens}, stable_path)
            print(f"saved stable checkpoint to {stable_path} ({done_tokens / 1e6:.1f}M tokens)")
            saved_stable = True

        if step % cfg.log_every == 0:
            torch.cuda.synchronize()
            tok_per_s = cfg.batch_tokens / (time.time() - t0)
            extra = {"mtp_loss": (mtp_sum / accum).item()} if cfg.mtp else {}
            record(step=step, tokens=(step + 1) * cfg.batch_tokens, loss=(loss_sum / accum).item(), **extra,
                   lr=lr, grad_norm=grad_norm.item(), tok_per_s=round(tok_per_s),
                   mfu=tok_per_s * flops_per_token / (cfg.peak_tflops * 1e12))

    hours = (time.time() - t_start) / 3600
    Path("ckpt").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": asdict(mcfg),
                "train_config": asdict(cfg), "step": max_steps}, Path("ckpt") / f"{run}.pt")
    bench = run_benchmarks(model, tok, save_items_path=Path("runs/bench") / f"{run}.npz")  # about 30 s for v0
    append_bench_row(run, bench)
    hellaswag = bench["hellaswag"]["acc_norm"]
    append_result(cfg, run, n_params, max_steps * cfg.batch_tokens, hours, val_bpb, val_bpb_clean, hellaswag)
    for name, (_, avg) in averages.items():  # score each average in place of the final weights
        avg_run = f"{run}-{name}"
        with torch.no_grad():
            torch._foreach_copy_(params, avg)
        with autocast:
            bpb, bpb_clean = evaluate_bpb(train_model, data, token_bytes, cfg.micro_batch, cfg.seq_len,
                                          cfg.eval_tokens)
        torch.save({"model": model.state_dict(), "model_config": asdict(mcfg),
                    "train_config": asdict(replace(cfg, run=avg_run)), "step": max_steps},
                   Path("ckpt") / f"{avg_run}.pt")
        avg_bench = run_benchmarks(model, tok, save_items_path=Path("runs/bench") / f"{avg_run}.npz")
        append_bench_row(avg_run, avg_bench)
        note = "; ".join(filter(None, [cfg.note, f"tail average {name} of {run}"]))
        append_result(replace(cfg, note=note), avg_run, n_params,
                      max_steps * cfg.batch_tokens, hours, bpb, bpb_clean, avg_bench["hellaswag"]["acc_norm"])
        print(f"tail average {name}: val_bpb {bpb:.4f} ({bpb - val_bpb:+.4f} vs the final weights)")
    print(f"done in {hours:.2f} h, val_bpb {val_bpb:.4f}, hellaswag {hellaswag:.4f}")


if __name__ == "__main__":
    main()
