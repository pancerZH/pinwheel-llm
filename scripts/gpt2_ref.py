"""Score OpenAI GPT-2 124M on our zero-shot harness, as the reference the 50M model is chasing.

Writes the same outputs as `pinwheel.bench` does for our checkpoints: a row in runs/bench.tsv and
per-item scores in runs/bench/gpt2-124m.npz, so the experiment tree and `bench.py --compare` can use it.
GPT-2's own <|endoftext|> stands in for our BOS token.

GPT-2 runs in fp32, the precision it was trained in. Under the bf16 autocast that bench.py uses for our
models its HellaSwag rises from 29.89 to 30.26; our bf16-trained models move by under 0.1 point between
the two precisions (measured 2026-11-26), so fp32 here keeps the comparison fair.

With --bpb it measures val BPB on exactly the text behind train.py's val BPB (the first eval_tokens of
val.bin in seq_len windows): each window is decoded, split at BOS into documents, re-tokenized by GPT-2
with <|endoftext|> between documents, and scored with the same rule (the window's first token is context
only, separators count no bytes). Writes runs/gpt2-124m/ref.json.

    uv run --with transformers python scripts/gpt2_ref.py [--suite tier2] [--bpb]
"""

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

from pinwheel.bench import SUITES, append_bench_row, centered_score, main_score, run_benchmarks
from pinwheel.data import DTYPE
from pinwheel.tokenizer import BOS, load_tokenizer

NAME = "gpt2-124m"
EOT = 50256


class GPT2Tokenizer:
    def __init__(self):
        self.tok = Tokenizer.from_pretrained("gpt2")

    def encode(self, text):
        return self.tok.encode(text)

    def token_to_id(self, _token):
        return EOT


class GPT2Model(torch.nn.Module):
    """Adapter exposing what bench.py expects: model(x) -> logits and model.cfg.max_seq_len."""

    def __init__(self):
        super().__init__()
        self.hf = GPT2LMHeadModel.from_pretrained("gpt2")
        self.cfg = SimpleNamespace(max_seq_len=self.hf.config.n_positions)

    def forward(self, x):
        with torch.autocast("cuda", enabled=False):
            return self.hf(x).logits


def val_windows(data_dir, n_tokens, seq_len):
    """Documents inside each val BPB window of train.py, as text: windows start every seq_len tokens and
    hold seq_len + 1 tokens (inputs plus the last target); BOS splits documents."""
    tok = load_tokenizer(Path(data_dir).parent / "tokenizer.json")
    bos = tok.token_to_id(BOS)
    data = np.memmap(Path(data_dir) / "val.bin", dtype=DTYPE, mode="r")
    for start in range(0, min(n_tokens, len(data) - 1) // seq_len * seq_len, seq_len):
        docs, cur = [], []
        for i in data[start : start + seq_len + 1].tolist():
            if i == bos:
                docs.append(cur)
                cur = []
            else:
                cur.append(i)
        docs.append(cur)
        yield [tok.decode(d) for d in docs]


@torch.no_grad()
def retokenized_bpb(model, encode, sep_id, token_bytes, windows, max_len, batch_size=8):
    """BPB of `model` over the window texts, re-tokenized with `encode` and `sep_id` between documents.
    Every token after the first is predicted; separators count no bytes. Windows longer than max_len
    are scored in two overlapping passes, each target once."""
    device = next(model.parameters()).device
    pieces = []  # (token ids, index of the first target to count)
    for docs in windows:
        ids = []
        for k, doc in enumerate(docs):
            if k:
                ids.append(sep_id)
            ids.extend(encode(doc))
        if len(ids) <= max_len:
            pieces.append((ids, 1))
        else:
            pieces += [(ids[:max_len], 1), (ids[-max_len:], max_len - (len(ids) - max_len))]
    nats, nbytes = 0.0, 0
    for b in range(0, len(pieces), batch_size):
        batch = pieces[b : b + batch_size]
        T = max(len(ids) for ids, _ in batch)
        x = torch.zeros(len(batch), T, dtype=torch.long)
        keep = torch.zeros(len(batch), T - 1, dtype=torch.bool)  # targets x[:, 1:] that count
        for row, (ids, first) in enumerate(batch):
            x[row, : len(ids)] = torch.tensor(ids)
            keep[row, first - 1 : len(ids) - 1] = True
        x, keep = x.to(device), keep.to(device)
        y = x[:, 1:]
        loss = F.cross_entropy(model(x)[:, :-1].float().transpose(1, 2), y, reduction="none")
        b_y = token_bytes[y] * keep
        nats += (loss * (b_y > 0)).sum().item()
        nbytes += b_y.sum().item()
    return nats / (math.log(2) * nbytes), nbytes


def gpt2_token_bytes(tok):
    """Bytes per GPT-2 token (byte-level BPE: one character per byte); 0 for <|endoftext|>."""
    n = tok.get_vocab_size()
    lengths = [0 if i == EOT else len(tok.id_to_token(i)) for i in range(n)]
    return torch.tensor(lengths, dtype=torch.int64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=list(SUITES), default="tier2")
    p.add_argument("--bpb", action="store_true", help="measure val BPB instead of the benchmarks")
    p.add_argument("--data_dir", default="data/tok8192/fineweb-edu")
    p.add_argument("--eval_tokens", type=int, default=2097152, help="train.py's eval_tokens")
    p.add_argument("--seq_len", type=int, default=1024, help="train.py's seq_len")
    args = p.parse_args()

    model = GPT2Model().cuda()
    if args.bpb:
        tok = GPT2Tokenizer().tok
        t0 = time.time()
        windows = val_windows(args.data_dir, args.eval_tokens, args.seq_len)
        bpb, nbytes = retokenized_bpb(
            model, lambda s: tok.encode(s).ids, EOT, gpt2_token_bytes(tok).cuda(), windows, model.cfg.max_seq_len
        )
        print(f"val BPB {bpb:.4f} over {nbytes:,} bytes ({time.time() - t0:.0f} s)")
        out = Path("runs") / NAME / "ref.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "val_bpb": round(bpb, 4), "bytes": nbytes, "data_dir": args.data_dir,
            "eval_tokens": args.eval_tokens, "seq_len": args.seq_len, "date": time.strftime("%Y-%m-%d"),
        }, indent=2) + "\n")
        return

    tasks = SUITES[args.suite]
    t0 = time.time()
    # A 50k-token vocabulary makes logits ~6x larger than ours: smaller batches fit in 16 GB.
    results = run_benchmarks(
        model, GPT2Tokenizer(), tasks, save_items_path=Path("runs/bench") / f"{NAME}.npz", batch_size=8
    )
    for t, r in results.items():
        print(f"{t:10s} {main_score(t, r):.4f}  (n={r['n']})")
    core = sum(centered_score(t, r) for t, r in results.items()) / len(results)
    print(f"CORE centered avg: {core:+.4f}  ({time.time() - t0:.0f} s)")
    append_bench_row(NAME, results)


if __name__ == "__main__":
    main()
