"""Score OpenAI GPT-2 124M on our zero-shot harness, as the reference the 50M model is chasing.

Writes the same outputs as `pinwheel.bench` does for our checkpoints: a row in runs/bench.tsv and
per-item scores in runs/bench/gpt2-124m.npz, so the experiment tree and `bench.py --compare` can use it.
GPT-2's own <|endoftext|> stands in for our BOS token.

GPT-2 runs in fp32, the precision it was trained in. Under the bf16 autocast that bench.py uses for our
models its HellaSwag rises from 29.89 to 30.26; our bf16-trained models move by under 0.1 point between
the two precisions (measured 2026-11-26), so fp32 here keeps the comparison fair.

    uv run --with transformers python scripts/gpt2_ref.py [--suite tier2]
"""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

from pinwheel.bench import SUITES, append_bench_row, centered_score, main_score, run_benchmarks

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=list(SUITES), default="tier2")
    args = p.parse_args()

    model = GPT2Model().cuda()
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
