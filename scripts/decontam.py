"""Mark the val tokens that are part of a 64-token span found verbatim in a training file.

    uv run python scripts/decontam.py data/tok8192/fineweb-edu-4b --out data/tok8192/fineweb-edu data/tok8192/fineweb-edu-4b
    uv run python scripts/decontam.py data/tok8192/tinystories

Writes val_overlap.npy (one bool per token of the evaluated val region) into each output directory.
The 4B FineWeb train file starts with the 1B one, so its mask covers every FineWeb run. Eval windows
that contain a marked token are left out of the auxiliary val_bpb_clean (pinwheel/eval.py).
"""

import argparse
from pathlib import Path

import numpy as np

SPAN = 64
MULT = np.uint64(0x9E3779B97F4A7C15)
POW = {}  # MULT ** n (mod 2^64) for the doubling steps
with np.errstate(over="ignore"):
    for n in (1, 2, 4, 8, 16, 32):
        p = np.uint64(1)
        for _ in range(n):
            p = p * MULT
        POW[n] = p


def span_hashes(tokens):
    """A 64-bit hash of every 64-token span: combine halves 1 -> 2 -> 4 -> ... -> 64 tokens long."""
    h, n = tokens.astype(np.uint64), 1
    with np.errstate(over="ignore"):
        while n < SPAN:
            h = h[:-n] * POW[n] + h[n:]
            n *= 2
    return h


def overlap_mask(val, train, chunk=1 << 26):
    val_h = span_hashes(val)
    order = np.argsort(val_h, kind="stable")
    keys, first_val = np.unique(val_h[order], return_index=True)
    first_val = order[first_val]  # a val position for every distinct hash
    found = np.zeros(len(keys), bool)
    for start in range(0, len(train) - SPAN + 1, chunk):
        part = np.asarray(train[start : start + chunk + SPAN - 1])
        h = span_hashes(part)
        i = np.searchsorted(keys, h).clip(max=len(keys) - 1)
        hit = np.flatnonzero(keys[i] == h)
        new, first = np.unique(i[hit], return_index=True)
        keep = ~found[new]
        # Confirm each new hash on the tokens themselves (one train and one val occurrence).
        for k, p in zip(new[keep], hit[first[keep]]):
            v = first_val[k]
            if np.array_equal(part[p : p + SPAN], val[v : v + SPAN]):
                found[k] = True
        print(f"  {min(start + chunk, len(train)) / 1e9:.2f}B / {len(train) / 1e9:.2f}B tokens, "
              f"{found.sum()} matching spans", flush=True)
    starts = np.flatnonzero(np.isin(val_h, keys[found]))
    mask = np.zeros(len(val), bool)
    for s in starts:
        mask[s : s + SPAN] = True
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train_dir", help="dataset dir whose train.bin is scanned (its val.bin is the one marked)")
    ap.add_argument("--out", nargs="*", help="dirs to write val_overlap.npy to (default: train_dir)")
    ap.add_argument("--val_tokens", type=int, default=2097152 + 1, help="the evaluated val region")
    args = ap.parse_args()
    d = Path(args.train_dir)
    val = np.array(np.memmap(d / "val.bin", dtype=np.uint16, mode="r")[: args.val_tokens])
    train = np.memmap(d / "train.bin", dtype=np.uint16, mode="r")
    mask = overlap_mask(val, train)
    print(f"{mask.mean() * 100:.2f}% of {len(val)} val tokens lie in a verbatim {SPAN}-token match")
    for out in args.out or [args.train_dir]:
        np.save(Path(out) / "val_overlap.npy", mask)
        print(f"wrote {Path(out) / 'val_overlap.npy'}")


if __name__ == "__main__":
    main()
