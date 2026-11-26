"""Tokenized datasets on disk: data/tok<V>/<dataset>/{train,val}.bin, flat uint16 token ids."""

import json
from pathlib import Path

import numpy as np
import torch

DTYPE = np.uint16


class TokenData:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.meta = json.loads((self.dir / "meta.json").read_text())
        self.tokenizer_path = self.dir.parent / "tokenizer.json"
        # Val tokens inside a 64-token span found verbatim in training text (scripts/decontam.py).
        overlap = self.dir / "val_overlap.npy"
        self.val_overlap = np.load(overlap) if overlap.exists() else None

    def _tokens(self, split):
        # A fresh memmap on every call keeps numpy from holding on to the pages (nanoGPT trick).
        return np.memmap(self.dir / f"{split}.bin", dtype=DTYPE, mode="r")

    def random_batch(self, B, T, device):
        """B random windows of T + 1 tokens from train.bin, as (inputs, targets)."""
        data = self._tokens("train")
        starts = np.random.randint(0, len(data) - T - 1, size=B)
        x = torch.from_numpy(np.stack([data[s : s + T + 1] for s in starts]).astype(np.int64))
        x = x.pin_memory().to(device, non_blocking=True)
        return x[:, :-1], x[:, 1:]

    def val_batches(self, B, T, n_tokens, device):
        """The same windows every time: the first n_tokens of val.bin, cut into (B, T) batches.

        Also yields a (B,) bool tensor: True for a window with no token in val_overlap, or None
        when there is no mask for this part of val.
        """
        data = self._tokens("val")
        for b in range(min(n_tokens, len(data) - 1) // (B * T)):
            lo, hi = b * B * T, (b + 1) * B * T + 1
            chunk = torch.from_numpy(data[lo:hi].astype(np.int64)).to(device)
            clean = None
            if self.val_overlap is not None and hi <= len(self.val_overlap):
                ov = self.val_overlap[lo:hi]
                clean = torch.tensor([not ov[i * T : i * T + T + 1].any() for i in range(B)], device=device)
            yield chunk[:-1].view(B, T), chunk[1:].view(B, T), clean
