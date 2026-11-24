"""Validation bits-per-byte: the main metric, comparable across tokenizers."""

import math

import torch


@torch.no_grad()
def evaluate_bpb(model, data, token_bytes, B, T, n_tokens):
    """Total loss in bits over the fixed val windows, divided by the UTF-8 bytes they cover."""
    device = token_bytes.device
    nats = torch.zeros((), device=device, dtype=torch.float64)
    nbytes = torch.zeros((), device=device, dtype=torch.int64)
    for x, y in data.val_batches(B, T, n_tokens, device):
        loss = model(x, y, reduction="none")
        b = token_bytes[y.reshape(-1)]
        nats += (loss * (b > 0)).sum()  # special tokens (document boundaries) cover no bytes
        nbytes += b.sum()
    return (nats / (math.log(2) * nbytes)).item()
