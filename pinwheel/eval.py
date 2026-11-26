"""Validation bits-per-byte: the main metric, comparable across tokenizers.

The auxiliary val_bpb_clean counts only the windows with no 64-token span found verbatim in the
training text (scripts/decontam.py marks them). Score checkpoints with:

    uv run python -m pinwheel.eval ckpt/<run>.pt [...] --data_dir data/tok8192/fineweb-edu
"""

import math

import torch


@torch.no_grad()
def evaluate_bpb(model, data, token_bytes, B, T, n_tokens):
    """Total loss in bits over the fixed val windows, divided by the UTF-8 bytes they cover.

    Returns (val_bpb, val_bpb_clean); the second is None without a val_overlap mask.
    """
    device = token_bytes.device
    nats = torch.zeros((), device=device, dtype=torch.float64)
    nbytes = torch.zeros((), device=device, dtype=torch.int64)
    clean_nats, clean_bytes, masked = nats.clone(), nbytes.clone(), False
    for x, y, clean in data.val_batches(B, T, n_tokens, device):
        loss = model(x, y, reduction="none")
        b = token_bytes[y.reshape(-1)]
        nats += (loss * (b > 0)).sum()  # special tokens (document boundaries) cover no bytes
        nbytes += b.sum()
        if clean is not None:
            masked = True
            keep = clean.repeat_interleave(T)
            clean_nats += (loss * (b > 0) * keep).sum()
            clean_bytes += (b * keep).sum()
    bpb = (nats / (math.log(2) * nbytes)).item()
    return bpb, (clean_nats / (math.log(2) * clean_bytes)).item() if masked else None


def main():
    import argparse

    from pinwheel.data import TokenData
    from pinwheel.model import GPT, GPTConfig
    from pinwheel.tokenizer import load_tokenizer, token_byte_lengths

    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--data_dir", default="data/tok8192/fineweb-edu")
    ap.add_argument("--eval_tokens", type=int, default=2097152)
    args = ap.parse_args()
    torch.set_float32_matmul_precision("high")
    data = TokenData(args.data_dir)
    token_bytes = token_byte_lengths(load_tokenizer(data.tokenizer_path)).to("cuda")
    for path in args.ckpts:
        ckpt = torch.load(path, map_location="cuda", weights_only=True)
        model = GPT(GPTConfig(**ckpt["model_config"])).cuda().eval()
        model.load_state_dict(ckpt["model"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bpb, clean = evaluate_bpb(model, data, token_bytes, 32, model.cfg.max_seq_len, args.eval_tokens)
        print(f"{path}\tval_bpb {bpb:.4f}\tval_bpb_clean {clean:.4f}" if clean is not None else f"{path}\tval_bpb {bpb:.4f}")


if __name__ == "__main__":
    main()
