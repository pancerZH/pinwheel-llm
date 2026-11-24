"""Sample from a checkpoint.

    uv run python -m pinwheel.sample ckpt/<run>.pt --prompt "Once upon a time"
    uv run python -m pinwheel.sample ckpt/<run>.pt          # interactive
"""

import argparse
from pathlib import Path

import torch

from pinwheel.model import GPT, GPTConfig
from pinwheel.tokenizer import BOS, load_tokenizer


def complete(model, tok, prompt, max_new_tokens=200, temperature=0.8, top_k=50, top_p=None,
             generator=None, stream=False):
    """Continue a prompt; stops early if the model starts a new document (BOS)."""
    device = next(model.parameters()).device
    bos = tok.token_to_id(BOS)
    ids = ([bos] + tok.encode(prompt).ids)[-(model.cfg.max_seq_len - 1) :]
    out, shown = [], ""
    idx = torch.tensor([ids], device=device)
    for t in model.generate(idx, max_new_tokens, temperature, top_k, top_p, generator):
        if t.item() == bos:
            break
        out.append(t.item())
        if stream:
            text = tok.decode(out)
            print(text[len(shown) :], end="", flush=True)
            shown = text
    return tok.decode(out)


def load_checkpoint(path, device="cuda"):
    ckpt = torch.load(path, map_location=device)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tok = load_tokenizer(Path(ckpt["train_config"]["data_dir"]).parent / "tokenizer.json")
    return model, tok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--prompt")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--top_p", type=float)
    args = p.parse_args()

    model, tok = load_checkpoint(args.ckpt)
    kwargs = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_k=args.top_k, top_p=args.top_p, stream=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if args.prompt is not None:
            print(args.prompt, end="")
            complete(model, tok, args.prompt, **kwargs)
            print()
            return
        while True:
            try:
                prompt = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                return
            print(prompt, end="")
            complete(model, tok, prompt, **kwargs)
            print()


if __name__ == "__main__":
    main()
