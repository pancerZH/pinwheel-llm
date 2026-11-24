"""Train the tokenizer and tokenize datasets into data/tok<V>/<dataset>/{train,val}.bin.

    uv run python scripts/prepare.py tokenizer                       # data/tok8192/tokenizer.json
    uv run python scripts/prepare.py tinystories                     # about 0.5B tokens
    uv run python scripts/prepare.py fineweb-edu --max_tokens 1e9

Parquet files are downloaded one at a time into the Hugging Face cache, so later runs
(e.g. other vocab sizes) read them from disk.
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

from pinwheel.data import DTYPE
from pinwheel.tokenizer import BOS, load_tokenizer, train_tokenizer

FINEWEB_EDU = ("HuggingFaceFW/fineweb-edu", "sample/10BT/")
TINYSTORIES_TRAIN = ("roneneldan/TinyStories", "data/train-")
TINYSTORIES_VAL = ("roneneldan/TinyStories", "data/validation-")
FINEWEB_VAL_DOCS = 10_000  # the first FineWeb-Edu documents are the fixed validation set


def parquet_texts(source):
    """The "text" column of every parquet file under a prefix of a dataset repo, in file order."""
    repo, prefix = source
    files = sorted(f for f in HfApi().list_repo_files(repo, repo_type="dataset")
                   if f.startswith(prefix) and f.endswith(".parquet"))
    for name in files:
        path = hf_hub_download(repo, name, repo_type="dataset")
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=["text"]):
            yield from batch.column("text").to_pylist()


def write_split(path, texts, tok, max_tokens=None):
    """Tokenize texts, each starting with BOS, into one flat file. Returns the token count."""
    bos, n = tok.token_to_id(BOS), 0
    with open(path, "wb") as f:
        for batch in itertools.batched(texts, 1024):
            ids = np.concatenate([np.array([bos] + e.ids, dtype=DTYPE) for e in tok.encode_batch(batch)])
            if max_tokens is not None:
                ids = ids[: max_tokens - n]
            ids.tofile(f)
            n += len(ids)
            print(f"\r{path}: {n / 1e6:,.1f}M tokens", end="", flush=True)
            if max_tokens is not None and n >= max_tokens:
                break
    print()
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["tokenizer", "tinystories", "fineweb-edu"])
    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--docs", type=int, default=200_000, help="tokenizer: FineWeb-Edu documents to train on")
    p.add_argument("--max_tokens", type=float, default=1e9, help="fineweb-edu: train tokens to write")
    args = p.parse_args()
    root = Path("data") / f"tok{args.vocab_size}"

    if args.what == "tokenizer":
        # Skip the validation documents so the tokenizer never sees them.
        texts = itertools.islice(parquet_texts(FINEWEB_EDU), FINEWEB_VAL_DOCS, FINEWEB_VAL_DOCS + args.docs)
        tok = train_tokenizer(texts, args.vocab_size)
        root.mkdir(parents=True, exist_ok=True)
        tok.save(str(root / "tokenizer.json"))
        print(f"wrote {root / 'tokenizer.json'}")
        return

    assert args.vocab_size <= np.iinfo(DTYPE).max + 1, "vocab too large for the token dtype"
    tok = load_tokenizer(root / "tokenizer.json")
    out = root / args.what
    out.mkdir(parents=True, exist_ok=True)
    if args.what == "tinystories":
        n_val = write_split(out / "val.bin", parquet_texts(TINYSTORIES_VAL), tok)
        n_train = write_split(out / "train.bin", parquet_texts(TINYSTORIES_TRAIN), tok)
    else:
        texts = parquet_texts(FINEWEB_EDU)  # one stream: the first documents go to val, the rest to train
        n_val = write_split(out / "val.bin", itertools.islice(texts, FINEWEB_VAL_DOCS), tok)
        n_train = write_split(out / "train.bin", texts, tok, int(args.max_tokens))
    meta = {"dataset": args.what, "vocab_size": args.vocab_size, "train_tokens": n_train, "val_tokens": n_val}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


if __name__ == "__main__":
    main()
