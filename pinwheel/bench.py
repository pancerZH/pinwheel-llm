"""Zero-shot benchmarks: HellaSwag, ARC-Easy, PIQA and BLiMP.

    uv run python -m pinwheel.bench ckpt/<run>.pt [--tasks hellaswag piqa] [--limit 500]

Multiple choice is scored by the log-probability of each answer given the question:
`acc` picks the highest total log-prob, `acc_norm` divides it by the answer's UTF-8 bytes
first (so long answers are not penalized, and it does not depend on the tokenizer).
BLiMP compares the total log-prob of a grammatical and an ungrammatical sentence.
Results are printed and appended to runs/bench.tsv.
"""

import argparse
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

from pinwheel.tokenizer import BOS

CHANCE = {"hellaswag": 0.25, "arc_easy": 0.25, "piqa": 0.5, "blimp": 0.5}


def read_parquet(repo, name):
    return pq.read_table(hf_hub_download(repo, name, repo_type="dataset")).to_pylist()


# Each loader returns (context, [answers], index of the right answer) triples.
def hellaswag():
    rows = read_parquet("Rowan/hellaswag", "data/validation-00000-of-00001.parquet")  # 10,042
    return [(r["ctx"], [" " + e for e in r["endings"]], int(r["label"])) for r in rows]


def arc_easy():
    rows = read_parquet("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet")  # 2,376
    return [(f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]],
             r["choices"]["label"].index(r["answerKey"])) for r in rows]


def piqa():
    rows = read_parquet("baber/piqa", "piqa_validation.parquet")  # 1,838
    return [(f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], r["label"]) for r in rows]


def blimp():
    """67 phenomena x 1,000 minimal pairs; the context is empty, the answers are whole sentences."""
    files = sorted(f for f in HfApi().list_repo_files("nyu-mll/blimp", repo_type="dataset")
                   if f.endswith(".parquet"))
    return [("", [r["sentence_good"], r["sentence_bad"]], 0) for f in files for r in read_parquet("nyu-mll/blimp", f)]


TASKS = {"hellaswag": hellaswag, "arc_easy": arc_easy, "piqa": piqa, "blimp": blimp}


@torch.no_grad()
def answer_logprobs(model, tok, pairs, batch_size=64):
    """Total log-prob of each answer given its context: [BOS] context answer -> sum over answer tokens."""
    device = next(model.parameters()).device
    bos, max_len = tok.token_to_id(BOS), model.cfg.max_seq_len
    seqs = []
    for ctx, ans in pairs:
        a = tok.encode(ans).ids[: max_len - 1]
        c = ([bos] + tok.encode(ctx).ids)[-(max_len - len(a)) :]  # keep the end of a long context
        seqs.append((c + a, len(a)))
    out = torch.empty(len(seqs), dtype=torch.float64)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))  # similar lengths, less padding
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        T = max(len(seqs[i][0]) for i in idx)
        x = torch.zeros(len(idx), T, dtype=torch.long)
        mask = torch.zeros(len(idx), T, dtype=torch.bool)  # marks the positions that predict answer tokens
        for row, i in enumerate(idx):
            ids, n = seqs[i]
            x[row, : len(ids)] = torch.tensor(ids)
            mask[row, len(ids) - n - 1 : len(ids) - 1] = True
        x, mask = x.to(device), mask.to(device)
        logp = F.log_softmax(model(x).float(), dim=-1)
        tok_logp = logp[:, :-1].gather(-1, x[:, 1:, None]).squeeze(-1)  # log p(x[t+1] | x[:t+1])
        out[idx] = (tok_logp * mask[:, :-1]).sum(1).double().cpu()
    return out


def run_task(model, tok, items):
    pairs = [(ctx, a) for ctx, answers, _ in items for a in answers]
    logp = answer_logprobs(model, tok, pairs).tolist()
    correct = correct_norm = 0
    k = 0
    for _, answers, label in items:
        scores = logp[k : k + len(answers)]
        norm = [s / len(a.encode()) for s, a in zip(scores, answers)]
        correct += max(range(len(answers)), key=scores.__getitem__) == label
        correct_norm += max(range(len(answers)), key=norm.__getitem__) == label
        k += len(answers)
    return {"acc": correct / len(items), "acc_norm": correct_norm / len(items), "n": len(items)}


def run_benchmarks(model, tok, tasks=tuple(TASKS), limit=None):
    """{task: {"acc", "acc_norm", "n"}}; the main number is acc_norm, except acc for BLiMP."""
    model.eval()
    results = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for name in tasks:
            items = TASKS[name]()[:limit]
            results[name] = run_task(model, tok, items)
    return results


def main_score(name, r):
    return r["acc"] if name == "blimp" else r["acc_norm"]


def append_bench_row(name, results):
    path = Path("runs/bench.tsv")
    if not path.exists():
        path.write_text("ckpt\tdate\t" + "\t".join(TASKS) + "\n")
    cells = [f"{main_score(t, results[t]):.4f}" if t in results else "" for t in TASKS]
    with open(path, "a") as f:
        f.write("\t".join([name, time.strftime("%Y-%m-%d"), *cells]) + "\n")


def main():
    from pinwheel.sample import load_checkpoint

    p = argparse.ArgumentParser()
    p.add_argument("ckpt")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--limit", type=int, help="only the first N items per task (quick checks)")
    args = p.parse_args()

    model, tok = load_checkpoint(args.ckpt)
    t0 = time.time()
    results = run_benchmarks(model, tok, args.tasks, args.limit)
    print(f"{'task':10s} {'acc':>7s} {'acc_norm':>9s} {'chance':>7s} {'n':>7s}")
    for name, r in results.items():
        print(f"{name:10s} {r['acc']:7.4f} {r['acc_norm']:9.4f} {CHANCE[name]:7.2f} {r['n']:7d}")
    print(f"({time.time() - t0:.0f} s)")

    if args.limit is None:  # only full runs go into the table
        append_bench_row(Path(args.ckpt).stem, results)


if __name__ == "__main__":
    main()
