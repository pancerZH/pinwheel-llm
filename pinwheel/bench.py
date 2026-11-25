"""Zero-shot benchmarks with discrete + continuous probe metrics and paired tests.

    uv run python -m pinwheel.bench ckpt/<run>.pt [--suite tier1|tier2] [--tasks ...] [--limit 500]
    uv run python -m pinwheel.bench --compare runs/bench/<a>.npz runs/bench/<b>.npz

Multiple choice is scored by the log-probability of each answer given the question:
`acc` picks the highest total log-prob; `acc_norm` divides by UTF-8 bytes first (tokenizer-independent).
Continuous probes (`prob`, `margin`, `brier`, `target_bpb`) come from the same forward pass at zero
extra cost and are far more sensitive than 0/1 accuracy for 50M models.
Per-item scores are saved to runs/bench/<name>.npz for paired significance testing (McNemar + paired t).
"""

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

from pinwheel.tokenizer import BOS

CHANCE = {
    "hellaswag": 0.25,
    "arc_easy": 0.25,
    "piqa": 0.5,
    "blimp": 0.5,
    "sciq": 0.25,
    "lambada": 0.0,
}
SUITES = {
    "tier1": ("hellaswag", "arc_easy", "piqa"),
    "default": ("hellaswag", "arc_easy", "piqa", "blimp"),
    "tier2": ("hellaswag", "arc_easy", "piqa", "blimp", "sciq", "lambada"),
}
BLIMP_GROUPS = 67  # 67 grammatical phenomena, each 1,000 minimal pairs
# Short answers after "Answer:": target_bpb is dominated by the first answer token, whose cost reflects
# the model's format prior (it expects " The ...") and moves all options together. Its seed std is ~0.035,
# about 20x that of val BPB, so the paired p-value overstates it; judge these tasks by acc / prob.
SHORT_ANSWER_TASKS = ("arc_easy", "sciq")


def read_parquet(repo, name):
    return pq.read_table(hf_hub_download(repo, name, repo_type="dataset")).to_pylist()


# Each loader returns (context, [answers], index of the right answer) triples.
# Single-answer tasks (lambada) have len(answers) == 1 and label == 0.
def hellaswag(limit=None):
    rows = read_parquet("Rowan/hellaswag", "data/validation-00000-of-00001.parquet")  # 10,042
    return [(r["ctx"], [" " + e for e in r["endings"]], int(r["label"])) for r in rows[:limit]]


def arc_easy(limit=None):
    rows = read_parquet("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet")  # 2,376
    return [
        (f"Question: {r['question']}\nAnswer:", [" " + t for t in r["choices"]["text"]],
         r["choices"]["label"].index(r["answerKey"]))
        for r in rows[:limit]
    ]


def piqa(limit=None):
    rows = read_parquet("baber/piqa", "piqa_validation.parquet")  # 1,838
    return [(f"Question: {r['goal']}\nAnswer:", [" " + r["sol1"], " " + r["sol2"]], r["label"])
            for r in rows[:limit]]


def stratified_takes(limit, n_groups):
    """Evenly split `limit` across `n_groups` strata so every stratum is represented."""
    q, rem = divmod(limit, n_groups)
    return [q + (i < rem) for i in range(n_groups)]


def blimp(limit=None):
    """67 phenomena x 1,000 minimal pairs; limit uses stratified sampling across all 67 phenomena."""
    files = sorted(f for f in HfApi().list_repo_files("nyu-mll/blimp", repo_type="dataset")
                   if f.endswith(".parquet"))
    if limit is None:
        return [("", [r["sentence_good"], r["sentence_bad"]], 0)
                for f in files for r in read_parquet("nyu-mll/blimp", f)]
    items = []
    for f, take in zip(files, stratified_takes(limit, len(files))):
        if take > 0:
            rows = read_parquet("nyu-mll/blimp", f)[:take]
            items.extend(("", [r["sentence_good"], r["sentence_bad"]], 0) for r in rows)
    return items


def sciq(limit=None):
    rows = read_parquet("allenai/sciq", "data/test-00000-of-00001.parquet")  # 1,000
    items = []
    for r in rows[:limit]:
        sup = r["support"].strip()
        ctx = (f"{sup}\n" if sup else "") + f"Question: {r['question']}\nAnswer:"
        answers = [" " + r["distractor1"], " " + r["distractor2"], " " + r["distractor3"], " " + r["correct_answer"]]
        items.append((ctx, answers, 3))
    return items


def lambada(limit=None):
    """LAMBADA-OpenAI (en test, 5,153): predict the final word of a passage."""
    rows = read_parquet("EleutherAI/lambada_openai", "en/test/en.parquet")
    items = []
    for r in rows[:limit]:
        ctx, last = r["text"].strip().rsplit(" ", 1)
        items.append((ctx, [" " + last], 0))
    return items


TASKS = {
    "hellaswag": hellaswag,
    "arc_easy": arc_easy,
    "piqa": piqa,
    "blimp": blimp,
    "sciq": sciq,
    "lambada": lambada,
}


@torch.no_grad()
def answer_logprobs(model, tok, pairs, batch_size=64, return_exact=False):
    """Total log-prob of each answer given its context: [BOS] context answer -> sum over answer tokens."""
    device = next(model.parameters()).device
    bos, max_len = tok.token_to_id(BOS), model.cfg.max_seq_len
    seqs = []
    for ctx, ans in pairs:
        a = tok.encode(ans).ids[: max_len - 1]
        c = ([bos] + tok.encode(ctx).ids)[-(max_len - len(a)) :]  # keep the end of a long context
        seqs.append((c + a, len(a)))
    out = torch.empty(len(seqs), dtype=torch.float64)
    exact = torch.empty(len(seqs), dtype=torch.bool) if return_exact else None
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
        m = mask[:, :-1]
        out[idx] = (tok_logp * m).sum(1).double().cpu()
        if return_exact:
            # Greedy decoding matches all target tokens iff teacher-forced argmax matches every masked token.
            exact[idx] = ((logp[:, :-1].argmax(-1) == x[:, 1:]) | ~m).all(1).cpu()
    return (out, exact) if return_exact else out


def score_item(scores, answers, label, exact=False, use_norm=True):
    """Compute discrete + continuous probe metrics for one question from raw answer log-probs."""
    n_bytes = [len(a.encode()) for a in answers]
    target_bpb = -scores[label] / (math.log(2) * n_bytes[label])
    if len(answers) == 1:  # open-ended single target (e.g. LAMBADA)
        prob = math.exp(min(0.0, scores[0]))
        return {
            "acc": int(exact),
            "acc_norm": int(exact),
            "prob": prob,
            "margin": float("nan"),
            "brier": (1.0 - prob) ** 2,
            "target_bpb": target_bpb,
        }
    norm = [s / b for s, b in zip(scores, n_bytes)]
    acc = int(max(range(len(answers)), key=scores.__getitem__) == label)
    acc_norm = int(max(range(len(answers)), key=norm.__getitem__) == label)
    primary = norm if use_norm else scores
    m = max(primary)
    exps = [math.exp(x - m) for x in primary]
    z = sum(exps)
    probs = [e / z for e in exps]
    prob = probs[label]
    best_wrong = max(primary[j] for j in range(len(primary)) if j != label)
    brier = sum((p - (1.0 if j == label else 0.0)) ** 2 for j, p in enumerate(probs))
    return {
        "acc": acc,
        "acc_norm": acc_norm,
        "prob": prob,
        "margin": primary[label] - best_wrong,
        "brier": brier,
        "target_bpb": target_bpb,
    }


def run_task(model, tok, items, use_norm=True, return_items=False):
    pairs = [(ctx, a) for ctx, answers, _ in items for a in answers]
    need_exact = any(len(answers) == 1 for _, answers, _ in items)
    if need_exact:
        logp_t, exact_t = answer_logprobs(model, tok, pairs, return_exact=True)
        logp, exacts = logp_t.tolist(), exact_t.tolist()
    else:
        logp = answer_logprobs(model, tok, pairs).tolist()
        exacts = [False] * len(pairs)

    n = len(items)
    acc = np.empty(n, dtype=np.uint8)
    acc_norm = np.empty(n, dtype=np.uint8)
    prob = np.empty(n, dtype=np.float32)
    margin = np.empty(n, dtype=np.float32)
    brier = np.empty(n, dtype=np.float32)
    target_bpb = np.empty(n, dtype=np.float32)
    k = 0
    for i, (_, answers, label) in enumerate(items):
        m = len(answers)
        r = score_item(logp[k : k + m], answers, label, exact=exacts[k], use_norm=use_norm)
        acc[i], acc_norm[i] = r["acc"], r["acc_norm"]
        prob[i], margin[i], brier[i], target_bpb[i] = r["prob"], r["margin"], r["brier"], r["target_bpb"]
        k += m

    summary = {
        "acc": float(acc.mean()),
        "acc_norm": float(acc_norm.mean()),
        "prob": float(prob.mean()),
        "margin": float("nan") if np.isnan(margin).all() else float(np.nanmean(margin)),
        "brier": float(brier.mean()),
        "target_bpb": float(target_bpb.mean()),
        "n": n,
    }
    if return_items:
        correct = acc_norm if use_norm else acc
        return summary, {"correct": correct, "prob": prob, "margin": margin, "brier": brier, "target_bpb": target_bpb}
    return summary


def run_benchmarks(model, tok, tasks=SUITES["default"], limit=None, save_items_path=None):
    """{task: {"acc", "acc_norm", "prob", "margin", "brier", "target_bpb", "n"}}."""
    model.eval()
    results, per_item = {}, {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for name in tasks:
            items = TASKS[name](limit=limit)
            use_norm = name != "blimp"
            if save_items_path is not None:
                results[name], item_data = run_task(model, tok, items, use_norm=use_norm, return_items=True)
                for field, arr in item_data.items():
                    per_item[f"{name}__{field}"] = arr
            else:
                results[name] = run_task(model, tok, items, use_norm=use_norm)
    if save_items_path is not None:
        save_items_path = Path(save_items_path)
        save_items_path.parent.mkdir(parents=True, exist_ok=True)
        if save_items_path.exists():
            with np.load(save_items_path) as old:
                merged = {k: old[k] for k in old.files}
            merged.update(per_item)
            per_item = merged
        np.savez_compressed(save_items_path, **per_item)
    return results


def main_score(name, r):
    return r["acc"] if name in ("blimp", "lambada") else r["acc_norm"]


def centered_score(name, r):
    """DCLM CORE centering: (score - chance) / (1 - chance), mapping random -> 0, perfect -> 1."""
    c = CHANCE[name]
    return (main_score(name, r) - c) / (1.0 - c)


def append_bench_row(name, results, path=Path("runs/bench.tsv")):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        lines = path.read_text().splitlines()
        cols = lines[0].split("\t")
        extra = [t for t in TASKS if t in results and t not in cols]
        if extra:
            cols.extend(extra)
            pad = "\t" * len(extra)
            lines = ["\t".join(cols)] + [line + pad for line in lines[1:]]
            path.write_text("\n".join(lines) + "\n")
    else:
        cols = ["ckpt", "date", *(t for t in TASKS if t in results or t in SUITES["default"])]
        path.write_text("\t".join(cols) + "\n")
    task_cols = cols[2:]
    cells = [f"{main_score(t, results[t]):.4f}" if t in results else "" for t in task_cols]
    with open(path, "a") as f:
        f.write("\t".join([name, time.strftime("%Y-%m-%d"), *cells]) + "\n")


def mcnemar_test(a_correct, b_correct):
    """Continuity-corrected McNemar test on paired binary outcomes; returns (n01, n10, p_value)."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    n01 = int((~a & b).sum())  # A wrong, B right
    n10 = int((a & ~b).sum())  # A right, B wrong
    disc = n01 + n10
    if disc == 0 or abs(n01 - n10) <= 1:
        return n01, n10, 1.0
    chi2 = (abs(n01 - n10) - 1.0) ** 2 / disc
    return n01, n10, math.erfc(math.sqrt(chi2 / 2.0))


def paired_diff_test(a, b, n_clusters=None):
    """Paired mean difference (b - a), SE, and two-sided normal p-value (clustered if n_clusters given)."""
    d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    if n_clusters and len(d) % n_clusters == 0:
        d = d.reshape(n_clusters, -1).mean(axis=1)
    mean = float(d.mean())
    if len(d) < 2:
        return mean, float("nan"), 1.0
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    if se == 0.0:
        return mean, 0.0, 1.0 if mean == 0.0 else 0.0
    return mean, se, math.erfc(abs(mean / se) / math.sqrt(2.0))


def compare_saved(path_a, path_b):
    """Compare two saved runs/bench/*.npz files on all shared tasks with paired significance tests."""
    da, db = np.load(path_a), np.load(path_b)
    tasks_a = {k.split("__")[0] for k in da.files}
    tasks_b = {k.split("__")[0] for k in db.files}
    shared = [t for t in TASKS if t in tasks_a and t in tasks_b]
    print(f"Paired comparison: {Path(path_a).stem} (A) -> {Path(path_b).stem} (B)")
    print(f"{'task':10s} {'Δacc':>8s} {'McNemar p':>10s} {'Δprob':>9s} {'p(prob)':>9s} {'Δbpb':>9s} {'p(bpb)':>9s}")
    for t in shared:
        ca, cb = da[f"{t}__correct"], db[f"{t}__correct"]
        if len(ca) != len(cb):
            print(f"{t:10s} skipped (mismatched n: {len(ca)} vs {len(cb)})")
            continue
        clusters = BLIMP_GROUPS if t == "blimp" and len(ca) % BLIMP_GROUPS == 0 else None
        d_acc = float(cb.mean() - ca.mean())
        _, _, p_mc = mcnemar_test(ca, cb)
        if clusters:
            # For BLiMP, cluster SE over the 67 phenomena is more honest than i.i.d. McNemar.
            _, _, p_acc = paired_diff_test(ca, cb, n_clusters=clusters)
        else:
            p_acc = p_mc
        d_prob, _, p_prob = paired_diff_test(da[f"{t}__prob"], db[f"{t}__prob"], n_clusters=clusters)
        d_bpb, _, p_bpb = paired_diff_test(da[f"{t}__target_bpb"], db[f"{t}__target_bpb"], n_clusters=clusters)
        mark = " *" if t in SHORT_ANSWER_TASKS else ""
        print(f"{t:10s} {d_acc:+8.4f} {p_acc:10.4g} {d_prob:+9.5f} {p_prob:9.4g} {d_bpb:+9.5f} {p_bpb:9.4g}{mark}")
    if any(t in SHORT_ANSWER_TASKS for t in shared):
        print("* Δbpb unreliable here: short answers make it a first-token format prior shared by all options,\n"
              "  with seed std ~0.035 that the paired p-value ignores. Judge these tasks by Δacc / Δprob.")


def main():
    from pinwheel.sample import load_checkpoint

    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="?", help="checkpoint path (omit when using --compare)")
    p.add_argument("--suite", choices=list(SUITES), help="preset task suite (overrides default tasks)")
    p.add_argument("--tasks", nargs="+", choices=list(TASKS), help="explicit list of tasks")
    p.add_argument("--limit", type=int, help="smoke-test cap per task (BLiMP is stratified across 67 phenomena)")
    p.add_argument("--compare", nargs=2, metavar=("A_NPZ", "B_NPZ"), help="paired test between two saved .npz files")
    args = p.parse_args()

    if args.compare:
        compare_saved(args.compare[0], args.compare[1])
        return
    if not args.ckpt:
        p.error("ckpt is required unless --compare is given")

    tasks = tuple(args.tasks) if args.tasks else SUITES[args.suite or "default"]
    model, tok = load_checkpoint(args.ckpt)
    name = Path(args.ckpt).stem
    save_path = Path("runs/bench") / f"{name}.npz" if args.limit is None else None
    t0 = time.time()
    results = run_benchmarks(model, tok, tasks, args.limit, save_items_path=save_path)

    print(f"{'task':10s} {'acc':>7s} {'acc_norm':>9s} {'prob':>7s} {'margin':>8s} {'brier':>7s} {'t_bpb':>7s} {'n':>6s}")
    for t, r in results.items():
        margin_s = f"{r['margin']:+8.4f}" if not math.isnan(r["margin"]) else f"{'—':>8s}"
        print(f"{t:10s} {r['acc']:7.4f} {r['acc_norm']:9.4f} {r['prob']:7.4f} {margin_s} "
              f"{r['brier']:7.4f} {r['target_bpb']:7.4f} {r['n']:6d}")
    core = sum(centered_score(t, r) for t, r in results.items()) / len(results)
    print(f"CORE centered avg: {core:+.4f}  ({time.time() - t0:.0f} s)")

    if args.limit is None:  # only full runs go into the table and per-item archive
        append_bench_row(name, results)
        print(f"saved per-item scores to {save_path}")
    else:
        print("(smoke test with --limit: not appended to runs/bench.tsv)")


if __name__ == "__main__":
    main()
