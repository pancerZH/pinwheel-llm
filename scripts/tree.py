"""Build a self-contained interactive experiment tree webpage (runs/tree.html).

    uv run python scripts/tree.py
    uv run python scripts/tree.py --spec tree.toml -o runs/tree.html [--serve]

Reads runs/results.tsv, runs/bench.tsv, runs/<run>/{config.json,log.jsonl,samples.md},
runs/bench/<run>.npz, ckpt/<run>.pt, and the lightweight hand-written topology file
(tree.toml). Never hand-copies metrics: every number comes from the run artifacts.
"""

from __future__ import annotations

import argparse
import csv
import functools
import http.server
import json
import math
import re
import statistics
import tomllib
import webbrowser
from pathlib import Path

import numpy as np

from pinwheel.bench import (
    BLIMP_GROUPS,
    CHANCE,
    SHORT_ANSWER_TASKS,
    mcnemar_test,
    paired_diff_test,
)

BENCH_TASKS = ("hellaswag", "arc_easy", "piqa", "blimp", "sciq", "lambada")
DIFF_FIELDS = (
    ("arch", "架构"),
    ("n_layer", "层数"),
    ("d_model", "宽度 d_model"),
    ("n_head", "Q 头数"),
    ("n_kv_head", "KV 头数"),
    ("d_ff", "FFN 宽度"),
    ("vocab_size", "词表"),
    ("data", "数据集"),
    ("max_tokens", "训练 tokens"),
    ("batch_tokens", "Batch tokens"),
    ("schedule", "LR 调度"),
    ("lr", "峰值 LR"),
    ("warmup_steps", "Warmup 步数"),
    ("cooldown_frac", "退火比例"),
    ("min_lr_frac", "最低 LR 比例"),
    ("weight_decay", "Weight decay"),
    ("seed", "Seed"),
    ("cooldown_from", "分支退火起点"),
)
PLACEHOLDER = "__PINWHEEL_TREE_DATA__"


def load_results(runs_dir: Path) -> dict[str, dict]:
    """Read runs/results.tsv into {run_id: row_dict}."""
    path = runs_dir / "results.tsv"
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rid = (row.get("run_id") or "").strip()
            if not rid:
                continue
            out[rid] = {
                "run_id": rid,
                "date": (row.get("date") or "").strip(),
                "git_sha": (row.get("git_sha") or "").strip(),
                "change": (row.get("change") or "").strip(),
                "params": (row.get("params") or "").strip(),
                "tokens": (row.get("tokens") or "").strip(),
                "wall_h": _to_float(row.get("wall_h")),
                "val_bpb": _to_float(row.get("val_bpb")),
                "notes": (row.get("notes") or "").strip(),
            }
    return out


def load_bench(runs_dir: Path) -> dict[str, dict]:
    """Read runs/bench.tsv, merging duplicate rows per run (latest non-empty score wins)."""
    path = runs_dir / "bench.tsv"
    if not path.exists():
        return {}
    merged: dict[str, dict[str, float]] = {}
    dates: dict[str, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            rid = (row.get("ckpt") or "").strip()
            if not rid:
                continue
            scores = merged.setdefault(rid, {})
            if row.get("date"):
                dates[rid] = row["date"].strip()
            for task in BENCH_TASKS:
                val = _to_float(row.get(task))
                if val is not None:
                    scores[task] = val

    out: dict[str, dict] = {}
    for rid, scores in merged.items():
        if not scores:
            continue
        centered = [(scores[t] - CHANCE[t]) / (1.0 - CHANCE[t]) for t in scores if t in CHANCE]
        out[rid] = {
            "date": dates.get(rid, ""),
            "scores": scores,
            "core": sum(centered) / len(centered) if centered else None,
            "tier": "tier2" if ("sciq" in scores and "lambada" in scores) else "default",
        }
    return out


def extract_config_summary(cfg: dict) -> dict:
    """Normalize config.json into a compact, comparable flat dictionary."""
    t = cfg.get("train") or {}
    m = cfg.get("model") or {}
    data_dir = t.get("data_dir") or ""
    cd_from = t.get("cooldown_from") or ""
    return {
        "arch": t.get("arch") or m.get("arch") or "llama",
        "n_layer": m.get("n_layer", t.get("n_layer")),
        "d_model": m.get("d_model", t.get("d_model")),
        "n_head": m.get("n_head", t.get("n_head")),
        "n_kv_head": m.get("n_kv_head", t.get("n_kv_head")),
        "d_ff": m.get("d_ff", t.get("d_ff")),
        "vocab_size": m.get("vocab_size"),
        "data": Path(data_dir).name if data_dir else "",
        "max_tokens": _fmt_tokens(t.get("max_tokens")),
        "batch_tokens": t.get("batch_tokens"),
        "schedule": t.get("schedule") or "cosine",
        "lr": t.get("lr"),
        "warmup_steps": t.get("warmup_steps"),
        "cooldown_frac": t.get("cooldown_frac") if (t.get("schedule") or "cosine") == "wsd" else None,
        "min_lr_frac": t.get("min_lr_frac"),
        "weight_decay": t.get("weight_decay"),
        "seed": t.get("seed"),
        "cooldown_from": Path(cd_from).name if cd_from else "",
    }


def compute_config_diff(cur: dict | None, ref: dict | None) -> list[dict]:
    """Return human-readable differences between two normalized config summaries."""
    if not cur or not ref:
        return []
    diffs = []
    for key, label in DIFF_FIELDS:
        a, b = ref.get(key), cur.get(key)
        if a == b:
            continue
        diffs.append({"key": key, "label": label, "from": _fmt_val(a), "to": _fmt_val(b)})
    return diffs


def parse_log(log_path: Path) -> dict:
    """Extract val_bpb curve points and steady-state throughput/MFU from log.jsonl."""
    if not log_path.exists():
        return {"val_curve": [], "tok_per_s": None, "mfu": None, "last_step": None, "last_tokens_m": None}
    val_curve: list[list[float]] = []
    tok_speeds: list[float] = []
    mfus: list[float] = []
    last_step = None
    last_tokens = None
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = row.get("step")
            toks = row.get("tokens")
            if step is not None:
                last_step = int(step)
            if toks is not None:
                last_tokens = float(toks)
            if "val_bpb" in row and toks is not None:
                val_curve.append([round(float(toks) / 1e6, 2), round(float(row["val_bpb"]), 5)])
            if "tok_per_s" in row and row["tok_per_s"] is not None:
                tok_speeds.append(float(row["tok_per_s"]))
            if "mfu" in row and row["mfu"] is not None:
                mfus.append(float(row["mfu"]))
    # Steady-state median over the second half (avoids torch.compile warmup dip).
    tok_med = round(statistics.median(tok_speeds[len(tok_speeds) // 2 :])) if tok_speeds else None
    mfu_med = round(statistics.median(mfus[len(mfus) // 2 :]), 4) if mfus else None
    return {
        "val_curve": val_curve,
        "tok_per_s": tok_med,
        "mfu": mfu_med,
        "last_step": last_step,
        "last_tokens_m": round(last_tokens / 1e6, 1) if last_tokens is not None else None,
    }


def parse_final_samples(samples_path: Path) -> dict | None:
    """Parse the final `## step N` section of samples.md into {step, items: [{prompt, text}]}."""
    if not samples_path.exists():
        return None
    text = samples_path.read_text(encoding="utf-8")
    sections = re.split(r"(?m)^##\s+step\s+(\d+)\s*$", text)
    if len(sections) < 3:
        return None
    step = int(sections[-2])
    body = sections[-1].strip()
    items = []
    # Each sample entry starts at column 0 with `- **<prompt>**` and continues until the next `- **`.
    pattern = re.compile(r"(?ms)^-\s+\*\*(.+?)\*\*(.*?)(?=^-\s+\*\*|\Z)")
    for m in pattern.finditer(body):
        items.append({"prompt": m.group(1), "text": m.group(2).rstrip()})
    return {"step": step, "items": items}


def compute_paired_bench(npz_ref: Path, npz_cur: Path) -> list[dict]:
    """Compute paired significance tests (ref -> cur) using pinwheel.bench's exact formulas."""
    if not (npz_ref.exists() and npz_cur.exists()):
        return []
    with np.load(npz_ref) as da, np.load(npz_cur) as db:
        tasks_a = {k.split("__")[0] for k in da.files}
        tasks_b = {k.split("__")[0] for k in db.files}
        shared = [t for t in BENCH_TASKS if t in tasks_a and t in tasks_b]
        rows = []
        for t in shared:
            ca, cb = da[f"{t}__correct"], db[f"{t}__correct"]
            if len(ca) != len(cb):
                continue
            clusters = BLIMP_GROUPS if t == "blimp" and len(ca) % BLIMP_GROUPS == 0 else None
            d_acc = float(cb.mean() - ca.mean())
            if clusters:
                _, _, p_acc = paired_diff_test(ca, cb, n_clusters=clusters)
            else:
                _, _, p_acc = mcnemar_test(ca, cb)
            d_prob, _, p_prob = paired_diff_test(da[f"{t}__prob"], db[f"{t}__prob"], n_clusters=clusters)
            d_bpb, _, p_bpb = paired_diff_test(
                da[f"{t}__target_bpb"], db[f"{t}__target_bpb"], n_clusters=clusters
            )
            rows.append(
                {
                    "task": t,
                    "d_acc": round(d_acc, 5),
                    "p_acc": _clean_float(p_acc),
                    "d_prob": round(d_prob, 6),
                    "p_prob": _clean_float(p_prob),
                    "d_bpb": round(d_bpb, 6),
                    "p_bpb": _clean_float(p_bpb),
                    "short_answer": t in SHORT_ANSWER_TASKS,
                }
            )
    return rows


def load_run_artifacts(run_id: str, runs_dir: Path, ckpt_dir: Path) -> dict:
    """Gather config, training curve, final samples, and artifact paths for a single run."""
    rdir = runs_dir / run_id
    cfg_path = rdir / "config.json"
    cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
    cfg_summary = extract_config_summary(cfg_raw) if cfg_raw else None
    log_info = parse_log(rdir / "log.jsonl")
    samples = parse_final_samples(rdir / "samples.md")

    ckpt_path = ckpt_dir / f"{run_id}.pt"
    npz_path = runs_dir / "bench" / f"{run_id}.npz"
    return {
        "config": cfg_summary,
        "val_curve": log_info["val_curve"],
        "tok_per_s": log_info["tok_per_s"],
        "mfu": log_info["mfu"],
        "samples": samples,
        "artifacts": {
            "ckpt_exists": ckpt_path.exists(),
            "ckpt_path": f"ckpt/{run_id}.pt",
            "ckpt_mb": round(ckpt_path.stat().st_size / (1024 * 1024), 1) if ckpt_path.exists() else None,
            "npz_exists": npz_path.exists(),
            "npz_path": f"runs/bench/{run_id}.npz",
            "samples_exists": samples is not None and len(samples["items"]) > 0,
            "samples_path": f"runs/{run_id}/samples.md",
        },
    }


def build_tree_payload(runs_dir: Path, ckpt_dir: Path, spec_path: Path) -> dict:
    """Combine tree.toml with all run artifacts into the JSON payload for the webpage."""
    spec = tomllib.loads(spec_path.read_text(encoding="utf-8"))
    noise_floor = float(spec.get("noise_floor", 0.005))
    raw_nodes: dict[str, dict] = spec.get("nodes") or {}
    if not raw_nodes:
        raise ValueError(f"{spec_path} defines no [nodes.*] entries")

    results = load_results(runs_dir)
    benches = load_bench(runs_dir)

    # Validate tree references first.
    roots = []
    for nid, meta in raw_nodes.items():
        parent = meta.get("parent")
        if parent:
            if parent not in raw_nodes:
                raise ValueError(f"node {nid!r} references missing parent {parent!r}")
        else:
            roots.append(nid)
        cmp_id = meta.get("compare_to")
        if cmp_id is not None and cmp_id != "" and cmp_id not in raw_nodes:
            raise ValueError(f"node {nid!r} references missing compare_to {cmp_id!r}")
    if len(roots) != 1:
        raise ValueError(f"expected exactly 1 root node (no parent), found {roots}")
    root_id = roots[0]

    # Build children adjacency preserving declaration order in tree.toml.
    children_map: dict[str, list[str]] = {nid: [] for nid in raw_nodes}
    for nid, meta in raw_nodes.items():
        if meta.get("parent"):
            children_map[meta["parent"]].append(nid)

    # Resolve each node's raw run data.
    nodes: dict[str, dict] = {}
    for nid, meta in raw_nodes.items():
        status = meta.get("status", "branch")
        is_group = status == "group"
        res = None if is_group else results.get(nid)
        bench = None if is_group else benches.get(nid)
        art = None if is_group else load_run_artifacts(nid, runs_dir, ckpt_dir)

        nodes[nid] = {
            "id": nid,
            "parent": meta.get("parent"),
            "children": children_map[nid],
            "status": status,
            "milestone": meta.get("milestone") or "",
            "label": meta.get("label") or nid,
            "protocol": meta.get("protocol") or "",
            "takeaway": meta.get("takeaway") or "",
            "collapsed": bool(meta.get("collapsed", False)),
            "is_group": is_group,
            "run_id": None if is_group else nid,
            "date": res["date"] if res else "",
            "git_sha": res["git_sha"] if res else "",
            "change": res["change"] if res else "",
            "params": res["params"] if res else "",
            "tokens": res["tokens"] if res else "",
            "wall_h": res["wall_h"] if res else None,
            "val_bpb": res["val_bpb"] if res else None,
            "bench": bench,
            "config": art["config"] if art else None,
            "val_curve": art["val_curve"] if art else [],
            "tok_per_s": art["tok_per_s"] if art else None,
            "mfu": art["mfu"] if art else None,
            "samples": art["samples"] if art else None,
            "artifacts": art["artifacts"] if art else None,
            "compare_to": None,
            "cross_compare": False,
            "delta_bpb": None,
            "in_noise": False,
            "config_diff": [],
            "paired_bench": [],
            "compared_by": [],
        }

    def nearest_run_ancestor(nid: str) -> str | None:
        cur = nodes[nid]["parent"]
        while cur:
            if not nodes[cur]["is_group"]:
                return cur
            cur = nodes[cur]["parent"]
        return None

    # Resolve compare_to, deltas, diffs, and paired bench tests.
    for nid, meta in raw_nodes.items():
        node = nodes[nid]
        if node["is_group"]:
            continue
        if "compare_to" in meta:
            cmp_id = meta["compare_to"] or None
        else:
            cmp_id = nearest_run_ancestor(nid)
        node["compare_to"] = cmp_id
        if cmp_id and cmp_id in nodes:
            ref = nodes[cmp_id]
            node["cross_compare"] = cmp_id != node["parent"]
            if node["val_bpb"] is not None and ref["val_bpb"] is not None:
                delta = round(node["val_bpb"] - ref["val_bpb"], 4)
                node["delta_bpb"] = delta
                node["in_noise"] = abs(delta) < noise_floor
            node["config_diff"] = compute_config_diff(node["config"], ref["config"])
            npz_ref = runs_dir / "bench" / f"{cmp_id}.npz"
            npz_cur = runs_dir / "bench" / f"{nid}.npz"
            node["paired_bench"] = compute_paired_bench(npz_ref, npz_cur)
            ref["compared_by"].append(
                {
                    "id": nid,
                    "label": node["label"],
                    "delta_from_other": round(-node["delta_bpb"], 4) if node["delta_bpb"] is not None else None,
                    "protocol": node["protocol"],
                }
            )

    # Summarize group nodes (descendant run count + best val_bpb).
    def collect_descendant_runs(nid: str) -> list[dict]:
        out = []
        for cid in nodes[nid]["children"]:
            if nodes[cid]["is_group"]:
                out.extend(collect_descendant_runs(cid))
            else:
                out.append(nodes[cid])
        return out

    for nid, node in nodes.items():
        if not node["is_group"]:
            continue
        desc = collect_descendant_runs(nid)
        valid = [d for d in desc if d["val_bpb"] is not None]
        best = min(valid, key=lambda d: d["val_bpb"]) if valid else None
        node["group_stats"] = {
            "count": len(desc),
            "best_id": best["id"] if best else None,
            "best_label": best["label"] if best else None,
            "best_bpb": best["val_bpb"] if best else None,
        }

    # Identify trunk path (root -> adopted chain) and mark HEAD.
    trunk: list[str] = [root_id]
    cur = root_id
    while True:
        adopted_kids = [c for c in nodes[cur]["children"] if nodes[c]["status"] == "adopted"]
        if not adopted_kids:
            break
        cur = adopted_kids[0]
        trunk.append(cur)
    head_id = trunk[-1]
    nodes[head_id]["is_head"] = True

    # Detect uncataloged finished runs and currently running directories.
    cataloged = set(nodes.keys())
    uncataloged_finished = [
        {"run_id": rid, "val_bpb": r["val_bpb"], "date": r["date"], "change": r["change"]}
        for rid, r in results.items()
        if rid not in cataloged
    ]
    in_progress = []
    if runs_dir.exists():
        for p in sorted(runs_dir.iterdir()):
            if not p.is_dir() or p.name == "bench" or p.name in results or p.name in cataloged:
                continue
            if (p / "log.jsonl").exists():
                log_info = parse_log(p / "log.jsonl")
                in_progress.append(
                    {
                        "run_id": p.name,
                        "step": log_info["last_step"],
                        "tokens_m": log_info["last_tokens_m"],
                    }
                )

    return {
        "meta": {
            "title": spec.get("title", "pinwheel-llm 实验树"),
            "subtitle": spec.get("subtitle", ""),
            "noise_floor": noise_floor,
            "root_id": root_id,
            "head_id": head_id,
            "trunk": trunk,
            "total_runs": sum(1 for n in nodes.values() if not n["is_group"]),
            "bench_tasks": list(BENCH_TASKS),
        },
        "nodes": nodes,
        "uncataloged": uncataloged_finished,
        "in_progress": in_progress,
    }


def render_html(payload: dict, template_path: Path, out_path: Path) -> Path:
    """Inject JSON payload safely into the static HTML template."""
    template = template_path.read_text(encoding="utf-8")
    if PLACEHOLDER not in template:
        raise ValueError(f"template {template_path} is missing {PLACEHOLDER}")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Prevent `</script>` or `<!--` in sample text from breaking out of the inline <script>.
    encoded = encoded.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    html = template.replace(PLACEHOLDER, encoded, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path


def _to_float(val: str | None) -> float | None:
    if val is None or val.strip() == "":
        return None
    try:
        return float(val)
    except ValueError:
        return None


def _clean_float(val: float) -> float | None:
    if val is None or math.isnan(val) or math.isinf(val):
        return None
    return val


def _fmt_tokens(val) -> str:
    if val is None:
        return ""
    n = float(val)
    if n >= 1e9:
        return f"{n / 1e9:g}B"
    if n >= 1e6:
        return f"{n / 1e6:g}M"
    return str(int(n))


def _fmt_val(val) -> str:
    if val is None or val == "":
        return "—"
    if isinstance(val, float):
        return f"{val:g}"
    return str(val)


def main():
    p = argparse.ArgumentParser(description="Generate interactive experiment tree HTML from runs/ and tree.toml")
    p.add_argument("--spec", type=Path, default=Path("tree.toml"), help="hand-written topology TOML")
    p.add_argument("--runs", type=Path, default=Path("runs"), help="runs directory")
    p.add_argument("--ckpt", type=Path, default=Path("ckpt"), help="checkpoints directory")
    p.add_argument("-o", "--out", type=Path, default=Path("runs/tree.html"), help="output HTML path")
    p.add_argument("--serve", action="store_true", help="serve runs/ over a local HTTP server and open in browser")
    p.add_argument("--port", type=int, default=8765, help="port for --serve")
    args = p.parse_args()

    template_path = Path(__file__).with_name("tree_template.html")
    payload = build_tree_payload(args.runs, args.ckpt, args.spec)
    out = render_html(payload, template_path, args.out)
    m = payload["meta"]
    print(f"wrote {out} ({out.stat().st_size / 1024:.1f} KB, {m['total_runs']} runs, head={m['head_id']})")

    if payload["uncataloged"]:
        print("\nUncataloged finished runs (add to tree.toml):")
        for r in payload["uncataloged"]:
            print(f'  [nodes."{r["run_id"]}"]')
            print(f'  parent = "{m["head_id"]}"')
            print('  status = "branch"')
            print(f'  label = "{r["run_id"]}"')
            print(f'  takeaway = "{r["change"]}"\n')

    if payload["in_progress"]:
        names = ", ".join(f"{r['run_id']} ({r['tokens_m']}M tok)" for r in payload["in_progress"])
        print(f"In-progress runs (not yet in results.tsv): {names}")

    if args.serve:
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(out.parent.resolve()))
        with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
            url = f"http://127.0.0.1:{args.port}/{out.name}"
            print(f"serving at {url} (Ctrl-C to stop)")
            webbrowser.open(url)
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
