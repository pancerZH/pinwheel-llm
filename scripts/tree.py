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
import http.server
import json
import math
import re
import socket
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
    ("arch", "架构", "Architecture"),
    ("n_layer", "层数", "Layers"),
    ("d_model", "宽度 d_model", "Width d_model"),
    ("n_head", "Q 头数", "Q heads"),
    ("n_kv_head", "KV 头数", "KV heads"),
    ("d_ff", "FFN 宽度", "FFN width"),
    ("vocab_size", "词表", "Vocab size"),
    ("data", "数据集", "Dataset"),
    ("max_tokens", "训练 tokens", "Max tokens"),
    ("batch_tokens", "Batch tokens", "Batch tokens"),
    ("schedule", "LR 调度", "LR schedule"),
    ("lr", "峰值 LR", "Peak LR"),
    ("warmup_steps", "Warmup 步数", "Warmup steps"),
    ("cooldown_frac", "退火比例", "Cooldown frac"),
    ("min_lr_frac", "最低 LR 比例", "Min LR frac"),
    ("weight_decay", "Weight decay", "Weight decay"),
    ("seed", "Seed", "Seed"),
    ("cooldown_from", "分支退火起点", "Cooldown from"),
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
                "val_bpb_clean": _to_float(row.get("val_bpb_clean")),
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
    for key, label, label_en in DIFF_FIELDS:
        a, b = ref.get(key), cur.get(key)
        if a == b:
            continue
        diffs.append({"key": key, "label": label, "label_en": label_en, "from": _fmt_val(a), "to": _fmt_val(b)})
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


def build_scaling_chart(run_ids: list[str], runs_dir: Path, results: dict[str, dict]) -> dict | None:
    """Compute scaling-law points (params, compute FLOPs, LR, val BPB) and power-law fit BPB = a * C^-b."""
    seen: set[str] = set()
    points = []
    for rid in run_ids:
        if rid in seen or rid not in results or results[rid]["val_bpb"] is None:
            continue
        seen.add(rid)
        cfg_path = runs_dir / rid / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        n = cfg.get("params")
        m = cfg.get("model") or {}
        t = cfg.get("train") or {}
        if not (n and m.get("n_layer") and m.get("d_model") and t.get("seq_len") and t.get("max_tokens") and t.get("batch_tokens")):
            continue
        tokens = int(t["max_tokens"] // t["batch_tokens"]) * int(t["batch_tokens"])
        flops_per_token = 6 * n + 12 * m["n_layer"] * m["d_model"] * t["seq_len"]
        compute = float(flops_per_token * tokens)
        points.append(
            {
                "id": rid,
                "params": int(n),
                "params_m": round(n / 1e6, 2),
                "tokens_m": round(tokens / 1e6, 1),
                "compute": compute,
                "lr": float(t["lr"]),
                "bpb": float(results[rid]["val_bpb"]),
            }
        )
    if not points:
        return None

    by_size: dict[int, list[dict]] = {}
    for pt in points:
        by_size.setdefault(pt["params"], []).append(pt)
    best = [min(rs, key=lambda r: r["bpb"]) for _, rs in sorted(by_size.items())]

    fit = None
    if len(best) >= 2:
        c = np.array([r["compute"] for r in best], dtype=np.float64)
        y = np.array([r["bpb"] for r in best], dtype=np.float64)
        slope, intercept = np.polyfit(np.log(c), np.log(y), 1)
        a = float(math.exp(intercept))
        b = float(-slope)
        fit = {
            "a": round(a, 4),
            "b": round(b, 5),
            "drop_per_2x_pct": round((1.0 - 2.0 ** (-b)) * 100.0, 2),
            "residuals": [
                {"id": r["id"], "params_m": r["params_m"], "residual": round(r["bpb"] - a * (r["compute"] ** (-b)), 4)}
                for r in best
            ],
        }
    return {
        "kind": "scaling",
        "lrs": sorted({pt["lr"] for pt in points}),
        "points": sorted(points, key=lambda r: (r["params"], r["lr"])),
        "best_ids": [r["id"] for r in best],
        "fit": fit,
    }


def resolve_cooldown_offset_tokens(cfg_raw: dict | None, runs_dir: Path) -> float:
    """Infer cumulative starting tokens for a run branched via --cooldown_from."""
    if not cfg_raw:
        return 0.0
    cd_from = (cfg_raw.get("train") or {}).get("cooldown_from") or ""
    if not cd_from:
        return 0.0
    stem = Path(cd_from).stem
    is_stable = stem.endswith("-stable")
    src_id = stem[: -len("-stable")] if is_stable else stem
    src_cfg_path = runs_dir / src_id / "config.json"
    if src_cfg_path.exists():
        src_cfg = json.loads(src_cfg_path.read_text(encoding="utf-8"))
        st = src_cfg.get("train") or {}
        if is_stable and st.get("save_stable_at"):
            target = float(st["save_stable_at"])
            bs = int(st.get("batch_tokens") or 0)
            if bs > 0:
                return float(math.ceil(target / bs) * bs)
            return target
    src_log = parse_log(runs_dir / src_id / "log.jsonl")
    if src_log["last_tokens_m"] is not None:
        return src_log["last_tokens_m"] * 1e6
    return 0.0


def load_run_artifacts(run_id: str, runs_dir: Path, ckpt_dir: Path) -> dict:
    """Gather config, training curve, final samples, and artifact paths for a single run."""
    rdir = runs_dir / run_id
    cfg_path = rdir / "config.json"
    cfg_raw = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else None
    cfg_summary = extract_config_summary(cfg_raw) if cfg_raw else None
    log_info = parse_log(rdir / "log.jsonl")
    offset_toks = resolve_cooldown_offset_tokens(cfg_raw, runs_dir)
    offset_m = round(offset_toks / 1e6, 2) if offset_toks > 0 else 0.0
    val_curve = (
        [[round(t_m + offset_m, 2), bpb] for t_m, bpb in log_info["val_curve"]]
        if offset_m > 0
        else log_info["val_curve"]
    )
    samples = parse_final_samples(rdir / "samples.md")

    ckpt_path = ckpt_dir / f"{run_id}.pt"
    npz_path = runs_dir / "bench" / f"{run_id}.npz"
    return {
        "config": cfg_summary,
        "val_curve": val_curve,
        "token_offset_m": offset_m,
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
    noise_floor = float(spec.get("noise_floor", 0.01))
    raw_nodes: dict[str, dict] = spec.get("nodes") or {}
    raw_refs: dict[str, dict] = spec.get("refs") or {}
    if not raw_nodes:
        raise ValueError(f"{spec_path} defines no [nodes.*] entries")

    results = load_results(runs_dir)
    benches = load_bench(runs_dir)

    # Validate tree and reference links first.
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
        for item in meta.get("inspired_by") or []:
            src_id = item if isinstance(item, str) else (item.get("from") or "")
            if src_id not in raw_nodes:
                raise ValueError(f"node {nid!r} references missing inspired_by {src_id!r}")
        for item in meta.get("refs") or []:
            ref_id = item if isinstance(item, str) else (item.get("id") or "")
            if ref_id not in raw_refs:
                raise ValueError(f"node {nid!r} references missing ref {ref_id!r}")
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
        is_placeholder = bool(meta.get("placeholder", False))
        # A target is an outside reference model scored on our bench (e.g. GPT-2 124M): no training run.
        is_target = status == "target"
        no_run = is_group or is_placeholder or is_target
        res = None if no_run else results.get(nid)
        bench = None if (is_group or is_placeholder) else benches.get(nid)
        art = None if no_run else load_run_artifacts(nid, runs_dir, ckpt_dir)
        val_bpb = (
            float(meta["val_bpb"])
            if (is_placeholder and meta.get("val_bpb") is not None)
            else (res["val_bpb"] if res else None)
        )
        ref_json = runs_dir / nid / "ref.json"  # a target's val BPB on our val text (scripts/gpt2_ref.py --bpb)
        val_bpb_clean = res.get("val_bpb_clean") if res else None
        if is_target and ref_json.exists():
            ref_data = json.loads(ref_json.read_text(encoding="utf-8"))
            val_bpb, val_bpb_clean = ref_data.get("val_bpb"), ref_data.get("val_bpb_clean")

        nodes[nid] = {
            "id": nid,
            "parent": meta.get("parent"),
            "children": children_map[nid],
            "status": status,
            "milestone": meta.get("milestone") or "",
            "milestone_en": meta.get("milestone_en") or "",
            "label": meta.get("label") or nid,
            "label_en": meta.get("label_en") or "",
            "protocol": meta.get("protocol") or "",
            "protocol_en": meta.get("protocol_en") or "",
            "takeaway": meta.get("takeaway") or "",
            "takeaway_en": meta.get("takeaway_en") or "",
            "collapsed": bool(meta.get("collapsed", False)),
            "is_group": is_group,
            "is_placeholder": is_placeholder,
            "is_target": is_target,
            "target_metric": (meta.get("target_metric") or "hellaswag") if is_target else "",
            "run_id": None if no_run else nid,
            "date": res["date"] if res else "",
            "git_sha": res["git_sha"] if res else "",
            "change": res["change"] if res else "",
            "params": (meta.get("params") or "") if (is_placeholder or is_target) else (res["params"] if res else ""),
            "tokens": (meta.get("tokens") or "") if (is_placeholder or is_target) else (res["tokens"] if res else ""),
            "wall_h": res["wall_h"] if res else None,
            "val_bpb": val_bpb,
            "val_bpb_clean": val_bpb_clean,
            "bench": bench,
            "config": art["config"] if art else None,
            "val_curve": art["val_curve"] if art else [],
            "token_offset_m": art["token_offset_m"] if art else 0.0,
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
            "inspired_by": [],
            "inspired": [],
            "refs": [],
        }

    def nearest_run_ancestor(nid: str) -> str | None:
        cur = nodes[nid]["parent"]
        while cur:
            if not nodes[cur]["is_group"]:
                return cur
            cur = nodes[cur]["parent"]
        return None

    def closest_to_target(target: dict) -> str | None:
        """Our run with the best score on the target's metric (the one chasing it most closely)."""
        metric = target["target_metric"]
        scored = [
            n for n in nodes.values()
            if not (n["is_group"] or n["is_placeholder"] or n["is_target"])
            and n["bench"] and n["bench"]["scores"].get(metric) is not None
        ]
        return max(scored, key=lambda n: n["bench"]["scores"][metric])["id"] if scored else None

    # Resolve compare_to, deltas, diffs, and paired bench tests.
    for nid, meta in raw_nodes.items():
        node = nodes[nid]
        if node["is_group"]:
            continue
        if "compare_to" in meta:
            cmp_id = meta["compare_to"] or None
        elif node["is_target"]:
            cmp_id = closest_to_target(node)
        else:
            cmp_id = nearest_run_ancestor(nid)
        node["compare_to"] = cmp_id
        if cmp_id and cmp_id in nodes:
            ref = nodes[cmp_id]
            node["cross_compare"] = cmp_id != node["parent"]
            if (
                not node["is_placeholder"]
                and not ref["is_placeholder"]
                and node["val_bpb"] is not None
                and ref["val_bpb"] is not None
            ):
                delta = round(node["val_bpb"] - ref["val_bpb"], 4)
                node["delta_bpb"] = delta
                node["in_noise"] = abs(delta) < noise_floor
            node["config_diff"] = compute_config_diff(node["config"], ref["config"])
            if not node["is_placeholder"] and not ref["is_placeholder"]:
                npz_ref = runs_dir / "bench" / f"{cmp_id}.npz"
                npz_cur = runs_dir / "bench" / f"{nid}.npz"
                node["paired_bench"] = compute_paired_bench(npz_ref, npz_cur)
            ref["compared_by"].append(
                {
                    "id": nid,
                    "label": node["label"],
                    "label_en": node["label_en"],
                    "delta_from_other": round(-node["delta_bpb"], 4) if node["delta_bpb"] is not None else None,
                    "protocol": node["protocol"],
                    "protocol_en": node["protocol_en"],
                }
            )

    # Resolve inspired_by links (multi-source DAG inspiration edges, works for both runs and groups).
    for nid, meta in raw_nodes.items():
        node = nodes[nid]
        for item in meta.get("inspired_by") or []:
            if isinstance(item, str):
                src_id, reason, reason_en = item, "", ""
            else:
                src_id = item["from"]
                reason = item.get("reason") or ""
                reason_en = item.get("reason_en") or ""
            src = nodes[src_id]
            node["inspired_by"].append(
                {
                    "id": src_id,
                    "label": src["label"],
                    "label_en": src["label_en"],
                    "reason": reason,
                    "reason_en": reason_en,
                }
            )
            src["inspired"].append(
                {
                    "id": nid,
                    "label": node["label"],
                    "label_en": node["label_en"],
                    "reason": reason,
                    "reason_en": reason_en,
                }
            )

    # Build external reference / knowledge nodes and bidirectional links.
    refs: dict[str, dict] = {}
    for rid, rmeta in raw_refs.items():
        links = []
        if rmeta.get("url"):
            links.append({"label": rmeta.get("url_label") or rmeta["url"], "url": rmeta["url"]})
        for lk in rmeta.get("links") or []:
            if isinstance(lk, str):
                links.append({"label": lk, "url": lk})
            elif lk.get("url"):
                links.append({"label": lk.get("label") or lk["url"], "url": lk["url"]})
        refs[rid] = {
            "id": rid,
            "kind": rmeta.get("kind") or "paper",
            "label": rmeta.get("label") or rid,
            "label_en": rmeta.get("label_en") or "",
            "authors": rmeta.get("authors") or "",
            "summary": rmeta.get("summary") or "",
            "summary_en": rmeta.get("summary_en") or "",
            "links": links,
            "used_by": [],
        }

    for nid, meta in raw_nodes.items():
        node = nodes[nid]
        for item in meta.get("refs") or []:
            if isinstance(item, str):
                ref_id, reason, reason_en = item, "", ""
            else:
                ref_id = item["id"]
                reason = item.get("reason") or ""
                reason_en = item.get("reason_en") or ""
            r = refs[ref_id]
            node["refs"].append(
                {
                    "id": ref_id,
                    "kind": r["kind"],
                    "label": r["label"],
                    "label_en": r["label_en"],
                    "reason": reason,
                    "reason_en": reason_en,
                    "primary_url": r["links"][0]["url"] if r["links"] else "",
                }
            )
            r["used_by"].append(
                {
                    "id": nid,
                    "label": node["label"],
                    "label_en": node["label_en"],
                    "val_bpb": node["val_bpb"],
                    "reason": reason,
                    "reason_en": reason_en,
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
        valid = [d for d in desc if d["val_bpb"] is not None and not d["is_placeholder"]]
        best = min(valid, key=lambda d: d["val_bpb"]) if valid else None
        node["group_stats"] = {
            "count": len(desc),
            "best_id": best["id"] if best else None,
            "best_label": best["label"] if best else None,
            "best_label_en": best["label_en"] if best else None,
            "best_bpb": best["val_bpb"] if best else None,
        }
        meta = raw_nodes[nid]
        if meta.get("chart") == "scaling":
            extra = list(meta.get("chart_extra_runs") or [])
            chart_ids = [d["id"] for d in desc] + extra
            node["scaling_chart"] = build_scaling_chart(chart_ids, runs_dir, results)

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
            "title_en": spec.get("title_en", "pinwheel-llm Experiment Tree"),
            "subtitle": spec.get("subtitle", ""),
            "subtitle_en": spec.get("subtitle_en", ""),
            "noise_floor": noise_floor,
            "root_id": root_id,
            "head_id": head_id,
            "trunk": trunk,
            "total_runs": sum(
                1 for n in nodes.values() if not (n["is_group"] or n["is_placeholder"] or n["is_target"])
            ),
            "bench_tasks": list(BENCH_TASKS),
        },
        "nodes": nodes,
        "refs": refs,
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


def _lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))
            return s.getsockname()[0]
    except OSError:
        return None


def _serve_single_file(html_path: Path, host: str, port: int) -> None:
    """Serve only the generated HTML file (no directory listing, no sibling files)."""
    target_name = "/" + html_path.name

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path not in ("/", target_name):
                self.send_error(404)
                return
            body = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            pass

    class Server(http.server.ThreadingHTTPServer):
        allow_reuse_address = True

    with Server((host, port), Handler) as httpd:
        if host in ("0.0.0.0", ""):
            ip = _lan_ip() or "127.0.0.1"
            print(f"serving on LAN: http://{ip}:{port}/ (and http://127.0.0.1:{port}/) — Ctrl-C to stop", flush=True)
        else:
            url = f"http://{host}:{port}/"
            print(f"serving at {url} — Ctrl-C to stop", flush=True)
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


def main():
    p = argparse.ArgumentParser(description="Generate interactive experiment tree HTML from runs/ and tree.toml")
    p.add_argument("--spec", type=Path, default=Path("tree.toml"), help="hand-written topology TOML")
    p.add_argument("--runs", type=Path, default=Path("runs"), help="runs directory")
    p.add_argument("--ckpt", type=Path, default=Path("ckpt"), help="checkpoints directory")
    p.add_argument("-o", "--out", type=Path, default=Path("runs/tree.html"), help="output HTML path")
    p.add_argument("--serve", action="store_true", help="serve only the generated HTML over HTTP")
    p.add_argument("--host", default="127.0.0.1", help="bind address for --serve (use 0.0.0.0 for LAN)")
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
        _serve_single_file(out.resolve(), args.host, args.port)


if __name__ == "__main__":
    main()
