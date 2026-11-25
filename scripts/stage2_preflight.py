"""Stage 2 pre-flight runner: batch-size sweep + WSD confirmation with optional GPU blackout guard.

    uv run python -u scripts/stage2_preflight.py [--quick] [--dry-run] \
        [--blackout 03:00-03:35 07:30-07:50 18:00-18:20] \
        [--avoid-service svc1.service svc2.service]

Runs sequentially and is resumable (skips any run whose ckpt/<run>.pt already exists):
  1. Batch-size sweep on FineWeb-Edu 1B (cosine, compared against existing 512k baseline 1e-v0-lr2e-3):
     - Primary probes:  256k @ lr 1.5e-3, 128k @ lr 1e-3
     - Bracket points:  256k @ lr 2e-3,   128k @ lr 1.4e-3  (skipped with --quick)
  2. WSD + (1-sqrt) cooldown confirmation on the winning (batch_tokens, lr):
     - 2b-wsd-1b (1B tokens, cooldown_frac=0.2, min_lr_frac=0.0, saves stable ckpt at 500M)
     - 2b-wsd-500m-cd (100M cooldown branched from 2b-wsd-1b-stable.pt)
  3. Tier 2 benchmark + paired significance comparison against 1e-v0-lr2e-3.
"""

import argparse
import csv
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = "data/tok8192/fineweb-edu"
BASELINE_RUN = "1e-v0-lr2e-3"
BASELINE_SPEC = (524288, 2e-3)  # (batch_tokens, lr) of 1e-v0-lr2e-3

BATCH_SWEEP = [
    # (run_name, batch_tokens, lr, is_quick)
    ("2a-bs256k-lr1.5e-3", 262144, 1.5e-3, True),
    ("2a-bs128k-lr1e-3", 131072, 1e-3, True),
    ("2a-bs256k-lr2e-3", 262144, 2e-3, False),
    ("2a-bs128k-lr1.4e-3", 131072, 1.4e-3, False),
]


def parse_blackouts(specs: list[str]):
    """Parse 'HH:MM-HH:MM' strings into [((sh, sm), (eh, em), spec), ...]."""
    out = []
    for s in specs:
        a, b = s.split("-")
        sh, sm = (int(x) for x in a.split(":"))
        eh, em = (int(x) for x in b.split(":"))
        out.append(((sh, sm), (eh, em), s))
    return out


def next_safe_start(now: datetime, est_minutes: int, blackouts):
    """Return (earliest_start, blocked_label) so [start, start + est_minutes] avoids all blackouts."""
    start = now
    blocked = None
    dur = timedelta(minutes=est_minutes)
    for _ in range(len(blackouts) * 2 + 1):
        end = start + dur
        moved = False
        for day_offset in (-1, 0, 1):
            day = (start + timedelta(days=day_offset)).date()
            for (sh, sm), (eh, em), label in blackouts:
                b_start = datetime(day.year, day.month, day.day, sh, sm)
                b_end = datetime(day.year, day.month, day.day, eh, em)
                if start < b_end and end > b_start:
                    start = b_end
                    blocked = label
                    moved = True
                    break
            if moved:
                break
        if not moved:
            return start, blocked
    return start, blocked


def active_gpu_services(services: list[str]):
    """Return list of watched systemd user services currently active or activating."""
    active = []
    for svc in services:
        res = subprocess.run(
            ["systemctl", "--user", "is-active", svc],
            capture_output=True, text=True,
        )
        if res.stdout.strip() in ("active", "activating"):
            active.append(svc)
    return active


def wait_for_gpu_window(est_minutes: int, blackouts, services: list[str], dry_run: bool = False):
    while True:
        now = datetime.now()
        safe, label = next_safe_start(now, est_minutes, blackouts)
        if safe > now:
            wait_s = (safe - now).total_seconds()
            print(f"[guard] {now:%H:%M} -> next run (~{est_minutes} min) overlaps {label}; "
                  f"waiting until {safe:%H:%M} ({wait_s / 60:.0f} min)", flush=True)
            if dry_run:
                return
            time.sleep(wait_s)
            continue
        if dry_run:
            return
        running = active_gpu_services(services)
        if running:
            print(f"[guard] {now:%H:%M} -> waiting for {', '.join(running)} to finish...", flush=True)
            time.sleep(60)
            continue
        return


def run_train(name: str, extra_args: list[str], est_minutes: int, blackouts, services, dry_run: bool = False):
    ckpt = ROOT / "ckpt" / f"{name}.pt"
    if ckpt.exists():
        print(f"[skip] {name} already completed ({ckpt.name} exists)", flush=True)
        return
    run_dir = ROOT / "runs" / name
    if run_dir.exists() and not dry_run:
        backup = ROOT / "runs" / f"{name}.aborted-{time.strftime('%Y%m%d-%H%M%S')}"
        run_dir.rename(backup)
        print(f"[clean] moved incomplete {run_dir.name} -> {backup.name}", flush=True)

    wait_for_gpu_window(est_minutes, blackouts, services, dry_run=dry_run)
    cmd = ["uv", "run", "python", "-u", "-m", "pinwheel.train",
           "--data_dir", DATA_DIR, "--run", name, *extra_args]
    out_log = ROOT / "runs" / f"{name}.out"
    print(f"[run]  {datetime.now():%H:%M} {' '.join(cmd)}", flush=True)
    if dry_run:
        return
    out_log.parent.mkdir(parents=True, exist_ok=True)
    with open(out_log, "w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
    print(f"[done] {datetime.now():%H:%M} {name} (log: {out_log.relative_to(ROOT)})", flush=True)


def read_val_bpb(results_path=ROOT / "runs" / "results.tsv") -> dict[str, float]:
    if not results_path.exists():
        return {}
    out = {}
    with open(results_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                out[row["run_id"]] = float(row["val_bpb"])
            except (KeyError, ValueError):
                pass
    return out


def pick_best_batch_config(sweep_entries, results: dict[str, float]):
    """Pick the (run_name, batch_tokens, lr, val_bpb) with lowest val_bpb across baseline + sweep."""
    candidates = [(BASELINE_RUN, BASELINE_SPEC[0], BASELINE_SPEC[1])] + [
        (name, bs, lr) for name, bs, lr, _ in sweep_entries
    ]
    scored = [(results[name], name, bs, lr) for name, bs, lr in candidates if name in results]
    if not scored:
        return BASELINE_RUN, BASELINE_SPEC[0], BASELINE_SPEC[1], float("nan")
    bpb, name, bs, lr = min(scored)
    return name, bs, lr, bpb


def run_cmd(cmd: list[str], dry_run: bool = False):
    print(f"[cmd]  {' '.join(cmd)}", flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    p = argparse.ArgumentParser(description="Run Stage 2 pre-flight experiments with GPU blackout guard.")
    p.add_argument("--quick", action="store_true", help="only run the 2 primary batch-size probes + WSD (~5.2h)")
    p.add_argument("--dry-run", action="store_true", help="print commands and guard decisions without running")
    p.add_argument("--blackout", nargs="*", default=[], metavar="HH:MM-HH:MM",
                   help="local-time windows to keep GPU free (e.g. 03:00-03:35 07:30-07:50)")
    p.add_argument("--avoid-service", nargs="*", default=[], metavar="UNIT",
                   help="systemd --user services to wait for if active before starting a run")
    args = p.parse_args()

    blackouts = parse_blackouts(args.blackout)
    sweep = [e for e in BATCH_SWEEP if e[3]] if args.quick else BATCH_SWEEP
    print(f"=== Stage 2 pre-flight ({'quick' if args.quick else 'full'}, {len(sweep)} batch-size runs + WSD) ===",
          flush=True)

    # 1. Batch-size sweep (cosine, 1B tokens)
    for name, bs, lr, _ in sweep:
        run_train(
            name,
            ["--max_tokens", "1e9", "--batch_tokens", str(bs), "--lr", str(lr),
             "--note", f"2a: batch size {bs // 1024}k, lr {lr}"],
            est_minutes=115,
            blackouts=blackouts,
            services=args.avoid_service,
            dry_run=args.dry_run,
        )

    # 2. Pick winner (lowest val BPB) and run WSD + (1-sqrt) cooldown confirmation
    results = read_val_bpb()
    win_name, win_bs, win_lr, win_bpb = pick_best_batch_config(sweep, results)
    print(f"\n[winner] {win_name}: batch_tokens={win_bs}, lr={win_lr}, val_bpb={win_bpb:.4f}", flush=True)

    run_train(
        "2b-wsd-1b",
        ["--max_tokens", "1e9", "--batch_tokens", str(win_bs), "--lr", str(win_lr),
         "--schedule", "wsd", "--cooldown_frac", "0.2", "--min_lr_frac", "0.0",
         "--save_stable_at", "5e8",
         "--note", f"2b: WSD (1-sqrt 20% cd), bs {win_bs // 1024}k, lr {win_lr}"],
        est_minutes=115,
        blackouts=blackouts,
        services=args.avoid_service,
        dry_run=args.dry_run,
    )
    run_train(
        "2b-wsd-500m-cd",
        ["--cooldown_from", "ckpt/2b-wsd-1b-stable.pt", "--seed", "1000",
         "--max_tokens", "1e8", "--batch_tokens", str(win_bs), "--lr", str(win_lr),
         "--schedule", "wsd", "--min_lr_frac", "0.0",
         "--note", f"2b: WSD 100M cooldown branched from 500M stable (bs {win_bs // 1024}k)"],
        est_minutes=20,
        blackouts=blackouts,
        services=args.avoid_service,
        dry_run=args.dry_run,
    )

    # 3. Tier 2 benchmark + paired comparison on winner and WSD
    eval_targets = []
    if win_name != BASELINE_RUN:
        eval_targets.append(win_name)
    eval_targets.append("2b-wsd-1b")
    for target in eval_targets:
        wait_for_gpu_window(5, blackouts, args.avoid_service, dry_run=args.dry_run)
        run_cmd(["uv", "run", "python", "-m", "pinwheel.bench",
                 f"ckpt/{target}.pt", "--suite", "tier2"], dry_run=args.dry_run)
        run_cmd(["uv", "run", "python", "-m", "pinwheel.bench",
                 "--compare", f"runs/bench/{BASELINE_RUN}.npz", f"runs/bench/{target}.npz"],
                dry_run=args.dry_run)

    # 4. Print final val BPB summary
    final = read_val_bpb()
    print("\n=== Val BPB summary ===", flush=True)
    for r in [BASELINE_RUN, *(n for n, _, _, _ in sweep), "2b-wsd-1b", "2b-wsd-500m-cd"]:
        if r in final:
            print(f"  {r:22s}  val_bpb = {final[r]:.4f}", flush=True)


if __name__ == "__main__":
    main()
