#!/usr/bin/env bash
# Stage 2e: Muon check-up. 300M-token WSD runs from scratch (about 0.5 h each): the tuned AdamW
# reference, then Muon at three learning rates 2x apart. muon_lr is on AdamW's scale (the update RMS
# is matched), so the prediction is that 2e-3, the AdamW optimum, is already about right.
set -u
cd "$(dirname "$0")/.."

run() {  # name note [extra flags]
  name=$1 note=$2
  shift 2
  [ -d "runs/$name" ] && return  # already done: rerunning the script only adds new runs
  uv run python -u -m pinwheel.train --data_dir data/tok8192/fineweb-edu --run "$name" --note "$note" \
    --max_tokens 3e8 --eval_every 100 --batch_tokens 131072 --lr 2e-3 --schedule wsd --min_lr_frac 0.0 \
    --qk_norm false --zero_init false --emb_std 0.02 "$@" > "runs/$name.out" 2>&1
}

run 2e-adamw-300m "2e: AdamW reference, 300M tokens" --optimizer adamw
for lr in 2e-3 1e-3 4e-3; do
  run "2e-muon-lr$lr-300m" "2e: Muon on block matrices, muon_lr $lr, 300M tokens" --optimizer muon --muon_lr "$lr"
done
