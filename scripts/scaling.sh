#!/usr/bin/env bash
# Mini scaling law (stage 1 side line): 5M / 10M / 20M models on FineWeb-Edu at about 20 tokens per
# parameter, four learning rates each; the 50M point is 1c. The batch doubles with the model, so every
# run gets 1525 steps instead of a few hundred, and warmup stays at 100 steps (the 1b lesson).
set -u
cd "$(dirname "$0")/.."

run() {  # name d_model n_layer n_head n_kv_head d_ff max_tokens batch_tokens
  for lr in 1e-3 2e-3 4e-3 8e-3; do
    name="side-scaling-$1-lr$lr"
    [ -d "runs/$name" ] && continue  # already done: rerunning the script only adds new runs
    uv run python -u -m pinwheel.train --data_dir data/tok8192/fineweb-edu --run "$name" \
      --note "side: scaling law $1, ~20x tokens, lr $lr" \
      --d_model "$2" --n_layer "$3" --n_head "$4" --n_kv_head "$5" --d_ff "$6" \
      --max_tokens "$7" --batch_tokens "$8" --lr "$lr" \
      --optimizer adamw --schedule cosine --min_lr_frac 0.1 \
      --qk_norm false --zero_init false --emb_std 0.02 > "runs/$name.out" 2>&1
  done
}

run 5M  256 4 4 2 704  1e8 65536
run 10M 384 4 6 3 1024 2e8 131072
run 20M 512 5 8 4 1408 4e8 262144
