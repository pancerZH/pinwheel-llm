# pinwheel-llm

A ~50M-parameter English language model trained from scratch on one RTX 5070 Ti (16 GB), for learning.
The idea is to add one technique at a time and understand every part. Beating state-of-the-art results is not the goal.

**Main metric: validation bits-per-byte (val BPB).** It measures how well the model predicts held-out text. Lower is better.
It does not depend on the vocabulary, but it does depend on the data, so compare runs only on the same validation set.

## Quick start

```
uv run python scripts/prepare.py tokenizer           # 8k byte-level BPE
uv run python scripts/prepare.py tinystories         # or: fineweb-edu --max_tokens 1e9
uv run python -m pinwheel.train --data_dir data/tok8192/tinystories --max_tokens 5e8
uv run python -m pinwheel.sample ckpt/<run>.pt       # interactive sampling
uv run python scripts/plot.py runs/<a> runs/<b>      # compare loss / BPB curves
uv run pytest
```

Each run writes `runs/<run>/` (config, log, samples) and `ckpt/<run>.pt`, and adds a row to `runs/results.tsv`. `data/`, `runs/` and `ckpt/` are not in git.

## The model (v0)

A Llama-style decoder with RMSNorm, RoPE, SwiGLU, grouped-query attention and tied embeddings:
10 layers, width 640, 10 heads (5 KV heads), FFN 1728, vocab 8192, context 1024, **50.7M parameters**.
It is trained with AdamW, warmup + cosine LR, bf16 and `torch.compile`, at about 171k tokens/s (68% MFU).

## Milestones

✅ done · 🔄 in progress · TODO not started

### Stage 0: setup (M0) ✅
- ✅ Tokenizer: byte-level BPE with 8192 tokens, trained on 200k FineWeb-Edu documents.
- ✅ Data: TinyStories (549M train tokens). 🔄 FineWeb-Edu (1B train tokens).
- ✅ Sanity checks: initial loss 9.13 (expected ≈ ln 8192 = 9.01), and a single batch can be overfit to loss 0.0003.

### Stage 1: v0 baseline (M1)
| Step | What | Status / result |
|---|---|---|
| 1a | Train on TinyStories, 0.5B tokens | ✅ val BPB **0.517** in 0.82 h. Writes coherent short stories. Prompts about facts, math or code drift back into children's stories. |
| 1b | Learning-rate sweep: 5 LRs × 100M tokens on FineWeb-Edu | 🔄 running |
| 1c | Official baseline: FineWeb-Edu, 1B tokens | TODO. Every later experiment is compared against this run. |
| 1d | Inference: KV cache, temperature / top-k / top-p, CLI | ✅ built during setup |
| 1e | Noise floor: same config with 2–3 seeds | TODO. A gain smaller than this noise does not count. |
| side | Mini scaling law: 5M / 10M / 20M / 50M models | TODO |

What 1a taught us:
- **BPB depends on the data.** The same model scores 0.517 on TinyStories but 2.58 on FineWeb-Edu, because TinyStories uses simple words and repetitive plots.
- **No overfitting.** Train and val BPB are almost the same, and only about 60% of the training tokens were ever seen.

### Stage 2: v1, shape and optimizer ablations (M2): TODO
- Same parameter count, different shapes: wide and shallow vs. deep and narrow.
- Vocab size (4k / 8k / 16k), and tied vs. untied embeddings.
- Optimizer and schedule: Muon vs. AdamW, and WSD vs. cosine.
- A few modded-nanogpt tricks, such as QK-Norm, ReLU² and logit softcap.
- Goal: v1 beats v0 on val BPB by more than the noise floor.

### Stage 3: v2, architecture experiments (M3): TODO
- Multi-token prediction (MTP).
- Looped (recurrent-depth) layers, compared at equal parameters and at equal compute.
- Goal: a clear conclusion for each, positive or negative.

### Stage 4: v3, data and distillation (M4): TODO
- Overtrain to 5–10B tokens, then anneal on high-quality data.
- Distillation: synthetic data first, then logit distillation from a larger teacher that shares our tokenizer.
- Stretch goal: HellaSwag ≥ 29.5%, the level of GPT-2 124M (random guessing gets 25%).

### Stage 5: v4, post-training (M5): TODO
- Chat fine-tuning, then one narrow skill: verifiable reasoning with RL (GRPO), natural language to shell / SQL / JSON commands, or reading-comprehension answers from a given text.
- Goal: a measurable gain on an automatic eval, plus a CPU inference demo.

## Code

```
pinwheel/model.py      model, KV cache, sampling
pinwheel/train.py      training loop; every config field is a command-line flag
pinwheel/eval.py       val BPB
pinwheel/data.py       memmapped token shards
pinwheel/tokenizer.py  BPE tokenizer and per-token byte counts
pinwheel/sample.py     sample from a checkpoint
scripts/prepare.py     train the tokenizer, tokenize datasets
scripts/plot.py        plot runs
```
