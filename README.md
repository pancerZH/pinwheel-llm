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
uv run python -m pinwheel.bench ckpt/<run>.pt        # HellaSwag / ARC-Easy / PIQA / BLiMP, about 30 s
uv run python scripts/plot.py runs/<a> runs/<b>      # compare loss / BPB curves
uv run pytest
```

Each run writes `runs/<run>/` (config, log, samples) and `ckpt/<run>.pt`, and adds a row to `runs/results.tsv` (val BPB and HellaSwag) and `runs/bench.tsv` (all four benchmarks). `data/`, `runs/` and `ckpt/` are not in git.

## The model (v0)

A Llama-style decoder with RMSNorm, RoPE, SwiGLU, grouped-query attention and tied embeddings:
10 layers, width 640, 10 heads (5 KV heads), FFN 1728, vocab 8192, context 1024, **50.7M parameters**.
It is trained with AdamW at batch 128k tokens and LR 2e-3, with a WSD schedule (warmup, constant, then a 20% (1 − sqrt) cooldown to 0), bf16 and `torch.compile`, at about 170k tokens/s (68% MFU). Stage 1 used batch 512k and cosine LR.

## Milestones

✅ done · 🔄 in progress · TODO not started

### Stage 0: setup (M0) ✅
- ✅ Tokenizer: byte-level BPE with 8192 tokens, trained on 200k FineWeb-Edu documents.
- ✅ Data: TinyStories (549M train tokens) and FineWeb-Edu (1B train tokens).
- ✅ Sanity checks: initial loss 9.13 (expected ≈ ln 8192 = 9.01), and a single batch can be overfit to loss 0.0003.

### Stage 1: v0 baseline (M1)
| Step | What | Status / result |
|---|---|---|
| 1a | Train on TinyStories, 0.5B tokens | ✅ val BPB **0.517** in 0.82 h. Writes coherent short stories. Prompts about facts, math or code drift back into children's stories. |
| 1b | Learning-rate sweep on FineWeb-Edu, 100M tokens per run | ✅ Best LR **1e-3** (val BPB 1.668), with 2e-3 close behind (1.681). But at 1B tokens 2e-3 wins clearly (see 1e), so this 100M-token sweep picked the wrong LR. Warmup matters: with only 20 warmup steps, every LR ≥ 1e-3 stalled early and the sweep wrongly favored 5e-4. |
| 1c | Baseline: FineWeb-Edu, 1B tokens, warmup 100 | ✅ LR 1e-3: val BPB 1.147 in 1.64 h. That LR turned out too low (see 1e), so **the official baseline is `1e-v0-lr2e-3`: LR 2e-3, val BPB 1.1212**. Every later experiment is compared against it. Writes fluent, on-topic encyclopedia-style text, but facts are often wrong, it sometimes repeats itself, and it cannot do math or code. |
| 1d | Inference: KV cache, temperature / top-k / top-p, CLI | ✅ built during setup |
| 1e | Noise floor: the 1c config with 3 seeds, plus LR 2e-3 | ✅ Seeds 0 / 1 / 2: 1.1471 / 1.1442 / 1.1445, a spread of 0.003 (std 0.0016). A difference under about 0.005 between two runs does not count. LR 2e-3 reaches **1.1212**, 0.024 better than the seed mean, far above the noise; LR 4e-3 gives 1.1450, no better than 1e-3. So 2e-3 is the best LR and the new default. |
| side | Mini scaling law: 5M / 10M / 20M / 50M models at about 20 tokens per parameter (`scripts/scaling.sh`; the 50M point is 1c / 1e) | ✅ With the best LR per size, val BPB ≈ 9.32 · C^-0.052 (C = training FLOPs): each doubling of compute lowers BPB by about 3.6%. Best LR is 4e-3 for 5M–19M (8e-3 is worse, and worse the bigger the model) and 2e-3 for 50M (1e-3 and 4e-3 both reach about 1.145). Plot: `scripts/scaling.py`. |

What 1a taught us:
- **BPB depends on the data.** The same model scores 0.517 on TinyStories but 2.58 on FineWeb-Edu, because TinyStories uses simple words and repetitive plots.
- **No overfitting.** Train and val BPB are almost the same, and only about 60% of the training tokens were ever seen.

What the scaling law taught us:
- **Small models want larger learning rates.** 5M–19M models did best at 4e-3, while 8e-3 hurt more the bigger the model.
- **A sweep is done only when the best LR is inside the grid.** 2e-3 and then 4e-3 won at the edge; only 8e-3 losing closed the bracket.
- **Four points fit one power law within ±0.008 BPB** (after the 50M LR fix; ±0.016 before). The ends still sit above the line and the middle below it, by 3–5× the seed noise. Now that the 50M LR is bracketed, an irreducible-loss floor is the likelier reason, but four points cannot prove it.

Zero-shot benchmarks (`pinwheel/bench.py`; HellaSwag, ARC-Easy and PIQA use acc_norm, BLiMP uses acc):

| Checkpoint | HellaSwag | ARC-Easy | PIQA | BLiMP |
|---|---|---|---|---|
| chance | 25.0 | 25.0 | 50.0 | 50.0 |
| 1a, TinyStories | 26.5 | 27.0 | 51.9 | 62.7 |
| 5M / 10M / 20M (scaling law) | 26.2 / 26.7 / 26.9 | 32.7 / 33.3 / 35.2 | 52.9 / 53.1 / 55.5 | 63.9 / 67.0 / 69.5 |
| 50M, LR 1e-3, seeds 0 / 1 / 2 | 27.4 / 27.2 / 27.2 | 37.8 / 37.9 / 38.0 | 58.1 / 56.5 / 56.5 | 73.9 / 74.2 / 73.0 |
| 50M, LR 4e-3 | 27.4 | 37.8 | 57.3 | 74.9 |
| **50M, LR 2e-3 (baseline)** | **27.6** | **38.1** | **57.5** | **74.2** |
| OpenAI GPT-2 124M (reference) | 29.9 | 40.5 | 62.5 | 82.5 |

What the benchmarks taught us:
- **Every score rises with model size**, and an untrained model gets chance (26.3 on HellaSwag), so the harness works.
- **They are much noisier than val BPB.** Across seeds, PIQA moves 1.6 points and BLiMP 1.1. LR 2e-3 beats 1e-3 by 0.024 BPB, far above the noise, but by only 0.2–0.4 points on HellaSwag, within the noise. So val BPB stays the main metric and the benchmarks are a sanity check.
- **HellaSwag is 27.6; GPT-2 124M scores 29.9 on the same harness, so the stretch goal needs 2.3 more points.** llm.c reports 29.55 for GPT-2 because it normalizes by tokens instead of bytes; our harness reproduces that number exactly under llm.c's protocol, so the two rulers differ only in that choice.
- GPT-2 is scored in fp32, the precision it was trained in (`scripts/gpt2_ref.py`, which also adds SciQ 68.2 and LAMBADA 33.7). Under the bf16 autocast our runs use, its HellaSwag rises to 30.3, while our bf16-trained models move by under 0.1 point between the two precisions; fp32 keeps the comparison fair.
- TinyStories teaches grammar (BLiMP 62.7) but no world knowledge (the other three are at chance).

### Stage 2: v1, shape and optimizer ablations (M2) 🔄
Check published results before each experiment; where a reliable result exists, adopt it and run one confirmation instead of a sweep.
- ✅ Batch size first: DeepSeek LLM's hyperparameter scaling law predicts about 160k tokens for v0's compute (v0 uses 524k), and LR 2e-3, which matches what 1e found. Result (1B tokens, cosine, best LR per batch): 512k / 256k / 128k reach 1.1212 / 1.1006 / 1.0996, at the same throughput. 128k and 256k are tied, and both beat 512k by 0.02, well above the noise floor. New default: 128k, LR 1.4e-3.
- ✅ WSD instead of cosine, with a (1 − sqrt) cooldown: adopted from Hägele et al. 2024 and confirmed with one run; it also lets one long run give results at several token budgets. Result: at 128k / 1.4e-3, a 20% cooldown to 0 reaches **1.0915**, 0.008 better than cosine (one seed, within the stage 2 noise floor below, so this rests mainly on the paper). **New baseline: `2b-wsd-1b`**, 0.030 below the stage 1 baseline (HellaSwag 27.9 / ARC-Easy 38.9 / PIQA 58.6 / BLiMP 77.3). A 100M cooldown branched from the 500M point of the same run reaches 1.1252, with 60% of the tokens.
- ✅ LR at batch 128k with WSD: 1.4e-3 / 2e-3 / 2.8e-3 reach 1.0915 / **1.0850** / 1.0849 (one seed each). All three are within the seed noise below, so the optimum is flat from 1.4e-3 to 2.8e-3. **New default and baseline: LR 2e-3 (`2b-wsd-lr2e-3`, 1.0850).** Shrinking the LR with √batch, a common rule, lost about 0.01 at both 256k and 128k: here the best LR barely depends on the batch size, as the DeepSeek scaling law says.
- ✅ Noise floor of the new baseline: seeds 0 / 1 / 2 reach **1.0850** / 1.0901 / 1.0919 (mean 1.0890, std 0.0036, spread 0.0069), over twice the stage 1 noise. It is not eval sampling: on the full validation set (12.5M tokens, in five chunks) every chunk shows the same gap in the same order. The gap forms by about 150M tokens and the cooldown does not shrink it, and seed 0 was the worst seed in stage 1 but the best here, so reusing a seed does not cancel the noise. With 3 seeds per config the two noise estimates are not significantly different (F = 5.1, 5% critical value 19); a 4× smaller batch at 2× the LR (8× the LR / batch ratio) may make runs noisier, but that is untested. **From here on, a difference under 0.01 between two single runs does not count** (2·√2·0.0036 ≈ 0.01).
- ✅ Vanilla baseline: the same model with GPT-2's LayerNorm, learned absolute positions, GELU MLP and plain multi-head attention (equal parameters, `--arch gpt2 --n_kv_head 10 --d_ff 2216`, same recipe) reaches 1.1072, **0.022 worse** than the Llama-style 1.0850 (0.018 worse than its three-seed mean, 5× the seed std). The gap is largest early (0.12 at 66M tokens, 0.03 at 330M). It trains 5% faster (179k vs 170k tokens/s), far too little to make up the gap.
- ✅ Vanilla baseline on the old stage 1 recipe (512k, cosine, LR 2e-3): **1.1543**, 0.033 worse than v0's 1.1212 on the same recipe. This is now the root of the experiment tree. The gap is 0.011 larger than on the new recipe (about 2σ for four single runs), but aligned by optimizer step the two gaps match (0.047 vs 0.052 at step 1000, 0.033 vs 0.037 near step 1900): the old recipe has 4× fewer steps, and the gap shrinks with steps, not tokens.

  | | Old recipe (512k, cosine) | New recipe (128k, WSD) | Recipe gain |
  |---|---|---|---|
  | Vanilla GPT-2 | 1.1543 | 1.1072 | −0.047 |
  | Llama-style | 1.1212 | 1.0850 | −0.036 |
  | Architecture gain | −0.033 | −0.022 | |
- Lesson: on ARC-Easy and SciQ, the log-prob of the correct answer alone (`target_bpb`) mostly measures whether the model expects a bare phrase after "Answer:". It moves all options together and varies by about 0.035 between seeds (val BPB: 0.0016–0.0036), so compare those tasks by accuracy and `prob`; `bench.py --compare` marks them.
- Muon vs. a tuned AdamW. Expect about 1.3× (Wen et al. 2025), not the 2× often quoted.
- Same parameter count, different shapes: v0 vs. one deep-and-narrow shape (d448/L20), as MobileLLM and SmolLM2 suggest.
- Vocab size (4k / 8k / 16k), and tied vs. untied embeddings.
- A few modded-nanogpt tricks, such as QK-Norm, ReLU² and logit softcap.
- Goal: v1 beats v0 on val BPB by more than the noise floor.

### Stage 3: v2, architecture experiments (M3): TODO
- Multi-token prediction (MTP).
- Looped (recurrent-depth) layers, compared at equal parameters and at equal compute.
- Goal: a clear conclusion for each, positive or negative.

### Stage 4: v3, data and distillation (M4) 🔄
- Overtrain to 5–10B tokens, then anneal on high-quality data.
- ✅ First data-scaling runs (the baseline config on a fresh 4B-token set, same validation set): 1B / 2.5B / 4B tokens reach 1.0850 / 1.0430 / **1.0250** (the 2.5B point is a 0.5B cooldown branched from the 2B checkpoint of the 4B run). Each doubling of tokens gains about 0.03, only slightly less at the end (0.032 then 0.027), so at 1B (20 tokens per parameter) the model is far from its capacity. At 4B: HellaSwag **29.4**, 0.5 short of GPT-2 124M; CORE +0.273 → +0.302.
- Distillation: synthetic data first, then logit distillation from a larger teacher that shares our tokenizer.
- Stretch goal: HellaSwag ≥ 29.9% on our harness, the level of GPT-2 124M (llm.c reports 29.55% with per-token normalization; random guessing gets 25%).

### Stage 5: v4, post-training (M5): TODO
- Chat fine-tuning, then one narrow skill: verifiable reasoning with RL (GRPO), natural language to shell / SQL / JSON commands, or reading-comprehension answers from a given text.
- Goal: a measurable gain on an automatic eval, plus a CPU inference demo.

## Code

```
pinwheel/model.py            model, KV cache, sampling
pinwheel/train.py            training loop; every config field is a command-line flag
pinwheel/eval.py             val BPB
pinwheel/bench.py            zero-shot benchmarks (HellaSwag, ARC-Easy, PIQA, BLiMP, SciQ, LAMBADA) and paired tests
pinwheel/data.py             memmapped token shards
pinwheel/tokenizer.py        BPE tokenizer and per-token byte counts
pinwheel/sample.py           sample from a checkpoint
scripts/prepare.py           train the tokenizer, tokenize datasets
scripts/plot.py              plot loss / BPB curves
scripts/scaling.py           fit and plot the mini scaling law (run via scripts/scaling.sh)
scripts/tree.py              build interactive experiment tree webpage (runs/tree.html)
scripts/gpt2_ref.py          score OpenAI GPT-2 124M on our bench (the stretch-goal reference)
scripts/stage2_preflight.py  Stage 2 batch-size sweep and WSD confirmation runner
```
