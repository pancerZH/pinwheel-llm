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
The training defaults have since moved on with stage 2: Muon for the block matrices, QK-Norm, zero-initialized residual projections and a 10× smaller embedding init.

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
- **No sign of overfitting, but not proven.** Train and val BPB are almost the same, and only about 60% of the training tokens were ever seen. That alone does not rule out memorization: 28% of the evaluated TinyStories val tokens sit in 64-token spans that also appear verbatim in the training file (2,350 of the 8,692 val stories have exact copies there). The val windows without such overlap score 0.5151 against 0.5171 overall, so the copies do not make val easier.

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
- **On our main metric, the 4B-token run already ties GPT-2 124M**: on the same validation text (each 1024-token window decoded and re-tokenized per document by GPT-2's own tokenizer; the same pipeline reproduces 1.0250 for our model), GPT-2 reaches 1.0287 val BPB against our 1.0250, a 0.004 gap inside the noise floor. The text is FineWeb-Edu, the same distribution we train on and not GPT-2's WebText, so this is a home-field tie; on the downstream tasks GPT-2 is still ahead everywhere except ARC-Easy.
- TinyStories teaches grammar (BLiMP 62.7) but no world knowledge (the other three are at chance).

### Stage 2: v1, shape and optimizer ablations (M2) ✅
Check published results before each experiment; where a reliable result exists, adopt it and run one confirmation instead of a sweep.
- ✅ Batch size first: DeepSeek LLM's hyperparameter scaling law predicts about 160k tokens for v0's compute (v0 uses 524k), and LR 2e-3, which matches what 1e found. Result (1B tokens, cosine, best LR per batch): 512k / 256k / 128k reach 1.1212 / 1.1006 / 1.0996, at the same throughput. 128k and 256k are tied, and both beat 512k by 0.02, well above the noise floor. New default: 128k, LR 1.4e-3. This agrees with the ~160k prediction but does not locate the optimum: the best LR of both sweeps sat at its upper edge, and the fixed 100-step warmup gave 128k and 512k 13M and 52M warmup tokens.
- ✅ WSD instead of cosine, with a (1 − sqrt) cooldown: adopted from Hägele et al. 2024 and confirmed with one run; it also lets one long run give results at several token budgets. Result: at 128k / 1.4e-3, the WSD recipe (a 20% cooldown to 0) reaches **1.0915**, 0.008 better than cosine decaying to 10% of the peak (one seed, within the stage 2 noise floor below, so this rests mainly on the paper). The curve shape and the final LR changed together, so the shape alone was not measured. **New baseline: `2b-wsd-1b`**, 0.030 below the stage 1 baseline (HellaSwag 27.9 / ARC-Easy 38.9 / PIQA 58.6 / BLiMP 77.3). A 100M cooldown branched from the 500M point of the same run reaches 1.1252, with 60% of the tokens.
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
- ✅ Muon for the 70 weight matrices inside the blocks (`--optimizer muon`), AdamW for the embedding and norms. The update is scaled to AdamW's RMS (Moonshot), so AdamW's LR and weight decay carry over. Predictions written before the runs all held:
  - 300M-token check-up (WSD, from scratch): AdamW 1.1732; Muon at LR 1e-3 / **2e-3** / 4e-3 reaches 1.1507 / **1.1449** / 1.1553. The AdamW optimum is also Muon's; in the constant-LR phase AdamW needs **1.41×** the tokens to match Muon (predicted 1.3–1.4×, Wen et al. 2025); at 4e-3 that falls from 1.37× to 1.12× and the largest attention score doubles to 58.
  - 1B tokens, two seeds: **1.0716** / 1.0708 (mean **1.0712**), 0.0178 below the three-seed AdamW mean 1.0890 (predicted 0.015–0.020; the two seeds differ by 0.0008). On the data-scaling curve that is worth about 1.4× the tokens. Throughput is 2.5% lower (the optimizer step takes 21 ms vs 2 ms). **New default and baseline: Muon (`2e-muon-1b`).**
- ✅ QK-Norm (`--qk_norm`): an RMSNorm on every head's queries and keys before RoPE, so the attention scores can no longer grow with the weights. 300M tokens on Muon: at LR 2e-3, 1.1449 → 1.1414 (−0.0035, within noise); at 4e-3 the largest attention score falls from 57.8 to 18.5 and doubling the LR costs 0.0040 instead of 0.0104 (the 4e-3 QK-Norm run was on the other machine, so that gap is not yet confirmed). The best LR stays at 2e-3. Throughput is 1.5% lower.
- ✅ Zero-initialized `wo` and `w_down` (`--zero_init`), so every block starts as the identity, and a 10× smaller embedding init (`--emb_std 0.002`). With tied embeddings, zero-init alone makes the step-0 model copy its input: the final hidden state is the input token's own embedding, which scores 0.02 · 640 = 12.8 against itself (initial loss 12.77; ln 8192 = 9.01). The small embedding init removes that. 300M tokens on Muon + QK-Norm (reference 1.1414): zero-init alone 1.1432 (+0.0018), the small embedding alone 1.1360 (−0.0054), both 1.1327 / 1.1333 (two seeds, mean −0.0084). Most of the gain is the small embedding. At the end of the reference run the embedding still holds 0.78× its random initial matrix (27.6% of the energy for the 512 most frequent tokens; 3.0% with the small init): the output layer is the same table, so the hidden states learn to point at those random vectors, which then get reinforced. That explanation is not tested on its own.
- ✅ All three together at 1B tokens, two seeds: **1.0620** / 1.0617 (mean **1.0619**, clean 1.0723), 0.0094 below Muon's mean (clean −0.0093), past the two-seed threshold of 0.007 and better than predicted (1.063–1.067). The lead is about 0.013 in the constant-LR phase and 0.0094 after the cooldown. The benchmarks are within noise (HellaSwag 28.3 / 28.1 vs 28.0 / 28.0). **New defaults and baseline: `2e-stack-1b`**, 0.027 below the AdamW baseline.
- Two machines: a second GPU (RTX 5080 under WSL2) now runs some experiments. Rerunning a 300M run with the same seed there gives 1.1736 vs 1.1732, so val BPB is comparable across machines; the runs are not bit-identical (early gaps are as large as a seed change), but the same seed means the same data order, so a rerun on the other machine is not an independent seed. Throughput is only compared on the same machine.
- Val overlap (found in a review): 11% of the evaluated FineWeb val tokens lie in 64-token spans that appear verbatim in the 4B training file, which starts with the 1B one (543 of 2,048 windows; `scripts/decontam.py`). The main metric stays as it is, so old and new runs remain comparable; runs now also report `val_bpb_clean` on the other 1,505 windows, and every saved checkpoint was rescored. It is about 0.01 higher for every run (0.008 at 300M, 0.010 at 1B, 0.011 at 4B) and no comparison changes sign: Muon vs AdamW −0.0174 clean (−0.0178 full), 4B vs 1B −0.059 (−0.060). On the clean windows GPT-2 124M scores 1.0352 and the 4B run 1.0361: still a tie, but GPT-2 is now the one ahead, since our training file holds copies of the val text and GPT-2's does not.
- ✅ Same parameter count, deeper and narrower, as MobileLLM and SmolLM2 chose: d448/L20 at 300M tokens reaches 1.1330, an exact tie with v0's two-seed mean of 1.1330 (with a single KV group: 1.1375). It is 16% slower (137k vs 164k tokens/s: more attention FLOPs per token, and the smaller matrices run at 59% MFU instead of 65%), which at equal wall-clock costs about 0.007. **Not adopted**: v0's d640/L10 stays.
- Not tested: vocab size and tied vs. untied embeddings. Tao et al. 2024 put v0's optimal vocab at about 7k–11k, with 8k–16k a flat basin, so a change would stay under the noise floor; 8k tied stays.
- ✅ Tail averaging of the weights (`--tail_avg`), as in modded-nanogpt: averages kept over the cooldown (the last 20% of steps) and scored against the same run's final weights, so seed noise does not enter. 1B tokens: final weights 1.0636; EMA 0.99 +0.0002, EMA 0.998 +0.0007, a plain mean +0.0033. **Not adopted**: the cooldown to LR 0 already removes the noise an average would cancel. The run doubles as the baseline's third seed (1.0620 / 1.0617 / 1.0636, mean 1.0624).
- Not done: ReLU² (mostly a throughput change at equal parameters) and a logit softcap (our output logits peak at 24.9 and never pass 30, so a cap of 30 barely bends them), as well as cautious weight decay and FP8 training.
- ✅ Goal met: v1 beats v0 on val BPB by more than the noise floor. **v1 = `2e-stack-1b`**: v0's shape, trained with batch 128k, LR 2e-3, WSD and Muon, with QK-Norm, zero-initialized residual projections and the small embedding init. Three seeds at 1B tokens: 1.0620 / 1.0617 / 1.0636, mean **1.0624**, 0.0266 below v0 on the stage 2 recipe with AdamW (three-seed mean 1.0890; worth about 1.8× the tokens on the data-scaling curve) and 0.059 below the stage 1 baseline. About 0.032 of the gain over stage 1 comes from the recipe (batch size, schedule, LR) and 0.027 from the optimizer and the init.

### Stage 3: v2, architecture experiments (M3): TODO
- Multi-token prediction (MTP).
- Looped (recurrent-depth) layers, compared at equal parameters and at equal compute.
- Goal: a clear conclusion for each, positive or negative.

### Stage 4: v3, data and distillation (M4) 🔄
- Overtrain to 5–10B tokens, then anneal on high-quality data.
- ✅ First data-scaling runs (the baseline config on a fresh 4B-token set, same validation set): 1B / 2.5B / 4B tokens reach 1.0850 / 1.0430 / **1.0250** (the 2.5B point is a 0.5B cooldown branched from the 2B checkpoint of the 4B run). Along this curve each doubling of the budget gains about 0.03 (0.032 then 0.027), so at 1B (20 tokens per parameter) the model is far from its capacity. The budget changes two things at once: the loader samples windows with replacement, so 1B / 2.5B / 4B cover about 0.63B / 1.86B / 2.53B distinct file positions, and the curve does not separate more steps from more new text. At 4B: HellaSwag **29.4**, 0.5 short of GPT-2 124M; CORE +0.273 → +0.302.
- Distillation: synthetic data first, then logit distillation from a larger teacher that shares our tokenizer.
- Stretch goal: HellaSwag ≥ 29.9% on our harness, the level of GPT-2 124M (llm.c reports 29.55% with per-token normalization; random guessing gets 25%).

### Stage 5: v4, post-training (M5): TODO
- Chat fine-tuning, then one narrow skill: verifiable reasoning with RL (GRPO), natural language to shell / SQL / JSON commands, or reading-comprehension answers from a given text.
- Goal: a measurable gain on an automatic eval, plus a CPU inference demo.

## Code

```
pinwheel/model.py            model, KV cache, sampling
pinwheel/muon.py             Muon optimizer for the 2-D weights in the blocks
pinwheel/train.py            training loop; every config field is a command-line flag
pinwheel/eval.py             val BPB and the clean val BPB; scores checkpoints
pinwheel/bench.py            zero-shot benchmarks (HellaSwag, ARC-Easy, PIQA, BLiMP, SciQ, LAMBADA) and paired tests
pinwheel/data.py             memmapped token shards
pinwheel/tokenizer.py        BPE tokenizer and per-token byte counts
pinwheel/sample.py           sample from a checkpoint
scripts/prepare.py           train the tokenizer, tokenize datasets
scripts/plot.py              plot loss / BPB curves
scripts/scaling.py           fit and plot the mini scaling law (run via scripts/scaling.sh)
scripts/tree.py              build interactive experiment tree webpage (runs/tree.html)
scripts/flow.py              build the training-step data-flow webpage from checkpoints (runs/flow.html)
scripts/gpt2_ref.py          score OpenAI GPT-2 124M on our bench (the stretch-goal reference)
scripts/stage2_preflight.py  Stage 2 batch-size sweep and WSD confirmation runner
scripts/muon.sh              Stage 2e Muon check-up: 300M-token AdamW reference and Muon LR sweep
scripts/decontam.py          mark val tokens found verbatim in a training file (for val_bpb_clean)
```
