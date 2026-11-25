import math

import numpy as np
import torch
import torch.nn.functional as F

from pinwheel.bench import (
    answer_logprobs,
    append_bench_row,
    centered_score,
    mcnemar_test,
    paired_diff_test,
    score_item,
    stratified_takes,
)
from pinwheel.model import GPT, GPTConfig
from pinwheel.tokenizer import BOS, train_tokenizer

TINY = GPTConfig(vocab_size=300, n_layer=2, n_head=4, n_kv_head=2, d_model=64, d_ff=128, max_seq_len=64)


def test_batched_answer_logprobs_and_exact_match_one_by_one():
    torch.manual_seed(0)
    tok = train_tokenizer(["The cat sat on the mat. A dog ran in the park."] * 20, vocab_size=TINY.vocab_size)
    model = GPT(TINY).eval()
    pairs = [("The cat", " sat on the mat."), ("A dog ran", " in the park."), ("", "The dog sat.")]
    batched, exact = answer_logprobs(model, tok, pairs, batch_size=3, return_exact=True)

    for (ctx, ans), got, got_ex in zip(pairs, batched, exact):
        c, a = [tok.token_to_id(BOS)] + tok.encode(ctx).ids, tok.encode(ans).ids
        x = torch.tensor([c + a])
        logp = F.log_softmax(model(x).float(), dim=-1)[0]
        want = sum(logp[len(c) - 1 + j, t] for j, t in enumerate(a))
        want_ex = all(logp[len(c) - 1 + j].argmax().item() == t for j, t in enumerate(a))
        assert abs(got.item() - want.item()) < 1e-4
        assert bool(got_ex.item()) is want_ex


def test_score_item_continuous_probes():
    # Two answers: " a" (2 bytes, logp=-2.0 -> norm=-1.0), " bbbb" (5 bytes, logp=-2.5 -> norm=-0.5).
    # Label 0 wins raw acc (-2.0 > -2.5) but loses byte-norm acc_norm (-1.0 < -0.5).
    r = score_item([-2.0, -2.5], [" a", " bbbb"], label=0, use_norm=True)
    assert r["acc"] == 1 and r["acc_norm"] == 0
    p0 = math.exp(-1.0) / (math.exp(-1.0) + math.exp(-0.5))
    assert math.isclose(r["prob"], p0, rel_tol=1e-6)
    assert math.isclose(r["margin"], -0.5, rel_tol=1e-6)
    assert math.isclose(r["brier"], (p0 - 1.0) ** 2 + (1.0 - p0) ** 2, rel_tol=1e-6)
    assert math.isclose(r["target_bpb"], 2.0 / (math.log(2) * 2), rel_tol=1e-6)

    # Single-target (LAMBADA-style): exact match + target probability + target_bpb.
    r1 = score_item([-1.0], [" dog"], label=0, exact=True, use_norm=False)
    assert r1["acc"] == 1 and math.isnan(r1["margin"])
    assert math.isclose(r1["prob"], math.exp(-1.0), rel_tol=1e-6)
    assert math.isclose(r1["brier"], (1.0 - math.exp(-1.0)) ** 2, rel_tol=1e-6)
    assert math.isclose(r1["target_bpb"], 1.0 / (math.log(2) * 4), rel_tol=1e-6)


def test_stratified_takes_covers_all_blimp_groups():
    takes = stratified_takes(500, 67)
    assert len(takes) == 67
    assert sum(takes) == 500
    assert min(takes) == 7 and max(takes) == 8


def test_mcnemar_paired_diff_and_centered():
    n01, n10, p = mcnemar_test([0, 0, 0, 1] * 25, [1, 1, 0, 1] * 25)
    assert (n01, n10) == (50, 0) and p < 1e-10

    a = np.zeros(670, dtype=np.float64)
    b = np.full(670, 0.25, dtype=np.float64)
    mean, se, p_val = paired_diff_test(a, b, n_clusters=67)
    assert math.isclose(mean, 0.25) and se == 0.0 and p_val == 0.0
    assert math.isclose(centered_score("hellaswag", {"acc_norm": 0.25}), 0.0)
    assert math.isclose(centered_score("hellaswag", {"acc_norm": 1.0}), 1.0)


def test_append_bench_row_extends_columns(tmp_path):
    tsv = tmp_path / "bench.tsv"
    tsv.write_text("ckpt\tdate\thellaswag\tarc_easy\nold\t2026-11-24\t0.2700\t0.3800\n")
    append_bench_row(
        "new",
        {"hellaswag": {"acc_norm": 0.28}, "sciq": {"acc_norm": 0.61}, "lambada": {"acc": 0.12}},
        path=tsv,
    )
    lines = tsv.read_text().splitlines()
    assert lines[0] == "ckpt\tdate\thellaswag\tarc_easy\tsciq\tlambada"
    assert lines[1] == "old\t2026-11-24\t0.2700\t0.3800\t\t"
    assert lines[2].endswith("\t0.2800\t\t0.6100\t0.1200")
