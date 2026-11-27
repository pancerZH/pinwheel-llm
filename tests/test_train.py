from dataclasses import replace

import pytest
import torch

from pinwheel.model import GPT, GPTConfig
import json
import random

import numpy as np

from pinwheel.train import load_weights, rng_state, set_rng_state, tail_averages, trim_logs, update_averages


def test_tail_averages_parses_ema_decays_and_uniform():
    assert tail_averages("") == {}
    assert tail_averages("0.99, uniform") == {"ema0.99": 0.99, "uniform": None}


def test_update_averages_keeps_an_ema_and_a_plain_mean():
    steps = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 6.0]), torch.tensor([5.0, 4.0])]
    params = [torch.zeros(2), torch.zeros(1)]
    averages = {name: (decay, [torch.empty_like(p) for p in params])
                for name, decay in tail_averages("0.9,uniform").items()}
    ema = None
    for n, w in enumerate(steps, start=1):
        params[0].copy_(w)
        params[1].fill_(n)
        update_averages(averages, params, n)
        ema = w.clone() if ema is None else 0.9 * ema + 0.1 * w
    assert torch.allclose(averages["ema0.9"][1][0], ema)
    assert torch.allclose(averages["uniform"][1][0], torch.stack(steps).mean(0))
    assert torch.allclose(averages["uniform"][1][1], torch.tensor([2.0]))  # mean of 1, 2, 3


def test_load_weights_turns_a_plain_model_into_a_looped_one(tmp_path):
    cfg = GPTConfig(vocab_size=64, n_layer=4, n_head=2, n_kv_head=1, d_model=16, d_ff=32, max_seq_len=8)
    plain = GPT(cfg)
    torch.save({"model": plain.state_dict()}, tmp_path / "plain.pt")
    looped = GPT(replace(cfg, n_loop=2))
    assert load_weights(looped, tmp_path / "plain.pt") == ["loop_emb"]
    assert torch.equal(looped.blocks[2].attn.wq.weight, plain.blocks[2].attn.wq.weight)
    assert torch.count_nonzero(looped.loop_emb) == 0
    idx = torch.randint(0, 64, (1, 8))
    looped_once = looped(idx, idx, n_loop=1)  # one pass is the plain model again
    assert torch.allclose(looped_once, plain(idx, idx))
    with pytest.raises(ValueError):
        load_weights(GPT(replace(cfg, n_layer=3)), tmp_path / "plain.pt")


def test_rng_state_round_trip_continues_the_data_order():
    loop_rng = random.Random(0)
    np.random.seed(0)
    np.random.randint(0, 100, size=5)
    saved = rng_state(loop_rng) if torch.cuda.is_available() else None
    if saved is None:  # rng_state saves the CUDA generator too
        pytest.skip("needs CUDA")
    expected = np.random.randint(0, 10**9, size=8), loop_rng.random(), torch.rand(3)
    np.random.seed(123)
    set_rng_state(saved, loop_rng)
    got = np.random.randint(0, 10**9, size=8), loop_rng.random(), torch.rand(3)
    assert (got[0] == expected[0]).all() and got[1] == expected[1] and torch.equal(got[2], expected[2])


def test_trim_logs_keeps_what_came_before_the_checkpoint(tmp_path):
    lines = [{"step": 0, "val_bpb": 3.4}, {"step": 0, "loss": 9.0}, {"step": 100, "val_bpb": 1.9},
             {"step": 100, "loss": 2.0}, {"step": 110, "loss": 1.9}, {"step": 200, "val_bpb": 1.6}]
    (tmp_path / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in lines))
    (tmp_path / "samples.md").write_text("## step 100\n\n- a\n\n## step 200\n\n- b\n\n")
    trim_logs(tmp_path, 100)
    kept = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert kept == lines[:3]  # the eval at step 100 ran before the checkpoint; step 100's training after it
    assert (tmp_path / "samples.md").read_text() == "## step 100\n\n- a\n\n"
