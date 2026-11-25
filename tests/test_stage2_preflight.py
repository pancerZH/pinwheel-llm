import math
import sys
from datetime import datetime
from pathlib import Path

import torch

from pinwheel.model import GPT, GPTConfig
from pinwheel.train import TrainConfig, lr_at

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from stage2_preflight import BATCH_SWEEP, next_safe_start, parse_blackouts, pick_best_batch_config


def test_lr_at_cosine_and_wsd_and_cooldown_from():
    # Cosine unchanged
    cos = TrainConfig(lr=2e-3, min_lr_frac=0.1, warmup_steps=100, schedule="cosine")
    assert math.isclose(lr_at(0, cos, 1000), 2e-3 / 100)
    assert math.isclose(lr_at(99, cos, 1000), 2e-3)
    assert math.isclose(lr_at(1000, cos, 1000), 2e-4)

    # WSD: warmup -> constant plateau -> (1 - sqrt) cooldown
    wsd = TrainConfig(lr=2e-3, min_lr_frac=0.0, warmup_steps=100, schedule="wsd", cooldown_frac=0.2)
    assert math.isclose(lr_at(99, wsd, 1000), 2e-3)
    assert math.isclose(lr_at(100, wsd, 1000), 2e-3)
    assert math.isclose(lr_at(799, wsd, 1000), 2e-3)
    assert math.isclose(lr_at(800, wsd, 1000), 2e-3)  # t=0 -> 1 - sqrt(0) = 1
    assert math.isclose(lr_at(850, wsd, 1000), 2e-3 * (1 - math.sqrt(0.25)))  # t=0.25 -> 0.5
    assert math.isclose(lr_at(1000, wsd, 1000), 0.0)

    # Pure cooldown from a saved stable checkpoint skips warmup
    cd = TrainConfig(lr=2e-3, min_lr_frac=0.0, warmup_steps=100, schedule="wsd", cooldown_from="stable.pt")
    assert math.isclose(lr_at(0, cd, 200), 2e-3)
    assert math.isclose(lr_at(50, cd, 200), 2e-3 * 0.5)
    assert math.isclose(lr_at(200, cd, 200), 0.0)


def test_stable_ckpt_weights_only_roundtrip(tmp_path):
    cfg = GPTConfig(vocab_size=64, n_layer=1, n_head=2, n_kv_head=2, d_model=16, d_ff=32, max_seq_len=16)
    m = GPT(cfg)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 64, (2, 8))
    m(x, x).backward()
    opt.step()
    p = tmp_path / "stable.pt"
    torch.save({"model": m.state_dict(), "optimizer": opt.state_dict(), "step": 1, "tokens": 16}, p)
    loaded = torch.load(p, map_location="cpu", weights_only=True)
    m2 = GPT(cfg)
    opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-3)
    m2.load_state_dict(loaded["model"])
    opt2.load_state_dict(loaded["optimizer"])
    assert loaded["step"] == 1


def test_next_safe_start_blackout_guard():
    blackouts = parse_blackouts(["03:00-03:35", "07:30-07:50", "18:00-18:20"])

    # 20:00 + 115 min = 21:55 -> clear
    t0 = datetime(2026, 11, 24, 20, 0)
    assert next_safe_start(t0, 115, blackouts) == (t0, None)

    # 01:20 + 115 min = 03:15 overlaps 03:00-03:35 -> waits until 03:35
    t1 = datetime(2026, 11, 25, 1, 20)
    assert next_safe_start(t1, 115, blackouts) == (datetime(2026, 11, 25, 3, 35), "03:00-03:35")

    # Inside 03:00-03:35 -> waits until 03:35
    t2 = datetime(2026, 11, 25, 3, 10)
    assert next_safe_start(t2, 20, blackouts) == (datetime(2026, 11, 25, 3, 35), "03:00-03:35")

    # 02:40 + 15 min = 02:55 -> clear before 03:00!
    t3 = datetime(2026, 11, 25, 2, 40)
    assert next_safe_start(t3, 15, blackouts) == (t3, None)

    # 05:50 + 115 min = 07:45 overlaps 07:30-07:50 -> waits until 07:50
    t4 = datetime(2026, 11, 25, 5, 50)
    assert next_safe_start(t4, 115, blackouts) == (datetime(2026, 11, 25, 7, 50), "07:30-07:50")

    # 16:20 + 115 min = 18:15 overlaps 18:00-18:20 -> waits until 18:20
    t5 = datetime(2026, 11, 24, 16, 20)
    assert next_safe_start(t5, 115, blackouts) == (datetime(2026, 11, 24, 18, 20), "18:00-18:20")


def test_pick_best_batch_config():
    res = {"1e-v0-lr2e-3": 1.1212, "2a-bs256k-lr1.5e-3": 1.1150, "2a-bs128k-lr1e-3": 1.1125}
    name, bs, lr, bpb = pick_best_batch_config(BATCH_SWEEP, res)
    assert (name, bs, lr, bpb) == ("2a-bs128k-lr1e-3", 131072, 1e-3, 1.1125)
