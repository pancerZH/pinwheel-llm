import base64
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import flow  # noqa: E402
from pinwheel.model import GPT, GPTConfig  # noqa: E402
from pinwheel.muon import orthogonalize  # noqa: E402

TINY = dict(vocab_size=64, n_layer=2, n_head=4, n_kv_head=2, d_model=32, d_ff=48, max_seq_len=16)


@pytest.mark.parametrize("extra", [{}, {"qk_norm": True}, {"arch": "gpt2", "n_kv_head": 4}])
def test_trace_reproduces_the_model(extra):
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**TINY, **extra})).eval()
    idx = torch.randint(0, 64, (1, 9))
    t = flow.trace(model, idx)
    torch.testing.assert_close(t["logits"], model(idx)[0], atol=1e-5, rtol=1e-5)
    probs = t["layers"][0]["probs"]
    assert probs.shape == (4, 9, 9)
    torch.testing.assert_close(probs.sum(-1), torch.ones(4, 9))
    assert probs.triu(1).abs().max() == 0  # causal


def test_describe_runs():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**TINY)).eval()
    ids = torch.randint(0, 64, (1, 9))
    out = flow.describe(model, ids[:, :-1], ids[:, 1:])
    assert len(out["layers"]) == 2 and len(out["nats"]) == 8
    assert all(0 <= r < 64 for r in out["final"]["rank"])
    assert out["grads"]["total"] > 0 and len(out["grads"]["blocks"]) == 2


def test_heat_saturates_above_the_99th_percentile():
    x = torch.linspace(-1, 1, 200).reshape(2, 100)
    x[0, 0] = 50.0
    h = flow.heat(x)
    codes = np.frombuffer(base64.b64decode(h["q"]), dtype=np.int8).reshape(2, 100)
    assert codes[0, 0] == 127 and abs(codes).max() == 127
    assert h["s"] < 2  # the outlier does not set the colour scale


def test_ns_iterates_end_at_orthogonalize():
    torch.manual_seed(0)
    for shape in [(8, 20), (20, 8)]:
        g = torch.randn(shape)
        xs = flow.ns_iterates(g)
        assert len(xs) == 6
        last = xs[-1].T if shape[0] > shape[1] else xs[-1]
        torch.testing.assert_close(last, orthogonalize(g).float())
