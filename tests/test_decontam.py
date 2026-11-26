import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from decontam import overlap_mask  # noqa: E402


def test_overlap_mask_marks_exactly_the_copied_span():
    rng = np.random.default_rng(0)
    val = rng.integers(0, 8192, 1000, dtype=np.uint16)
    train = rng.integers(0, 8192, 5000, dtype=np.uint16)
    train[3000:3100] = val[200:300]  # a 100-token copy, split across two scan chunks below
    train[4000:4063] = val[500:563]  # 63 tokens: one short of a match
    mask = overlap_mask(val, train, chunk=3050)
    assert mask[200:300].all()
    assert mask.sum() == 100
