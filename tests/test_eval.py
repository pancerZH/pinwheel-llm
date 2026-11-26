import math

import numpy as np
import torch

from pinwheel.data import TokenData
from pinwheel.eval import evaluate_bpb


def test_val_bpb_clean_skips_windows_with_overlap(tmp_path):
    B, T = 2, 8
    val = (np.arange(4 * T + 1) % 50 + 1).astype(np.uint16)
    (tmp_path / "val.bin").write_bytes(val.tobytes())
    (tmp_path / "meta.json").write_text("{}")
    overlap = np.zeros(len(val), bool)
    overlap[T + 3] = True  # inside window 1 only (window i covers val[i*T : i*T + T + 1])
    np.save(tmp_path / "val_overlap.npy", overlap)
    data = TokenData(tmp_path)
    assert [c.tolist() for _, _, c in data.val_batches(B, T, 4 * T, "cpu")] == [[True, False], [True, True]]

    def model(x, y, reduction):  # loss = target id / 10 at every position
        return (y.float() / 10).reshape(-1)

    token_bytes = torch.ones(64, dtype=torch.int64)  # one byte per token
    bpb, clean = evaluate_bpb(model, data, token_bytes, B, T, 4 * T)
    loss = torch.from_numpy(val[1:].astype(np.int64)).view(4, T).float() / 10
    assert math.isclose(bpb, loss.mean().item() / math.log(2), rel_tol=1e-6)
    assert math.isclose(clean, loss[[0, 2, 3]].mean().item() / math.log(2), rel_tol=1e-6)

    (tmp_path / "val_overlap.npy").unlink()
    assert evaluate_bpb(model, TokenData(tmp_path), token_bytes, B, T, 4 * T)[1] is None
