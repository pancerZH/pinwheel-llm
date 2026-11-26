import torch

from pinwheel.model import GPT, GPTConfig
from pinwheel.muon import Muon, orthogonalize
from pinwheel.train import TrainConfig, make_optimizers


def test_orthogonalize_keeps_shape_and_flattens_singular_values():
    torch.manual_seed(0)
    for shape in [(64, 128), (128, 64)]:
        g = torch.randn(shape) * torch.logspace(0, -1, shape[1])  # columns span 10x in scale
        out = orthogonalize(g)
        assert out.shape == g.shape
        s = torch.linalg.svdvals(out.float())
        assert 0.6 < s.min() and s.max() < 1.25


def test_muon_matches_torch_muon():
    # torch keeps momentum as a sum and ours as an average; they differ only by a factor 1 - momentum,
    # which the normalization inside orthogonalize removes. With momentum 0.75 that factor is a power
    # of 2, so bf16 rounds both the same way and the results agree closely.
    torch.manual_seed(0)
    init = [torch.randn(64, 128), torch.randn(128, 64)]
    grads = [[torch.randn_like(w) for w in init] for _ in range(3)]
    ours = [torch.nn.Parameter(w.clone()) for w in init]
    ref = [torch.nn.Parameter(w.clone()) for w in init]
    opts = [Muon(ours, lr=0.01, weight_decay=0.1, momentum=0.75),
            torch.optim.Muon(ref, lr=0.01, weight_decay=0.1, momentum=0.75, adjust_lr_fn="match_rms_adamw")]
    for step_grads in grads:
        for params, opt in zip([ours, ref], opts):
            for p, g in zip(params, step_grads):
                p.grad = g.clone()
            opt.step()
    for w, a, b in zip(init, ours, ref):
        moved_ours, moved_ref = (a - w).detach(), (b - w).detach()
        assert (moved_ours - moved_ref).norm() < 1e-3 * moved_ref.norm()


def test_make_optimizers_puts_block_matrices_on_muon():
    cfg = TrainConfig(optimizer="muon", lr=2e-3, muon_lr=4e-3)
    model = GPT(GPTConfig(vocab_size=512, n_layer=2, n_head=4, n_kv_head=2, d_model=64, d_ff=128))
    adamw, muon = make_optimizers(model, cfg)
    names = {id(p): n for n, p in model.named_parameters()}
    on_muon = {names[id(p)] for g in muon.param_groups for p in g["params"]}
    on_adamw = {names[id(p)] for g in adamw.param_groups for p in g["params"]}
    assert len(on_muon) == 2 * 7  # wq wk wv wo w_gate w_up w_down per block
    assert all(n.startswith("blocks.") and n.endswith("weight") for n in on_muon)
    assert on_adamw == set(names.values()) - on_muon  # embedding and norm gains, each exactly once
    assert [g["base_lr"] for g in muon.param_groups] == [4e-3]
    assert [g["base_lr"] for g in adamw.param_groups] == [2e-3, 2e-3]

    (only_adamw,) = make_optimizers(model, TrainConfig())
    assert sum(len(g["params"]) for g in only_adamw.param_groups) == len(names)
