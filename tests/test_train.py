import torch

from pinwheel.train import tail_averages, update_averages


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
