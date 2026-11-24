import math

import torch

from pinwheel.model import GPT, GPTConfig, KVCache

TINY = GPTConfig(vocab_size=512, n_layer=2, n_head=4, n_kv_head=2, d_model=64, d_ff=128, max_seq_len=64)


def test_v0_has_about_50m_params():
    with torch.device("meta"):
        model = GPT(GPTConfig())
    assert 50e6 < model.num_params() < 51.5e6


def test_initial_loss_is_about_ln_vocab():
    torch.manual_seed(0)
    x, y = torch.randint(0, TINY.vocab_size, (2, 4, 32))
    # Not y = x: with tied embeddings, the input token itself already scores higher at init.
    loss = GPT(TINY)(x, y)
    assert abs(loss.item() - math.log(TINY.vocab_size)) < 0.1


def test_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = GPT(TINY).eval()
    x = torch.randint(0, TINY.vocab_size, (2, 16))
    caches = [KVCache(16) for _ in model.blocks]
    steps = [model(x[:, :8], caches=caches)]  # prompt, then one token at a time
    steps += [model(x[:, t : t + 1], caches=caches) for t in range(8, 16)]
    torch.testing.assert_close(torch.cat(steps, dim=1), model(x), atol=1e-5, rtol=1e-4)
