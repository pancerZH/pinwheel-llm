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


def test_gpt2_arch_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    cfg = GPTConfig(**{**TINY.__dict__, "arch": "gpt2", "n_kv_head": 4})
    model = GPT(cfg).eval()
    assert model.pos_emb is not None and isinstance(model.norm, torch.nn.LayerNorm)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    caches = [KVCache(16) for _ in model.blocks]
    steps = [model(x[:, :8], caches=caches)]
    steps += [model(x[:, t : t + 1], caches=caches) for t in range(8, 16)]
    torch.testing.assert_close(torch.cat(steps, dim=1), model(x), atol=1e-5, rtol=1e-4)


def test_qk_norm_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**TINY.__dict__, "qk_norm": True})).eval()
    x = torch.randint(0, TINY.vocab_size, (2, 16))
    caches = [KVCache(16) for _ in model.blocks]
    steps = [model(x[:, :8], caches=caches)]
    steps += [model(x[:, t : t + 1], caches=caches) for t in range(8, 16)]
    torch.testing.assert_close(torch.cat(steps, dim=1), model(x), atol=1e-5, rtol=1e-4)


def test_qk_norm_makes_attention_ignore_the_size_of_wq_and_wk():
    torch.manual_seed(0)
    x = torch.randint(0, TINY.vocab_size, (2, 16))
    for qk_norm in (False, True):
        model = GPT(GPTConfig(**{**TINY.__dict__, "qk_norm": qk_norm})).eval()
        before = model(x)
        with torch.no_grad():
            for block in model.blocks:
                block.attn.wq.weight *= 3
                block.attn.wk.weight *= 3
        same = torch.allclose(model(x), before, atol=1e-4)
        assert same == qk_norm  # only QK-Norm makes the scores independent of the weights' scale


def test_zero_init_makes_every_block_the_identity():
    torch.manual_seed(0)
    model = GPT(GPTConfig(**{**TINY.__dict__, "zero_init": True}))
    x = torch.randn(2, 16, TINY.d_model)
    cos, sin = model.cos[:16], model.sin[:16]
    for block in model.blocks:
        torch.testing.assert_close(block(x, cos, sin), x)


def test_emb_std_sets_the_embedding_scale():
    model = GPT(GPTConfig(**{**TINY.__dict__, "emb_std": 0.002}))
    assert abs(model.tok_emb.weight.std().item() - 0.002) < 0.0002
