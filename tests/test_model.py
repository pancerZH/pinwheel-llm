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


LOOPED = GPTConfig(**{**TINY.__dict__, "n_layer": 4, "n_loop": 3})  # block 0, blocks 1-2 three times, block 3


def test_loop_reruns_the_middle_blocks_with_one_embedding_per_pass():
    model = GPT(LOOPED)
    assert [b for b, _ in model.order] == [0, 1, 2, 1, 2, 1, 2, 3]
    assert [loop for _, loop in model.order] == [None, 0, None, 1, None, 2, None, None]
    plain = GPT(GPTConfig(**{**TINY.__dict__, "n_layer": 4}))
    d = TINY.d_model
    assert model.num_params() == plain.num_params() + 3 * d  # shared weights, plus the loop embedding
    block = sum(p.numel() for p in plain.blocks[0].parameters())
    # 4 more block passes, each with its matmuls and attention scores
    assert model.flops_per_token(64) - plain.flops_per_token(64) == 6 * (4 * block + 3 * d) + 12 * 4 * d * 64


def test_looped_kv_cache_matches_full_forward():
    torch.manual_seed(0)
    model = GPT(LOOPED).eval()
    with torch.no_grad():
        model.loop_emb.normal_()  # 0 at init, which would hide a mix-up between passes
    x = torch.randint(0, TINY.vocab_size, (2, 16))
    caches = [KVCache(16) for _ in model.order]
    steps = [model(x[:, :8], caches=caches)]
    steps += [model(x[:, t : t + 1], caches=caches) for t in range(8, 16)]
    torch.testing.assert_close(torch.cat(steps, dim=1), model(x), atol=1e-5, rtol=1e-4)


MTP2 = GPTConfig(**{**TINY.__dict__, "mtp": 2})


def test_mtp_adds_one_loss_per_module_and_leaves_the_next_token_loss_alone():
    torch.manual_seed(0)
    model = GPT(MTP2)
    x, y = torch.randint(0, TINY.vocab_size, (2, 2, 16))
    losses = model(x, y, mtp=True)
    assert len(losses) == 3
    torch.testing.assert_close(losses[0], model(x, y))
    assert [loss.numel() for loss in model(x, y, mtp=True, reduction="none")] == [32, 30, 28]


def test_mtp_module_k_reads_tokens_up_to_i_plus_k_only():
    torch.manual_seed(0)
    model = GPT(MTP2).eval()
    seq = torch.randint(0, TINY.vocab_size, (1, 17))
    p = 10  # change token 10
    changed = seq.clone()
    changed[0, p] = (seq[0, p] + 1) % TINY.vocab_size
    before = model(seq[:, :-1], seq[:, 1:], mtp=True, reduction="none")
    after = model(changed[:, :-1], changed[:, 1:], mtp=True, reduction="none")
    for k, (a, b) in enumerate(zip(before, after)):  # k = 0 is the next-token head
        i = p - k - 1  # the first position whose target is token 10
        torch.testing.assert_close(a[:i], b[:i])
        assert not torch.allclose(a[i], b[i])


def test_mtp_residual_stream_stays_in_float32_under_autocast():
    model = GPT(GPTConfig(**{**TINY.__dict__, "mtp": 1}))
    seen = []
    model.mtp[0].block.attn_norm.register_forward_hook(lambda m, args, out: seen.append(args[0].dtype))
    x, y = torch.randint(0, TINY.vocab_size, (2, 2, 16))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        model(x, y, mtp=True)
    assert seen == [torch.float32]  # as in the trunk, where the stream starts as the fp32 embedding
