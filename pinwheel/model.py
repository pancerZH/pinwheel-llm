"""A small Llama-style decoder: RMSNorm, RoPE, GQA, SwiGLU, tied embeddings.

qk_norm=True adds an RMSNorm on every head's queries and keys (QK-Norm), so attention scores depend
only on the angle between q and k and on the norms' gains, not on how large wq and wk grow.

arch="gpt2" swaps three of them back to GPT-2: LayerNorm, learned absolute positions and a GELU MLP
(pair it with n_kv_head = n_head for plain multi-head attention).

n_loop > 1 runs the middle layers n_loop times with the same weights (looped / recurrent depth): the
first n_prelude and the last n_coda layers run once, and a learned loop embedding, one vector per pass,
is added to the residual stream at the start of each pass.

mtp > 0 adds DeepSeek-V3's multi-token prediction modules, used only in training: module k mixes the
hidden state at position i with the embedding of token i + k, runs one more block and predicts token
i + k + 1 through the shared output head. forward(..., mtp=True) returns their losses too.
"""

import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    n_layer: int = 10
    n_head: int = 10
    n_kv_head: int = 5
    d_model: int = 640
    d_ff: int = 1728
    max_seq_len: int = 1024
    rope_base: float = 10000.0
    arch: str = "llama"  # "llama" or "gpt2" (see the module docstring)
    qk_norm: bool = False
    zero_init: bool = False  # wo and w_down start at 0, so every block starts as the identity
    emb_std: float = 0.02  # init std of the tied embedding
    n_loop: int = 1  # passes through the middle layers (see the module docstring)
    n_prelude: int = 1  # with n_loop > 1: layers before the loop, run once
    n_coda: int = 1  # with n_loop > 1: layers after the loop, run once
    mtp: int = 0  # multi-token prediction modules (see the module docstring)


def rope_tables(head_dim, max_len, base):
    inv_freq = 1.0 / base ** (torch.arange(0, head_dim, 2).float() / head_dim)
    angles = torch.outer(torch.arange(max_len).float(), inv_freq)  # (max_len, head_dim / 2)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each pair of channels (x1, x2) by an angle that depends on the position."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).to(x.dtype)


class KVCache:
    """Keys and values of one layer, filled in as tokens are generated."""

    def __init__(self, max_len):
        self.max_len, self.pos, self.k, self.v = max_len, 0, None, None

    def update(self, k, v):
        B, H, T, D = k.shape
        assert T == 1 or self.pos == 0, "a prompt can only go into an empty cache"
        if self.k is None:
            self.k = k.new_zeros(B, H, self.max_len, D)
            self.v = v.new_zeros(B, H, self.max_len, D)
        self.k[:, :, self.pos : self.pos + T] = k
        self.v[:, :, self.pos : self.pos + T] = v
        self.pos += T
        return self.k[:, :, : self.pos], self.v[:, :, : self.pos]


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head, self.n_kv_head = cfg.n_head, cfg.n_kv_head
        self.head_dim = cfg.d_model // cfg.n_head
        self.wq = nn.Linear(cfg.d_model, cfg.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_head * self.head_dim, cfg.d_model, bias=False)
        # One 64-dim gain for queries and one for keys, shared by the heads of a layer.
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if cfg.qk_norm else None
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6) if cfg.qk_norm else None

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        if self.q_norm is not None:  # norm 8 (= sqrt(head_dim)) times the gain, so |score| <= 8 * gain^2
            q, k = self.q_norm(q), self.k_norm(k)
        if cos is not None:  # RoPE; gpt2 adds learned positions to the input instead
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(k, v)
        # A prompt needs the causal mask; a single new token may see everything in the cache.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=T > 1, enable_gqa=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, -1))


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class GeluMLP(nn.Module):
    """GPT-2's MLP: up-projection, GELU, down-projection."""

    def __init__(self, cfg):
        super().__init__()
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.gelu(self.w_up(x), approximate="tanh"))


def make_norm(cfg):
    return nn.LayerNorm(cfg.d_model) if cfg.arch == "gpt2" else nn.RMSNorm(cfg.d_model, eps=1e-6)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = make_norm(cfg)
        self.attn = Attention(cfg)
        self.mlp_norm = make_norm(cfg)
        self.mlp = GeluMLP(cfg) if cfg.arch == "gpt2" else SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.mlp(self.mlp_norm(x))


class MTP(nn.Module):
    """One multi-token prediction module (DeepSeek-V3); its output goes through the shared head."""

    def __init__(self, cfg):
        super().__init__()
        self.h_norm, self.e_norm = make_norm(cfg), make_norm(cfg)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.block = Block(cfg)
        self.norm = make_norm(cfg)

    def forward(self, h, e, cos, sin):
        """h: hidden states at positions i, e: embeddings of tokens i + k; returns new hidden states."""
        x = self.proj(torch.cat([self.h_norm(h), self.e_norm(e)], dim=-1))
        # Under autocast proj returns bf16; keep this residual stream in h's dtype (fp32), like the trunk's.
        return self.block(x.to(h.dtype), cos, sin)


def layer_order(cfg):
    """(block index, loop pass starting there or None) for every layer, in the order they run."""
    if cfg.n_loop == 1:
        return [(b, None) for b in range(cfg.n_layer)]
    first, end = cfg.n_prelude, cfg.n_layer - cfg.n_coda
    if not 0 <= first < end <= cfg.n_layer:
        raise ValueError("the loop needs at least one layer between the prelude and the coda")
    return ([(b, None) for b in range(first)]
            + [(b, r if b == first else None) for r in range(cfg.n_loop) for b in range(first, end)]
            + [(b, None) for b in range(end, cfg.n_layer)])


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.arch not in ("llama", "gpt2"):
            raise ValueError(f"unknown arch: {cfg.arch}")
        self.cfg = cfg
        self.order = layer_order(cfg)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model) if cfg.arch == "gpt2" else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = make_norm(cfg)
        self.loop_emb = nn.Parameter(torch.zeros(cfg.n_loop, cfg.d_model)) if cfg.n_loop > 1 else None
        self.mtp = nn.ModuleList(MTP(cfg) for _ in range(cfg.mtp))  # created last: v0's init draws stay the same
        cos, sin = rope_tables(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        nn.init.normal_(self.tok_emb.weight, std=cfg.emb_std)
        # Projections that write into the residual stream start smaller, so its variance
        # does not grow with depth; with zero_init they start at 0.
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w_down.weight")):
                if cfg.zero_init:
                    nn.init.zeros_(p)
                else:
                    nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, idx, targets=None, caches=None, reduction="mean", mtp=False, n_loop=None):
        """Logits (B, T, V), or the cross-entropy loss when targets are given.

        With mtp=True, a list: the next-token loss, then the loss of each MTP module (module k scores
        the T - k positions whose token i + k + 1 is still among the targets). n_loop runs fewer passes
        through the loop than cfg.n_loop (training with a random number of passes).
        """
        T = idx.size(1)
        pos = caches[0].pos if caches else 0
        x = self.tok_emb(idx)
        if self.pos_emb is None:
            cos, sin = self.cos[pos : pos + T], self.sin[pos : pos + T]
        else:
            cos = sin = None
            x = x + self.pos_emb(torch.arange(pos, pos + T, device=idx.device))
        order = self.order if n_loop is None else self.loop_order(n_loop)
        for i, (b, loop) in enumerate(order):
            if loop is not None:
                x = x + self.loop_emb[loop]
            x = self.blocks[b](x, cos, sin, caches[i] if caches else None)
        logits = F.linear(self.norm(x), self.tok_emb.weight)  # tied: the output head is the embedding
        if targets is None:
            return logits
        losses = [self._loss(logits, targets, reduction)]
        for k, module in enumerate(self.mtp if mtp else [], start=1):
            n = T - k
            x = module(x[:, :n], self.tok_emb(targets[:, k - 1 : k - 1 + n]),
                       None if cos is None else cos[:n], None if sin is None else sin[:n])
            losses.append(self._loss(F.linear(module.norm(x), self.tok_emb.weight), targets[:, k:], reduction))
        return losses if mtp else losses[0]

    @staticmethod
    def _loss(logits, targets, reduction):
        return F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction)

    def loop_order(self, n_loop):
        if not 1 <= n_loop <= self.cfg.n_loop:
            raise ValueError(f"n_loop must be between 1 and {self.cfg.n_loop}")
        return layer_order(replace(self.cfg, n_loop=n_loop))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self, seq_len, n_loop=None):
        """Training FLOPs per token: 6 per parameter (forward + backward) plus the attention scores.

        A looped block counts once per pass, and every MTP module uses the output head once more.
        """
        cfg = self.cfg
        order = self.order if n_loop is None else self.loop_order(n_loop)
        block_params = [sum(p.numel() for p in block.parameters()) for block in self.blocks]
        reused = sum(block_params[b] for b, _ in order) - sum(block_params)
        matmul_params = self.num_params() + reused + cfg.mtp * cfg.vocab_size * cfg.d_model
        return 6 * matmul_params + 12 * (len(order) + cfg.mtp) * cfg.d_model * seq_len

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, generator=None):
        """Yield sampled token ids, a (B,) tensor per step, using a KV cache."""
        max_new_tokens = min(max_new_tokens, self.cfg.max_seq_len - idx.size(1))
        caches = [KVCache(idx.size(1) + max_new_tokens) for _ in self.order]  # one per layer run
        logits = self(idx, caches=caches)[:, -1]
        for i in range(max_new_tokens):
            next_id = sample_next(logits, temperature, top_k, top_p, generator)
            yield next_id
            if i + 1 < max_new_tokens:
                logits = self(next_id[:, None], caches=caches)[:, -1]


def sample_next(logits, temperature=1.0, top_k=None, top_p=None, generator=None):
    if temperature == 0:
        return logits.argmax(-1)
    logits = logits.float() / temperature
    if top_k:
        kth = logits.topk(min(top_k, logits.size(-1))).values[:, -1:]
        logits = logits.masked_fill(logits < kth, -float("inf"))
    if top_p is not None and top_p < 1:
        sorted_logits, order = logits.sort(dim=-1, descending=True)
        probs = sorted_logits.softmax(-1)
        # Keep the smallest set of tokens whose probabilities add up to top_p.
        drop = probs.cumsum(-1) - probs > top_p
        logits = logits.scatter(-1, order, sorted_logits.masked_fill(drop, -float("inf")))
    return torch.multinomial(logits.softmax(-1), 1, generator=generator).squeeze(-1)
