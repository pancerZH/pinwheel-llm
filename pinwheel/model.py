"""A small Llama-style decoder: RMSNorm, RoPE, GQA, SwiGLU, tied embeddings."""

import math
from dataclasses import dataclass

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

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
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


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.attn = Attention(cfg)
        self.mlp_norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin, cache=None):
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.mlp(self.mlp_norm(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.norm = nn.RMSNorm(cfg.d_model, eps=1e-6)
        cos, sin = rope_tables(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, std=0.02)
        # Projections that write into the residual stream start smaller, so its variance
        # does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "w_down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def forward(self, idx, targets=None, caches=None, reduction="mean"):
        """Logits (B, T, V), or the cross-entropy loss when targets are given."""
        T = idx.size(1)
        pos = caches[0].pos if caches else 0
        cos, sin = self.cos[pos : pos + T], self.sin[pos : pos + T]
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, caches[i] if caches else None)
        logits = F.linear(self.norm(x), self.tok_emb.weight)  # tied: the output head is the embedding
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.float().view(-1, logits.size(-1)), targets.reshape(-1), reduction=reduction
        )

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def flops_per_token(self, seq_len):
        """Training FLOPs per token: 6 per parameter (forward + backward) plus the attention scores."""
        return 6 * self.num_params() + 12 * self.cfg.n_layer * self.cfg.d_model * seq_len

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None, generator=None):
        """Yield sampled token ids, a (B,) tensor per step, using a KV cache."""
        max_new_tokens = min(max_new_tokens, self.cfg.max_seq_len - idx.size(1))
        caches = [KVCache(idx.size(1) + max_new_tokens) for _ in self.blocks]
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
