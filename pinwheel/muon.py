"""Muon (Keller Jordan 2024) for the 2-D weight matrices inside the blocks.

Each step takes the momentum of the gradient, G = U S V^T, and moves along U V^T instead: every
singular direction gets the same step size, so a few large directions no longer swamp the rest.
U V^T comes from a few Newton-Schulz iterations rather than an SVD. Embeddings and norm gains stay
on AdamW.
"""

import math

import torch


def orthogonalize(g, steps=5, eps=1e-7):
    """Push the singular values of g towards 1, keeping U and V.

    Each iteration maps a singular value s to a*s + b*s^3 + c*s^5. The coefficients are tuned to
    lift small values fast (about 3.4x per iteration), not to converge exactly: values that start
    above roughly 1/500 of the Frobenius norm end in 0.7-1.2, smaller ones stay small.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.size(0) > x.size(1)
    if tall:  # work on the wide side, so x @ x.T is the smaller Gram matrix
        x = x.T
    x = x / (x.norm() + eps)  # Frobenius norm >= largest singular value, so all s start <= 1
    for _ in range(steps):
        gram = x @ x.T
        poly = torch.addmm(gram, gram, gram, beta=b, alpha=c)  # b * gram + c * gram @ gram
        x = torch.addmm(x, poly, x, beta=a)  # a * x + poly @ x
    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    """SGD with Nesterov momentum whose update is orthogonalized, plus decoupled weight decay.

    The update is scaled by 0.2 * sqrt(max(rows, cols)) so its RMS matches a typical AdamW update
    (Liu et al. 2025, "Muon is Scalable for LLM Training"); lr and weight_decay are then on AdamW's
    scale. Same rule as torch.optim.Muon with adjust_lr_fn="match_rms_adamw".
    """

    def __init__(self, params, lr, weight_decay=0.0, momentum=0.95):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum))
        for group in self.param_groups:
            assert all(p.dim() == 2 for p in group["params"]), "Muon only takes 2-D weights"

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, wd, mu = group["lr"], group["weight_decay"], group["momentum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p)
                buf = state["momentum"]
                buf.lerp_(p.grad, 1 - mu)
                update = orthogonalize(p.grad.lerp(buf, mu))  # Nesterov: look one step ahead
                p.mul_(1 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr * 0.2 * math.sqrt(max(p.shape)))
