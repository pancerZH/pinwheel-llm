"""Build a self-contained webpage that walks through one training step of pinwheel-llm (runs/flow.html).

    uv run python scripts/flow.py [--text "..."] [--serve --host 0.0.0.0]
    uv run python scripts/flow.py --serve-only --host 0.0.0.0   # serve the page already built

Every run in RUNS that has a checkpoint reads one short text on the CPU, and the page keeps what
each stage produces: tokens, embeddings, the attention and MLP of every layer that runs (a looped
block once per pass), the per-layer readout (logit lens), the logits, the MTP modules' guesses,
the loss and the gradients. The AdamW and Muon updates come from the
optimizer state of a stable checkpoint plus one real training batch. Configs, val BPB, bench
scores and curves come from runs/. No number is typed in by hand.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pinwheel.model import GPT, GPTConfig, SwiGLU, apply_rope
from pinwheel.muon import orthogonalize
from pinwheel.tokenizer import BOS, load_tokenizer, token_byte_lengths
from pinwheel.train import TrainConfig, make_optimizers
from tree import _serve_single_file  # scripts/ is on sys.path when this file runs as a script

# The page starts on the first run that exists; "-stable" names a stable checkpoint of a run.
RUNS = (
    "2e-stack-1b", "2e-muon-1b", "3a-mtp1-1b-5080",
    "3d-cont-v0-200m", "3d-upcycle-loop2-200m", "3d-cont-v0-347m-5080",
    "2e-zeroinit-emb002-300m-5080", "2e-emb002-300m", "3a-mtp1-300m-5080",
    "3b-l10-loop2-300m", "3b-l6-300m", "3b-l6-loop2-300m", "3b-l6-randloop-300m", "2b-wsd-lr2e-3", "2b-wsd-1b", "2b-wsd-1b-stable", "2b-wsd-lr2.8e-3",
    "2a-bs128k-lr1.4e-3", "2a-bs256k-lr2e-3", "1e-v0-lr2e-3",
    "2c-gpt2-vanilla", "2c-gpt2-vanilla-old",
    "2e-adamw-300m", "2e-muon-lr1e-3-300m", "2e-muon-lr2e-3-300m", "2e-muon-lr4e-3-300m",
    "2e-qknorm-muon-lr2e-3-300m", "2e-qknorm-muon-lr4e-3-300m-5080", "2e-zeroinit-300m",
    "2d-wsd-2b-cd", "2d-wsd-4b", "1a-v0-tinystories",
)
SPECTRA_FROM = "2b-wsd-1b-stable"  # an AdamW stable checkpoint: its moments drive the update spectra
TEXT = ("Photosynthesis is the process by which plants use sunlight to make food. "
        "In photosynthesis, plants use")
COLS = 128  # channels kept per activation heatmap
PLACEHOLDER = "__PINWHEEL_FLOW_DATA__"


def sig(v, n=4):
    """Round to n significant digits (lists too), to keep the JSON small."""
    if isinstance(v, (list, tuple)):
        return [sig(x, n) for x in v]
    v = float(v)
    return float(f"{v:.{n}g}") if math.isfinite(v) else None


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def heat(x: torch.Tensor) -> dict:
    """A (T, C) activation as int8 codes over its first COLS channels; 127 is the 99th percentile
    of |x|, larger values saturate."""
    x = x[:, :COLS].float()
    s = float(x.abs().flatten().quantile(0.99)) or 1.0
    codes = (x / s * 127).round().clamp(-127, 127).to(torch.int8).numpy()
    return {"s": sig(s), "w": x.shape[1], "q": b64(codes)}


def rms(x: torch.Tensor) -> list:
    return sig(x.float().pow(2).mean(-1).sqrt().tolist())


def recipe(train: dict, model: dict, tokens: float) -> dict:
    """The swappable parts of a run, in the page's terms (old configs predate some fields and ran
    at their defaults)."""
    opt = train.get("optimizer") or "adamw"
    loop = model.get("n_loop") or 1
    return {
        "arch": model.get("arch") or "llama",
        "n_layer": model["n_layer"],
        "loop": f"{train['loop_min']}-{loop}" if train.get("loop_min") else str(loop),  # "1-4": random passes
        "mtp": bool(model.get("mtp")),
        "qk_norm": bool(model.get("qk_norm")),
        "zero_init": bool(model.get("zero_init")),
        "emb_std": model.get("emb_std") or 0.02,
        "optimizer": opt,
        "lr": train["muon_lr"] if opt == "muon" else train["lr"],
        "schedule": train.get("schedule") or "cosine",
        "batch": train["batch_tokens"],
        "tokens": round(tokens),
        "start": Path(train.get("init_from") or "").stem,  # the run whose weights it continued
        "data": Path(train["data_dir"]).name.removesuffix("-4b"),  # the 4B set shares the 1B set's val
    }


def start_tokens(train: dict, runs_dir: Path) -> float:
    """Tokens already seen when a run starts: a --cooldown_from branch starts at its parent's stable
    checkpoint, an --init_from run after all of its parent's tokens, a run from scratch at 0."""
    if train.get("init_from"):
        parent = json.loads((runs_dir / Path(train["init_from"]).stem / "config.json").read_text())["train"]
        return start_tokens(parent, runs_dir) + parent["max_tokens"]
    src = train.get("cooldown_from") or ""
    if not src:
        return 0.0
    parent = json.loads((runs_dir / Path(src).stem.removesuffix("-stable") / "config.json").read_text())["train"]
    return math.ceil(parent["save_stable_at"] / parent["batch_tokens"]) * parent["batch_tokens"]


def read_tsv(path: Path, key: str) -> dict:
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        return {row[key]: row for row in csv.DictReader(f, delimiter="\t")}


def read_log(run_dir: Path, offset=0.0, max_points=240) -> dict:
    train, val = [], []
    for line in (run_dir / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "loss" in r:
            train.append([r["tokens"] + offset, r["loss"], r["lr"], r["grad_norm"], r["tok_per_s"]])
        elif "val_bpb" in r:
            val.append([r["tokens"] + offset, r["val_bpb"]])
    tok_s = statistics.median(t[4] for t in train[len(train) // 10:]) if train else None
    train = train[:: max(1, len(train) // max_points)]
    return {"train": [sig(t[:4]) for t in train], "val": [sig(v, 5) for v in val], "tok_s": tok_s}


@torch.no_grad()
def trace(model: GPT, idx: torch.Tensor) -> dict:
    """GPT.forward for a batch of one, step by step, keeping every intermediate tensor.

    Every layer that runs gets its own entry: a looped block once per pass, with the block whose
    weights it uses and the pass it belongs to (None outside the loop)."""
    T = idx.size(1)
    x = model.tok_emb(idx)
    out = {"emb": x[0], "pos": None, "layers": []}
    if model.pos_emb is None:
        cos, sin = model.cos[:T], model.sin[:T]
    else:
        cos = sin = None
        out["pos"] = model.pos_emb.weight[:T]
        x = x + out["pos"]
    out["x0"] = x[0]
    causal = torch.ones(T, T, dtype=torch.bool).tril()
    cfg, n_pass = model.cfg, None
    looped = range(cfg.n_prelude, cfg.n_layer - cfg.n_coda) if cfg.n_loop > 1 else range(0)
    for b, loop in model.order:
        if loop is not None:  # a pass through the loop starts: add that pass's loop embedding
            n_pass = loop
            x = x + model.loop_emb[loop]
        inp, blk = x, model.blocks[b]
        a = blk.attn
        h = blk.attn_norm(x)
        q = a.wq(h).view(1, T, a.n_head, a.head_dim).transpose(1, 2)
        k = a.wk(h).view(1, T, a.n_kv_head, a.head_dim).transpose(1, 2)
        v = a.wv(h).view(1, T, a.n_kv_head, a.head_dim).transpose(1, 2)
        q_raw, k_raw = q, k
        if a.q_norm is not None:
            q, k = a.q_norm(q), a.k_norm(k)
        if cos is not None:
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        group = a.n_head // a.n_kv_head  # q head i reads kv head i // group, as SDPA's enable_gqa does
        k, v = k.repeat_interleave(group, 1), v.repeat_interleave(group, 1)
        scores = (q @ k.transpose(-1, -2) / math.sqrt(a.head_dim)).masked_fill(~causal, float("-inf"))
        probs = scores.softmax(-1)
        attn_out = a.wo((probs @ v).transpose(1, 2).reshape(1, T, -1))
        mid = x + attn_out
        h2 = blk.mlp_norm(mid)
        if isinstance(blk.mlp, SwiGLU):
            hid = F.silu(blk.mlp.w_gate(h2)) * blk.mlp.w_up(h2)
        else:
            hid = F.gelu(blk.mlp.w_up(h2), approximate="tanh")
        mlp_out = blk.mlp.w_down(hid)
        x = mid + mlp_out
        out["layers"].append(dict(
            block=b, n_pass=n_pass if b in looped else None, inp=inp[0], attn_norm=h[0],
            q_raw=q_raw[0], k_raw=k_raw[0], scores=scores[0].masked_fill(~causal, 0), probs=probs[0],
            attn_out=attn_out[0], mid=mid[0], mlp_norm=h2[0], hid=hid[0], mlp_out=mlp_out[0], out=x[0]))
    out["x"], out["cos"], out["sin"] = x, cos, sin
    out["logits"] = F.linear(model.norm(x), model.tok_emb.weight)[0]
    return out


def mtp_trace(model: GPT, t: dict, targets: torch.Tensor) -> list:
    """The MTP modules' logits on the trunk's last residual stream, as GPT.forward(mtp=True) runs
    them: module k reads position i and the embedding of token i + k, and guesses token i + k + 1."""
    x, T, out = t["x"], targets.size(1), []
    for k, module in enumerate(model.mtp, start=1):
        n = T - k
        cut = [None if c is None else c[:n] for c in (t["cos"], t["sin"])]
        x = module(x[:, :n], model.tok_emb(targets[:, k - 1:k - 1 + n]), *cut)
        out.append(F.linear(module.norm(x), model.tok_emb.weight)[0].float())
    return out


@torch.no_grad()
def guesses(logits: torch.Tensor, targets: torch.Tensor, k=5) -> dict:
    """The top k guesses at every position, and the probability and rank of the right one."""
    p = logits.softmax(-1)
    top = p.topk(k, -1)
    right = logits.gather(-1, targets[:, None])
    return {"ids": top.indices.tolist(), "p": sig(top.values.tolist(), 3),
            "tp": sig(p.gather(-1, targets[:, None])[:, 0].tolist(), 3),
            "rank": (logits > right).sum(-1).tolist()}


def readout(model: GPT, x: torch.Tensor, targets: torch.Tensor, k=5) -> dict:
    """Read the residual stream as a prediction: final norm, then the tied output head."""
    return guesses(F.linear(model.norm(x), model.tok_emb.weight).float(), targets, k)


def describe(model: GPT, idx: torch.Tensor, targets: torch.Tensor, mtp_weight=0.3) -> dict:
    """Everything the page shows about one model on the sample text."""
    t = trace(model, idx)
    torch.testing.assert_close(t["logits"], model(idx)[0], atol=1e-4, rtol=1e-4)  # the trace is the model
    y = targets[0]
    mtp = []
    if model.cfg.mtp:
        losses = model(idx, targets, mtp=True)
        for k, logits in enumerate(mtp_trace(model, t, targets), start=1):
            nats = F.cross_entropy(logits, y[k:], reduction="none")
            torch.testing.assert_close(nats.mean(), losses[k], atol=1e-4, rtol=1e-4)
            mtp.append({**guesses(logits, y[k:]), "nats": sig(nats.tolist())})
    layers = []
    for L in t["layers"]:
        hid = L["hid"].float()
        sd = float(hid.std()) or 1.0
        hist, _ = np.histogram(hid.numpy(), bins=41, range=(-4 * sd, 4 * sd))
        top = L["out"].abs().amax(0).topk(3)
        layers.append({
            "block": L["block"], "pass": L["n_pass"], "in_rms": rms(L["inp"]), "attn_norm": heat(L["attn_norm"]), "attn_norm_rms": rms(L["attn_norm"]),
            "q_rms": sig(L["q_raw"].pow(2).mean((-1, -2)).sqrt().tolist()),
            "k_rms": sig(L["k_raw"].pow(2).mean((-1, -2)).sqrt().tolist()),
            "score_max": sig(L["scores"].abs().amax((-1, -2)).tolist()),
            "probs": b64((L["probs"] * 255).round().to(torch.uint8).numpy()),
            "attn_out": heat(L["attn_out"]), "attn_out_rms": rms(L["attn_out"]),
            "mid_rms": rms(L["mid"]), "mlp_norm": heat(L["mlp_norm"]),
            "hid": heat(L["hid"]), "hid_rms": rms(L["hid"]), "hid_sd": sig(sd),
            "hid_hist": hist.tolist(), "hid_small": sig(float((hid.abs() < 0.1 * sd).float().mean()), 3),
            "mlp_out": heat(L["mlp_out"]), "mlp_out_rms": rms(L["mlp_out"]),
            "out": heat(L["out"]), "out_rms": rms(L["out"]),
            "outliers": [[int(c), sig(v, 3)] for v, c in zip(top.values.tolist(), top.indices.tolist())],
            "lens": readout(model, L["out"], y),
        })
    nats = F.cross_entropy(t["logits"].float(), y, reduction="none")
    probs = t["logits"].float().softmax(-1)

    model.zero_grad(set_to_none=True)
    if model.cfg.mtp:  # the training loss: next token plus mtp_weight times the mean MTP loss
        main, *extra = model(idx, targets, mtp=True)
        (main + mtp_weight * sum(extra) / len(extra)).backward()
    else:
        model(idx, targets).backward()
    grads = {n: p.grad.norm().item() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    suffixes = [n.removeprefix("blocks.0.") for n in grads if n.startswith("blocks.0.")]
    mtp_grad = [v for n, v in grads.items() if n.startswith("mtp.")]
    return {
        "emb": heat(t["emb"]), "emb_rms": rms(t["emb"]),
        "pos": heat(t["pos"]) if t["pos"] is not None else None, "x0_rms": rms(t["x0"]),
        "layers": layers,
        "loop_rms": rms(model.loop_emb) if model.loop_emb is not None else None,
        "mtp": mtp,
        "lens0": readout(model, t["x0"], y),
        "final": readout(model, t["layers"][-1]["out"], y, k=10),
        "nats": sig(nats.tolist()),
        "entropy": sig((-(probs * probs.clamp_min(1e-12).log2()).sum(-1)).tolist(), 3),
        "grads": {
            "names": suffixes,
            "blocks": [[sig(grads[f"blocks.{i}.{s}"], 3) for s in suffixes] for i in range(model.cfg.n_layer)],
            "other": {n: sig(v, 3) for n, v in grads.items() if not n.startswith(("blocks.", "mtp."))},
            "mtp": sig(math.sqrt(sum(v * v for v in mtp_grad)), 3) if mtp_grad else None,
            "total": sig(math.sqrt(sum(v * v for v in grads.values()))),
        },
    }


def param_info(model: GPT) -> dict:
    """Shapes of the non-block parameters and of block 0's, and how each optimizer splits them."""
    shapes = {n: list(p.shape) for n, p in model.named_parameters()
              if not n.startswith("blocks.") or n.startswith("blocks.0.")}
    split = {}
    for name in ("adamw", "muon"):
        opts = make_optimizers(model, TrainConfig(optimizer=name))
        split[name] = [{"opt": type(o).__name__, "wd": g["weight_decay"], "n": len(g["params"]),
                        "numel": sum(p.numel() for p in g["params"])} for o in opts for g in o.param_groups]
    return {"shapes": shapes, "total": model.num_params(), "split": split}


def ns_iterates(g: torch.Tensor, steps=5, eps=1e-7) -> list:
    """orthogonalize() with every iterate kept; the last one is checked against the real function."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.size(0) > x.size(1)
    if tall:
        x = x.T
    x = x / (x.norm() + eps)
    xs = [x]
    for _ in range(steps):
        gram = x @ x.T
        x = torch.addmm(x, torch.addmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
        xs.append(x)
    assert torch.equal(xs[-1].T if tall else xs[-1], orthogonalize(g))
    return [t.float() for t in xs]


def spectrum(t: torch.Tensor, idx: np.ndarray) -> list:
    return sig(torch.linalg.svdvals(t.float())[idx].tolist(), 3)


def update_spectra(ckpt_path: Path, data_dir: Path, n_seq=8, seq_len=512, seed=0) -> dict:
    """One real optimizer step, both ways, from a stable checkpoint's AdamW moments.

    Muon was never run from this checkpoint, so AdamW's first moment stands in for Muon's momentum
    (both are averages of the same gradients; beta 0.9 against 0.95).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    cfg = TrainConfig(**{**ckpt["train_config"], "optimizer": "adamw"})
    adamw = make_optimizers(model, cfg)[0]
    state = ckpt["optimizer"]
    adamw.load_state_dict(state[0] if isinstance(state, list) else state)

    data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    starts = np.random.default_rng(seed).integers(0, len(data) - seq_len - 1, n_seq)
    xy = torch.from_numpy(np.stack([data[s:s + seq_len + 1] for s in starts]).astype(np.int64))
    model(xy[:, :-1], xy[:, 1:]).backward()

    beta1, beta2 = adamw.param_groups[0]["betas"]
    eps, mu = adamw.param_groups[0]["eps"], 0.95
    mats = []
    for name, p in model.named_parameters():
        if not (name.startswith("blocks.") and p.dim() == 2):
            continue
        st, g = adamw.state[p], p.grad
        step = int(st["step"]) + 1
        m = st["exp_avg"].lerp(g, 1 - beta1)
        v = st["exp_avg_sq"] * beta2 + (1 - beta2) * g * g
        adam = (m / (1 - beta1 ** step)) / ((v / (1 - beta2 ** step)).sqrt() + eps)
        xs = ns_iterates(g.lerp(st["exp_avg"].lerp(g, 1 - mu), mu))  # Muon's Nesterov input
        muon = xs[-1] * 0.2 * math.sqrt(max(p.shape))
        n = min(p.shape)
        idx = np.unique(np.geomspace(1, n, 96).round().astype(int)) - 1
        kinds = {"g": g, "m": m, "adam": adam, "muon": muon}
        stats = {}
        for key, t in kinds.items():
            s = torch.linalg.svdvals(t.float())
            stats[key] = {"s1": sig(s[0], 3), "top1": sig(float(s[0] ** 2 / (s ** 2).sum()), 3),
                          "ratio": sig(float(s[0] / s.median()), 3), "rms": sig(float(t.pow(2).mean().sqrt()), 3)}
        layer, kind = name.split(".")[1], name.split(".")[-2]
        mats.append({"layer": int(layer), "kind": kind, "shape": list(p.shape), "idx": idx.tolist(),
                     "sv": {k: spectrum(t, idx) for k, t in kinds.items()},
                     "ns": [spectrum(t, idx) for t in xs], "stats": stats})
    return {"run": ckpt_path.stem, "tokens": ckpt["tokens"], "step": ckpt["step"], "lr": cfg.lr,
            "wd": cfg.weight_decay, "betas": [beta1, beta2], "mu": mu, "batch": [n_seq, seq_len], "mats": mats}


def data_window(data_dir: Path, n=24, seed=0) -> dict | None:
    if not (data_dir / "train.bin").exists():
        return None
    data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    s = int(np.random.default_rng(seed).integers(0, len(data) - n - 1))
    return {"ids": data[s:s + n + 1].astype(int).tolist(), "train_tokens": len(data)}


def build_payload(args) -> dict:
    torch.set_num_threads(args.threads)
    runs_dir, ckpt_dir = args.runs, args.ckpt
    tok = load_tokenizer(Path("data/tok8192/tokenizer.json"))
    token_bytes = token_byte_lengths(tok)
    ids = [tok.token_to_id(BOS)] + tok.encode(args.text).ids
    idx, targets = torch.tensor([ids[:-1]]), torch.tensor([ids[1:]])

    results, bench = read_tsv(runs_dir / "results.tsv", "run_id"), read_tsv(runs_dir / "bench.tsv", "ckpt")
    configs = {d.name: json.loads((d / "config.json").read_text())
               for d in sorted(runs_dir.iterdir()) if (d / "config.json").exists()}

    def run_recipe(run):
        c = configs[run]
        return recipe(c["train"], c["model"], start_tokens(c["train"], runs_dir) + c["train"]["max_tokens"])

    def seed_key(run):  # runs that differ only in the seed
        t, m = configs[run]["train"], configs[run]["model"]
        rest = {k: t.get(k) for k in ("warmup_steps", "cooldown_frac", "min_lr_frac", "weight_decay")}
        rest.update({k: m.get(k) for k in ("d_model", "n_kv_head", "d_ff", "vocab_size")})
        return json.dumps([run_recipe(run), rest], sort_keys=True)

    seeds = {}  # seed_key -> [(run, seed, val_bpb, date)] over every finished run
    for run in configs:
        if run in results and results[run]["val_bpb"]:
            row = results[run]
            seeds.setdefault(seed_key(run), []).append(
                (run, configs[run]["train"]["seed"], float(row["val_bpb"]), row["date"]))
    groups = [g for g in seeds.values() if len(g) >= 3]
    noise = None
    if groups:
        g = max(groups, key=lambda g: (max(r[3] for r in g), len(g)))
        sd = statistics.stdev(r[2] for r in g)
        noise = {"runs": [r[0] for r in g], "bpb": [r[2] for r in g], "std": sig(sd, 3),
                 "threshold": sig(2 * math.sqrt(2) * sd, 3)}

    variants = []
    torch.manual_seed(0)
    init = GPT(GPTConfig(vocab_size=tok.get_vocab_size())).eval()
    sources = [("init", init, None)] + [(r, None, ckpt_dir / f"{r}.pt") for r in RUNS]
    for run, model, path in sources:
        if model is None and not path.exists():
            print(f"skip {run}: no checkpoint")
            continue
        info = {"run": run}
        if model is None:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            model = GPT(GPTConfig(**ckpt["model_config"])).eval()
            model.load_state_dict(ckpt["model"])
            base = run.removesuffix("-stable")
            c = configs[base]
            log = read_log(runs_dir / base, start_tokens(c["train"], runs_dir))
            rec = run_recipe(base)
            row = results.get(base, {})
            info.update(note=row.get("change", ""), date=row.get("date", ""), wall_h=row.get("wall_h"),
                        start=start_tokens(c["train"], runs_dir), mtp_weight=c["train"].get("mtp_weight"),
                        micro_batch=c["train"]["micro_batch"],
                        seed=c["train"]["seed"], lr=c["train"]["lr"], muon_lr=c["train"].get("muon_lr"),
                        warmup=c["train"]["warmup_steps"], cooldown_frac=c["train"].get("cooldown_frac"),
                        min_lr_frac=c["train"].get("min_lr_frac"), tok_s=log["tok_s"], curves=log,
                        val_bpb=float(row["val_bpb"]) if row.get("val_bpb") else None,
                        bench={k: float(v) for k, v in bench.get(base, {}).items()
                               if k not in ("ckpt", "date") and v},
                        seeds=[[r, s, b] for r, s, b, _ in seeds.get(seed_key(base), []) if r != base])
            if run.endswith("-stable"):  # a mid-run checkpoint: the val BPB is the log's at that point
                rec["tokens"] = ckpt["tokens"]
                vals = [v for v in log["val"] if v[0] <= ckpt["tokens"]]
                info.update(stable=True, val_bpb=vals[-1][1] if vals else None, bench={}, seeds=[],
                            note="", mark=ckpt["tokens"])
        else:
            cfg = model.cfg
            rec = {"arch": "llama", "n_layer": cfg.n_layer, "loop": "1", "mtp": False, "qk_norm": False,
                   "zero_init": False, "emb_std": cfg.emb_std, "optimizer": None, "lr": None,
                   "schedule": None, "batch": None, "tokens": 0, "start": "", "data": None}
            info.update(note="", val_bpb=None, bench={}, seeds=[], curves=None)  # the page writes the notes
        info.update(recipe=rec, cfg={k: getattr(model.cfg, k) for k in
                                     ("n_layer", "n_head", "n_kv_head", "d_model", "d_ff", "vocab_size",
                                      "max_seq_len", "rope_base", "arch", "qk_norm", "zero_init", "emb_std",
                                      "n_loop", "n_prelude", "n_coda", "mtp")},
                    params=param_info(model), trace=describe(model, idx, targets, info.get("mtp_weight") or 0.3))
        variants.append(info)
        print(f"traced {run}: sample loss {np.mean(info['trace']['nats']):.3f} nats")

    spectra = None
    sp = ckpt_dir / f"{SPECTRA_FROM}.pt"
    if sp.exists():
        src = configs[SPECTRA_FROM.removesuffix("-stable")]["train"]["data_dir"]
        spectra = update_spectra(sp, Path(src))
        print(f"update spectra from {sp.name}: {len(spectra['mats'])} matrices")

    base_train = configs[variants[1]["run"].removesuffix("-stable")]["train"] if len(variants) > 1 else {}
    return {
        "text": args.text, "ids": ids, "vocab": [tok.decode([i]) or tok.id_to_token(i) for i in range(tok.get_vocab_size())],  # decode drops specials
        "token_bytes": token_bytes.tolist(),
        "windows": {name: data_window(Path("data/tok8192") / name) for name in ("fineweb-edu", "tinystories")},
        "train_shape": {k: base_train.get(k) for k in ("micro_batch", "seq_len", "batch_tokens", "eval_tokens")},
        "noise": noise, "variants": variants, "spectra": spectra,
    }


def render_html(payload: dict, template: Path, out: Path) -> Path:
    html = template.read_text(encoding="utf-8")
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html.replace(PLACEHOLDER, data, 1), encoding="utf-8")
    return out


def main():
    p = argparse.ArgumentParser(description="Build the training-step data-flow page from checkpoints and runs/")
    p.add_argument("--text", default=TEXT, help="sample text the models read")
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--ckpt", type=Path, default=Path("ckpt"))
    p.add_argument("-o", "--out", type=Path, default=Path("runs/flow.html"))
    p.add_argument("--threads", type=int, default=8, help="CPU threads (the GPU is left alone)")
    p.add_argument("--serve", action="store_true", help="serve only the generated HTML over HTTP")
    p.add_argument("--serve-only", action="store_true", help="serve the existing page without rebuilding it")
    p.add_argument("--host", default="127.0.0.1", help="bind address for serving (0.0.0.0 for LAN)")
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    out = args.out
    if not args.serve_only:
        out = render_html(build_payload(args), Path(__file__).with_name("flow_template.html"), out)
        print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    elif not out.exists():
        p.error(f"{out} does not exist yet: build it first without --serve-only")
    if args.serve or args.serve_only:
        _serve_single_file(out.resolve(), args.host, args.port)


if __name__ == "__main__":
    main()
