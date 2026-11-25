import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow importing scripts/tree.py directly like scripts/scaling.py imports plot.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import tree as tree_mod  # noqa: E402


def _write_run(runs_dir: Path, ckpt_dir: Path, run_id: str, cfg_overrides=None, log_rows=None, samples=None, ckpt=True):
    rdir = runs_dir / run_id
    rdir.mkdir(parents=True, exist_ok=True)
    train_cfg = {
        "data_dir": "data/tok8192/fineweb-edu",
        "n_layer": 10,
        "d_model": 640,
        "n_head": 10,
        "n_kv_head": 5,
        "d_ff": 1728,
        "max_tokens": 1e9,
        "batch_tokens": 131072,
        "schedule": "wsd",
        "lr": 0.002,
        "warmup_steps": 100,
        "cooldown_frac": 0.2,
        "min_lr_frac": 0.0,
        "weight_decay": 0.1,
        "seed": 0,
        "cooldown_from": "",
    }
    if cfg_overrides:
        train_cfg.update(cfg_overrides)
    cfg = {
        "train": train_cfg,
        "model": {
            "vocab_size": 8192,
            "n_layer": train_cfg["n_layer"],
            "d_model": train_cfg["d_model"],
            "n_head": train_cfg["n_head"],
            "n_kv_head": train_cfg["n_kv_head"],
            "d_ff": train_cfg["d_ff"],
        },
        "params": 50721920,
    }
    if "arch" in train_cfg:
        cfg["model"]["arch"] = train_cfg["arch"]
    (rdir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    if log_rows is not None:
        (rdir / "log.jsonl").write_text("\n".join(json.dumps(r) for r in log_rows) + "\n", encoding="utf-8")
    if samples is not None:
        (rdir / "samples.md").write_text(samples, encoding="utf-8")
    if ckpt:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / f"{run_id}.pt").write_bytes(b"0" * 2048)


def test_load_bench_merges_duplicate_rows_and_computes_core(tmp_path):
    (tmp_path / "bench.tsv").write_text(
        "ckpt\tdate\thellaswag\tarc_easy\tpiqa\tblimp\tsciq\tlambada\n"
        "run-a\t2026-11-24\t0.2700\t0.3800\t0.5700\t0.7400\t\t\n"
        "run-a\t2026-11-25\t0.2750\t0.3850\t0.5750\t0.7450\t0.6100\t0.1800\n"
        "run-a\t2026-11-25\t0.2760\t\t\t\t\t\n",
        encoding="utf-8",
    )
    benches = tree_mod.load_bench(tmp_path)
    assert "run-a" in benches
    s = benches["run-a"]["scores"]
    # Latest non-empty wins per column; empty trailing row must not wipe earlier Tier 2 columns.
    assert math.isclose(s["hellaswag"], 0.2760)
    assert math.isclose(s["sciq"], 0.6100)
    assert math.isclose(s["lambada"], 0.1800)
    assert benches["run-a"]["tier"] == "tier2"
    assert benches["run-a"]["core"] is not None


def test_parse_final_samples_and_log(tmp_path):
    samples_md = (
        "## step 1000\n\n"
        "- **Prompt A** old output\n\n"
        "## step 2000\n\n"
        "- **Prompt A** line 1\nline 2 with </script> tag\n"
        "- **Prompt B** second output\n"
    )
    sp = tmp_path / "samples.md"
    sp.write_text(samples_md, encoding="utf-8")
    parsed = tree_mod.parse_final_samples(sp)
    assert parsed["step"] == 2000
    assert len(parsed["items"]) == 2
    assert parsed["items"][0]["prompt"] == "Prompt A"
    assert "line 2" in parsed["items"][0]["text"]
    assert parsed["items"][1]["prompt"] == "Prompt B"

    lp = tmp_path / "log.jsonl"
    lp.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"step": 0, "tokens": 0, "val_bpb": 3.5},
                {"step": 10, "tokens": 1_000_000, "tok_per_s": 80000, "mfu": 0.30},
                {"step": 20, "tokens": 2_000_000, "tok_per_s": 170000, "mfu": 0.68},
                {"step": 30, "tokens": 3_000_000, "tok_per_s": 172000, "mfu": 0.69},
                {"step": 30, "tokens": 3_000_000, "val_bpb": 1.10},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    info = tree_mod.parse_log(lp)
    assert info["val_curve"] == [[0.0, 3.5], [3.0, 1.1]]
    assert info["tok_per_s"] == 171000
    assert math.isclose(info["mfu"], 0.685, abs_tol=1e-4)


def test_build_tree_payload_cross_compare_disabled_and_html_escaping(tmp_path):
    runs_dir = tmp_path / "runs"
    ckpt_dir = tmp_path / "ckpt"
    runs_dir.mkdir()
    (runs_dir / "bench").mkdir()

    (runs_dir / "results.tsv").write_text(
        "run_id\tdate\tgit_sha\tchange\tparams\ttokens\twall_h\tval_bpb\thellaswag\tnotes\n"
        "root-gpt2\t2026-11-25\taaa1111\tgpt2 vanilla\t50.7M\t1000M\t1.59\t1.1072\t0.2741\t\n"
        "v0-llama\t2026-11-24\tbbb2222\tv0 512k cosine\t50.7M\t1000M\t1.64\t1.1212\t0.2758\t\n"
        "head-wsd\t2026-11-25\tccc3333\t128k wsd 2e-3\t50.7M\t1000M\t1.67\t1.0850\t0.2812\t\n"
        "branch-ts\t2026-11-23\tddd4444\ttinystories\t50.7M\t500M\t0.82\t0.5171\t0.2650\t\n"
        "branch-noise\t2026-11-25\teee5555\tlr 2.8e-3\t50.7M\t1000M\t1.67\t1.0849\t0.2775\t\n"
        "orphan-run\t2026-11-25\tfff6666\tnew experiment\t50.7M\t1000M\t1.67\t1.0800\t0.2820\t\n",
        encoding="utf-8",
    )

    _write_run(runs_dir, ckpt_dir, "root-gpt2", {"arch": "gpt2", "n_kv_head": 10, "d_ff": 2216})
    _write_run(
        runs_dir,
        ckpt_dir,
        "v0-llama",
        {"batch_tokens": 524288, "schedule": "cosine"},
        samples="## step 100\n\n- **Test </script>** safe text\n",
    )
    _write_run(runs_dir, ckpt_dir, "head-wsd", {})
    _write_run(runs_dir, ckpt_dir, "branch-ts", {"data_dir": "data/tok8192/tinystories"})
    _write_run(runs_dir, ckpt_dir, "branch-noise", {"lr": 0.0028})
    _write_run(runs_dir, ckpt_dir, "orphan-run", {})

    # In-progress run directory (not in results.tsv yet)
    _write_run(runs_dir, ckpt_dir, "running-seed1", log_rows=[{"step": 100, "tokens": 13_107_200, "loss": 3.5}], ckpt=False)

    # Save paired .npz for root-gpt2 and head-wsd
    np.savez_compressed(
        runs_dir / "bench" / "root-gpt2.npz",
        hellaswag__correct=np.array([0, 0, 1, 1] * 25, dtype=np.uint8),
        hellaswag__prob=np.full(100, 0.26, dtype=np.float32),
        hellaswag__target_bpb=np.full(100, 1.5, dtype=np.float32),
    )
    np.savez_compressed(
        runs_dir / "bench" / "head-wsd.npz",
        hellaswag__correct=np.array([1, 1, 1, 1] * 25, dtype=np.uint8),
        hellaswag__prob=np.full(100, 0.30, dtype=np.float32),
        hellaswag__target_bpb=np.full(100, 1.4, dtype=np.float32),
    )

    spec_path = tmp_path / "tree.toml"
    spec_path.write_text(
        'title = "实验树"\n'
        'title_en = "Experiment Tree"\n'
        'noise_floor = 0.005\n'
        '[nodes."root-gpt2"]\nstatus = "root"\ncompare_to = "head-wsd"\nlabel = "GPT-2 原版"\nlabel_en = "Vanilla GPT-2"\ntakeaway = "root"\ntakeaway_en = "root en"\n'
        '[nodes."v0-llama"]\nparent = "root-gpt2"\nstatus = "adopted"\ncompare_to = ""\nlabel = "v0"\ntakeaway = "v0"\n'
        '[nodes."head-wsd"]\nparent = "v0-llama"\nstatus = "adopted"\nlabel = "WSD"\ntakeaway = "head"\n'
        '[nodes."branch-ts"]\nparent = "v0-llama"\nstatus = "branch"\ncompare_to = ""\nlabel = "TS"\ntakeaway = "ts"\n'
        '[nodes."branch-noise"]\nparent = "head-wsd"\nstatus = "branch"\nlabel = "lr2.8"\ntakeaway = "flat"\n',
        encoding="utf-8",
    )

    payload = tree_mod.build_tree_payload(runs_dir, ckpt_dir, spec_path)
    nodes = payload["nodes"]

    # Bilingual fields pass through (optional *_en defaults to empty string)
    assert payload["meta"]["title_en"] == "Experiment Tree"
    assert nodes["root-gpt2"]["label_en"] == "Vanilla GPT-2"
    assert nodes["root-gpt2"]["takeaway_en"] == "root en"
    assert nodes["v0-llama"]["label_en"] == ""

    # Cross-recipe comparison on root -> head-wsd (+0.0222, paired bench computed)
    assert nodes["root-gpt2"]["cross_compare"] is True
    assert math.isclose(nodes["root-gpt2"]["delta_bpb"], 0.0222, abs_tol=1e-4)
    assert len(nodes["root-gpt2"]["paired_bench"]) == 1
    assert any(d["key"] == "arch" and d["label_en"] == "Architecture" and d["to"] == "gpt2" for d in nodes["root-gpt2"]["config_diff"])
    assert nodes["head-wsd"]["compared_by"][0]["delta_from_other"] == -0.0222

    # Explicitly disabled compare_to must leave delta_bpb as None
    assert nodes["v0-llama"]["delta_bpb"] is None
    assert nodes["branch-ts"]["delta_bpb"] is None

    # Noise floor detection
    assert nodes["branch-noise"]["delta_bpb"] == -0.0001
    assert nodes["branch-noise"]["in_noise"] is True

    # Uncataloged finished run + in-progress run detected
    assert [r["run_id"] for r in payload["uncataloged"]] == ["orphan-run"]
    assert [r["run_id"] for r in payload["in_progress"]] == ["running-seed1"]

    # HTML rendering escapes </script> safely
    template_path = Path(__file__).resolve().parent.parent / "scripts" / "tree_template.html"
    out_html = tree_mod.render_html(payload, template_path, runs_dir / "tree.html")
    html = out_html.read_text(encoding="utf-8")
    assert tree_mod.PLACEHOLDER not in html
    assert "</script>**" not in html
    assert "\\u003c/script>" in html


def test_build_tree_payload_rejects_missing_parent(tmp_path):
    spec = tmp_path / "bad.toml"
    spec.write_text('[nodes."a"]\nparent = "ghost"\nstatus = "branch"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing parent"):
        tree_mod.build_tree_payload(tmp_path, tmp_path, spec)


def test_scaling_chart_power_law_fit(tmp_path):
    runs_dir = tmp_path / "runs"
    ckpt_dir = tmp_path / "ckpt"
    runs_dir.mkdir()
    (runs_dir / "results.tsv").write_text(
        "run_id\tdate\tgit_sha\tchange\tparams\ttokens\twall_h\tval_bpb\thellaswag\tnotes\n"
        "s-5m\t2026-11-24\taaa\t5m\t5.0M\t100M\t0.02\t1.4209\t\t\n"
        "s-5m-hi\t2026-11-24\taaa\t5m\t5.0M\t100M\t0.02\t1.4310\t\t\n"
        "s-50m\t2026-11-24\tbbb\t50m\t50.7M\t1000M\t1.64\t1.1212\t\t\n",
        encoding="utf-8",
    )
    _write_run(runs_dir, ckpt_dir, "s-5m", {"n_layer": 4, "d_model": 256, "max_tokens": 1e8, "batch_tokens": 65536, "lr": 0.004})
    _write_run(runs_dir, ckpt_dir, "s-5m-hi", {"n_layer": 4, "d_model": 256, "max_tokens": 1e8, "batch_tokens": 65536, "lr": 0.008})
    # Adjust exact params for 5M runs so by_size separates 5M and 50M
    for rid in ("s-5m", "s-5m-hi"):
        p = runs_dir / rid / "config.json"
        c = json.loads(p.read_text())
        c["params"] = 5_000_000
        c["train"]["seq_len"] = 1024
        p.write_text(json.dumps(c))
    _write_run(runs_dir, ckpt_dir, "s-50m", {"max_tokens": 1e9, "batch_tokens": 524288, "lr": 0.002})
    p50 = runs_dir / "s-50m" / "config.json"
    c50 = json.loads(p50.read_text())
    c50["train"]["seq_len"] = 1024
    p50.write_text(json.dumps(c50))

    spec = tmp_path / "tree.toml"
    spec.write_text(
        '[nodes."s-50m"]\nstatus = "root"\nlabel = "50M"\n'
        '[nodes."grp-scale"]\nparent = "s-50m"\nstatus = "group"\nchart = "scaling"\nchart_extra_runs = ["s-50m"]\nlabel = "Scaling"\n'
        '[nodes."s-5m"]\nparent = "grp-scale"\nstatus = "branch"\n'
        '[nodes."s-5m-hi"]\nparent = "grp-scale"\nstatus = "branch"\n',
        encoding="utf-8",
    )
    payload = tree_mod.build_tree_payload(runs_dir, ckpt_dir, spec)
    sc = payload["nodes"]["grp-scale"]["scaling_chart"]
    assert sc["kind"] == "scaling"
    assert len(sc["points"]) == 3
    assert sc["best_ids"] == ["s-5m", "s-50m"]
    assert sc["fit"]["b"] > 0 and sc["fit"]["drop_per_2x_pct"] > 0


def test_cooldown_from_shifts_val_curve_by_branch_tokens(tmp_path):
    runs_dir = tmp_path / "runs"
    ckpt_dir = tmp_path / "ckpt"
    runs_dir.mkdir()
    _write_run(
        runs_dir,
        ckpt_dir,
        "base-wsd",
        {"batch_tokens": 131072, "save_stable_at": 500_000_000.0},
    )
    _write_run(
        runs_dir,
        ckpt_dir,
        "branch-cd",
        {"cooldown_from": "ckpt/base-wsd-stable.pt", "max_tokens": 1e8},
        log_rows=[
            {"step": 0, "tokens": 0, "val_bpb": 1.1805},
            {"step": 762, "tokens": 99_876_864, "val_bpb": 1.1252},
        ],
    )
    art = tree_mod.load_run_artifacts("branch-cd", runs_dir, ckpt_dir)
    assert math.isclose(art["token_offset_m"], 500.04, abs_tol=0.05)
    assert math.isclose(art["val_curve"][0][0], 500.04, abs_tol=0.05)
    assert math.isclose(art["val_curve"][-1][0], 599.92, abs_tol=0.05)


def test_inspired_by_builds_bidirectional_dag_and_validates(tmp_path):
    runs_dir = tmp_path / "runs"
    ckpt_dir = tmp_path / "ckpt"
    runs_dir.mkdir()
    spec = tmp_path / "tree.toml"
    spec.write_text(
        '[nodes."a"]\nstatus = "root"\nlabel = "Root"\n'
        '[nodes."b"]\nparent = "a"\nstatus = "branch"\nlabel = "Branch"\n'
        '[nodes."c"]\nparent = "a"\nstatus = "adopted"\nlabel = "Adopted"\n'
        'inspired_by = ["a", { from = "b", reason = "负结果启发", reason_en = "Negative result insight" }]\n',
        encoding="utf-8",
    )
    payload = tree_mod.build_tree_payload(runs_dir, ckpt_dir, spec)
    nodes = payload["nodes"]
    assert [x["id"] for x in nodes["c"]["inspired_by"]] == ["a", "b"]
    assert nodes["c"]["inspired_by"][1]["reason_en"] == "Negative result insight"
    assert [x["id"] for x in nodes["b"]["inspired"]] == ["c"]

    bad = tmp_path / "bad_inspire.toml"
    bad.write_text('[nodes."a"]\nstatus = "root"\ninspired_by = ["missing"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing inspired_by"):
        tree_mod.build_tree_payload(runs_dir, ckpt_dir, bad)


def test_external_refs_bidirectional_and_validation(tmp_path):
    runs_dir = tmp_path / "runs"
    ckpt_dir = tmp_path / "ckpt"
    runs_dir.mkdir()
    spec = tmp_path / "tree.toml"
    spec.write_text(
        '[nodes."a"]\nstatus = "root"\nlabel = "Root"\n'
        'refs = ["ref-1", { id = "ref-2", reason = "公式吻合", reason_en = "Formula matched" }]\n'
        '[refs."ref-1"]\nkind = "method"\nlabel = "WSD"\nurl = "https://arxiv.org/abs/2405.18392"\n'
        '[refs."ref-2"]\nkind = "paper"\nlabel = "DeepSeek"\nlabel_en = "DeepSeek Law"\n'
        'links = [{ label = "arXiv", url = "https://arxiv.org/abs/2401.02954" }]\n',
        encoding="utf-8",
    )
    payload = tree_mod.build_tree_payload(runs_dir, ckpt_dir, spec)
    assert len(payload["refs"]) == 2
    assert payload["refs"]["ref-1"]["links"][0]["url"] == "https://arxiv.org/abs/2405.18392"
    assert payload["refs"]["ref-2"]["used_by"][0]["id"] == "a"
    assert payload["nodes"]["a"]["refs"][1]["reason_en"] == "Formula matched"

    bad = tmp_path / "bad_ref.toml"
    bad.write_text('[nodes."a"]\nstatus = "root"\nrefs = ["ghost"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing ref"):
        tree_mod.build_tree_payload(runs_dir, ckpt_dir, bad)
