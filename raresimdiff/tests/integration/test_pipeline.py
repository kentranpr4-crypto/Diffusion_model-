import copy, json, subprocess, sys
from pathlib import Path
import numpy as np
import pytest
import torch
from raresimdiff import generate
from raresimdiff.data.build_dataset import build_dataset
from raresimdiff.data.integrity import verify_dataset
from raresimdiff.data.dataset import load_split, load_normalizer
from raresimdiff.models.dit import RareScenarioDiT
from raresimdiff.policy_integration import exporters, metrics as PM, policies
from raresimdiff.policy_integration.env import ScenarioEnv
from raresimdiff.scenarios import codec as C, generator as G, schema as S
from raresimdiff.scenarios.prior import CategoricalPrior
from raresimdiff.training.export import export_generator, train_support
from raresimdiff.training import gpu_preflight
from raresimdiff.validation.pipeline import decode_candidates, validate_candidates
from raresimdiff.simulation.mujoco_backend import MuJoCoBackend


def test_dataset_integrity_and_split_separation(tiny_dataset):
    rep = verify_dataset(tiny_dataset)
    assert rep["ok"], rep["failures"]
    assert all(rep["checks"].values())
    man = json.load(open(tiny_dataset / "dataset_manifest.json"))
    assert man["counts"]["train"] > 100 and man["counts"]["val"] > 0
    assert (tiny_dataset / "manifests" / "train.manifest.json").exists()
    st = json.load(open(tiny_dataset / "preprocessing_stats.json"))
    assert st["source"]["computed_from"] == "train"


def test_integrity_detects_corruption(tiny_dataset, tmp_path):
    import shutil
    d = tmp_path / "copy"
    shutil.copytree(tiny_dataset, d)
    z = dict(np.load(d / "tensors_val.npz"))
    z["phys"][0, 0, 0] = np.nan
    np.savez_compressed(d / "tensors_val.npz", **z)
    rep = verify_dataset(d)
    assert not rep["ok"] and not rep["checks"]["file_hashes"]


def test_dataset_build_is_deterministic(tmp_path, cfg_dir):
    a = build_dataset(cfg_dir, str(tmp_path / "a"), n_parents=12, variants=3, workers=1, base_seed=5, quiet=True)
    b = build_dataset(cfg_dir, str(tmp_path / "b"), n_parents=12, variants=3, workers=2, base_seed=5, quiet=True)
    assert a["fingerprint"] == b["fingerprint"]
    with pytest.raises(FileExistsError):
        build_dataset(cfg_dir, str(tmp_path / "a"), n_parents=1, variants=1, workers=1, quiet=True)


def test_end_to_end_structural_with_random_weights(tiny_dataset, small_cfg, tmp_path):
    """Procedural scenario -> encode -> diffusion with RANDOM weights -> decode -> validate -> export episode.
    Structural smoke test only: random weights carry no learned capability."""
    torch.manual_seed(0)
    tr = load_split(tiny_dataset, "train")
    norm = load_normalizer(tiny_dataset)
    model = RareScenarioDiT(small_cfg.model)
    ck = tmp_path / "rand.safetensors"
    export_generator(ck, model.state_dict(), small_cfg.model, small_cfg.diffusion, norm, CategoricalPrior.from_split(tr),
                     train_support(tr), {"run_id": "rand", "step": 0})
    out = tmp_path / "gen"
    generate.main(["--checkpoint", str(ck), "--event-type", "object_slip", "--severity", "0.4", "0.9", "--num-scenarios", "12",
                   "--validate-physics", "--output", str(out), "--ddim-steps", "5"])
    s = json.load(open(out / "summary.json"))
    assert s["requested"] == 24 and s["accepted"] + s["rejected"] == 24
    rej = [json.loads(l) for l in open(out / "scenarios_rejected.jsonl")]
    assert all(r["validation"]["reasons"] for r in rej)  # every rejection carries a reason code
    # a perfect "denoiser": real scenario + tiny noise must decode, validate and export a replayable episode
    sp = G.make_variant(G.make_parent(21, task=0), 1, 0.6, seed=1)
    phys, mask, aid = C.pack_spec(sp)
    x = norm.normalize(phys[None]) + 0.01 * np.random.default_rng(0).normal(size=(1, S.L, S.D)).astype(np.float32) * S.CH_MASK
    cond = {"task": np.array([0]), "event": np.array([1]), "robot": np.array([0]), "severity": np.array([sp.severity], np.float32),
            "env": np.array([[sp.env[v.name] for v in S.ENV_VARS]], np.float32), "asset_ids": aid[None], "slot_mask": mask[None].astype(np.float32)}
    cands = decode_candidates(x, cond, norm)
    res = validate_candidates(cands, MuJoCoBackend())
    assert res[0].accepted, res[0].reasons
    meta = exporters.export_transitions([res[0].spec], policies.ScriptedBaselinePolicy(), tmp_path / "ep.npz")
    assert meta["n_transitions"] > 10
    # replayability: same scenario -> same trajectory
    m2 = exporters.export_transitions([res[0].spec], policies.ScriptedBaselinePolicy(), tmp_path / "ep2.npz")
    assert meta["sha256"] == m2["sha256"] or np.array_equal(np.load(tmp_path / "ep.npz")["action"], np.load(tmp_path / "ep2.npz")["action"])


def test_ood_conditions_emit_warnings(tiny_dataset, small_cfg, tmp_path):
    from raresimdiff.generate import support_warnings
    sup = {"object_slip": {"severity_min": 0.0, "severity_max": 0.9, "n": 10, "tasks": [0]}}
    assert support_warnings(sup, "object_slip", 0.5) == []
    assert "OUT_OF_DISTRIBUTION" in support_warnings(sup, "object_slip", 0.99)[0]
    assert "never appeared" in support_warnings(sup, "moving_obstacle", 0.5)[0]


def test_policy_export_fields_and_no_fabricated_actions(tmp_path):
    sps = G.generate_conditioned(1, 0.5, 3)
    meta = exporters.export_transitions(sps, policies.ScriptedBaselinePolicy(), tmp_path / "t.npz")
    z = np.load(tmp_path / "t.npz")
    for k in exporters.FIELDS:
        assert k in z.files
    assert z["is_synthetic"].all() and not z["is_expert_demo"].any()
    assert z["observation"].shape[0] == z["next_observation"].shape[0] == z["action"].shape[0]
    assert set(z["action_source"]) == {"scripted_baseline"}
    # next_observation chains with observation within an episode
    e0 = z["episode_index"] == 0
    assert np.allclose(z["observation"][e0][1:], z["next_observation"][e0][:-1])


def test_policy_evaluation_metrics_and_env():
    specs = G.generate_conditioned(1, 0.2, 4) + G.generate_conditioned(1, 0.9, 4) + [G.make_variant(G.make_parent(9, task=0), 0, 0, seed=1)]
    r = PM.evaluate_policy(policies.ScriptedBaselinePolicy(), specs, nominal_reference=1.0)
    for k in ("task_completion_rate", "failure_rate_by_event_family", "contact_or_collision_violation_rate",
              "recovery_success_rate", "nominal_regression", "success_by_severity_bin"):
        assert k in r
    null = PM.evaluate_policy(policies.NullPolicy(), specs)
    assert null["task_completion_rate"] <= r["task_completion_rate"]
    env = ScenarioEnv()
    obs, info = env.reset(options={"scenario": specs[0]})
    obs2, rew, term, trunc, _ = env.step(np.zeros(6))
    assert obs.shape == obs2.shape == (28,) and info["is_synthetic"]


def test_gpu_preflight_refuses_cpu_and_takes_no_steps(tiny_dataset, cfg_dir):
    rep = gpu_preflight.run_preflight(cfg_dir, str(tiny_dataset), allow_cpu=False, probe=False)
    assert rep["optimizer_steps_taken"] == 0
    if not torch.cuda.is_available():
        assert not rep["ok"] and not rep["checks"]["cuda_available"]["ok"]
    rep2 = gpu_preflight.run_preflight(cfg_dir, str(tiny_dataset), allow_cpu=True, probe=False)
    assert rep2["checks"]["dataset_integrity"]["ok"]


def test_official_run_orchestrator_smoke_and_no_overwrite(tiny_dataset, tmp_path, cfg_dir):
    from raresimdiff.training import run_official
    args = ["--config-dir", cfg_dir, "--data-dir", str(tiny_dataset), "--runs-root", str(tmp_path / "runs"), "--allow-cpu",
            "--max-steps", "6", "--eval-n", "4", "--eval-workers", "1"]
    run_official.main(["--preflight-only"] + args)
    assert not (tmp_path / "runs").exists()  # preflight-only creates nothing and trains nothing
    run_official.main(args)
    runs = list((tmp_path / "runs").iterdir())
    assert len(runs) == 1 and (runs[0] / "COMPLETED").exists() and (runs[0] / "export" / "best.safetensors").exists()
    assert (runs[0] / "REPORT.json").exists()
    with pytest.raises(SystemExit) as e:
        run_official.main(args)  # completed run is never overwritten
    assert e.value.code == 3
