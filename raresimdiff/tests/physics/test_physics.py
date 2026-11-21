import numpy as np
import pytest
from raresimdiff.scenarios import schema as S, generator as G
from raresimdiff.simulation.mujoco_backend import MuJoCoBackend
from raresimdiff.simulation.controllers import ScriptedBaseline, NullController
from raresimdiff.simulation.perception import PerceptionModel
from raresimdiff.simulation.rollout import run_episode
from raresimdiff.simulation.base import SimulatorBackend


def grasp_spec(event=0, sev=0.0, seed=3):
    return G.make_variant(G.make_parent(seed, task=0), event, sev, seed=1)


def nav_spec(event=2, sev=0.6, seed=1000):
    return G.make_variant(G.make_parent(seed, task=1), event, sev, seed=1)


def test_backend_implements_interface():
    be = MuJoCoBackend()
    assert isinstance(be, SimulatorBackend)
    for m in ("reset", "step", "get_observation", "get_state", "get_contacts", "get_metrics", "validate_scenario", "export_episode"):
        assert callable(getattr(be, m))


def test_deterministic_reset_and_stepping():
    sp = grasp_spec(4, 0.5)
    runs = []
    for _ in range(2):
        be = MuJoCoBackend()
        obs = be.reset(sp)
        ctl = ScriptedBaseline()
        for _ in range(60):
            obs, *_ = be.step(ctl.act(obs))
        runs.append((be.get_state()["flat"].copy(), obs.copy()))
    assert np.array_equal(runs[0][0], runs[1][0]) and np.array_equal(runs[0][1], runs[1][1])
    # reset twice on the same backend gives the same initial observation and state
    be = MuJoCoBackend()
    o1 = be.reset(sp); s1 = be.get_state()["flat"].copy()
    o2 = be.reset(sp); s2 = be.get_state()["flat"].copy()
    assert np.array_equal(o1, o2) and np.array_equal(s1, s2)


def test_scenario_loads_and_settles_without_penetration():
    be = MuJoCoBackend()
    for seed in range(8):
        sp = grasp_spec(0, 0.0, seed)
        vr = be.validate_scenario(sp)
        assert vr.ok, vr.reasons
        assert vr.details["max_penetration"] <= be.cfg.penetration_tol
        assert vr.details["max_obj_speed"] < 0.05
        z = be.data.xpos[be.obj_body[0]][2]
        assert z > S.TABLE_TOP_Z  # resting on (not through) the table


def test_validate_detects_initial_penetration():
    sp = grasp_spec(0, 0.0, 4)
    t = sp.objects[0]
    # force a deep geometric interpenetration of two objects (bypasses geometry validation on purpose)
    free = next(k for k in range(1, S.K_OBJ) if k not in {o.slot for o in sp.objects})
    sp.objects.append(S.ObjectSpec(free, 4, t.x + 0.005, t.y, 0.0, t.rot, 1.0, 0.2, 0.5))
    vr = MuJoCoBackend().validate_scenario(sp)
    assert not vr.ok and ({"INITIAL_PENETRATION", "UNSTABLE_SETTLE", "OFF_SUPPORT"} & set(vr.reasons))


def test_nominal_grasp_succeeds_and_is_lifted_physically():
    be = MuJoCoBackend()
    ok = [run_episode(be, grasp_spec(0, 0.0, s), ScriptedBaseline()).success for s in range(10)]
    assert sum(ok) >= 9
    r = run_episode(be, grasp_spec(0, 0.0, 2), ScriptedBaseline())
    assert r.metrics["finger_target_contact_steps"] > 10 and r.metrics["max_object_height"] > 0.07


def test_event_injection_friction_and_force():
    sp = grasp_spec(1, 0.9)
    be = MuJoCoBackend()
    be.reset(sp)
    gid = be.model.geom("obj0_geom").id
    mu0 = be.model.geom_friction[gid, 0]
    ctl = ScriptedBaseline()
    obs = be.get_observation()
    pre = None
    while be.t < sp.glob["event_onset"] - 0.05:
        obs, *_ = be.step(ctl.act(obs))
    assert be.model.geom_friction[gid, 0] == pytest.approx(mu0)  # untouched before onset
    for _ in range(6):
        obs, *_ = be.step(ctl.act(obs))
    assert be.model.geom_friction[gid, 0] == pytest.approx(max(0.05, mu0 * sp.glob["friction_multiplier"]))
    assert be.events.force_applied_impulse > 0
    assert be.get_metrics()["event_triggered"]


def test_severe_slip_fails_more_than_mild():
    be = MuJoCoBackend()
    res = {}
    for sev in (0.05, 0.95):
        res[sev] = np.mean([run_episode(be, G.make_variant(G.make_parent(100 + i, task=0), 1, sev, seed=i), ScriptedBaseline()).success
                            for i in range(20)])
    assert res[0.05] > res[0.95] + 0.4


def test_obstacle_moves_physically_without_teleporting():
    sp = nav_spec(2, 0.5)
    be = MuJoCoBackend()
    obs = be.reset(sp)
    start = be.data.xpos[be.obs_body[0]][:2].copy()
    ctl = NullController()
    prev, max_step, moved_before = start.copy(), 0.0, 0.0
    onset = sp.glob["event_onset"] + sp.obstacles[0].delay
    speed = np.hypot(sp.obstacles[0].vx, sp.obstacles[0].vy)
    while be.t < onset + sp.glob["event_duration"] + 0.5:
        be.step(np.zeros(6))
        cur = be.data.xpos[be.obs_body[0]][:2].copy()
        max_step = max(max_step, float(np.linalg.norm(cur - prev)))
        if be.t < onset - 0.05:
            moved_before = max(moved_before, float(np.linalg.norm(cur - start)))
        prev = cur
    assert moved_before < 0.02  # stationary before onset
    assert np.linalg.norm(prev - start) > 0.3  # moved after onset
    assert max_step <= (speed + 0.5) * be.cfg.control_dt * 1.5  # continuous motion, no jumps


def test_contact_detection_obstacle_collision():
    be = MuJoCoBackend()
    hit = 0
    for i in range(30):
        sp = G.make_variant(G.make_parent(1000 + i, task=1), 2, 0.95, seed=i)
        r = run_episode(be, sp, ScriptedBaseline())
        hit += r.termination_reason == "collision_obstacle"
        if r.termination_reason == "collision_obstacle":
            assert r.metrics["robot_obstacle_contact_steps"] >= 1
    assert hit >= 5


def test_perception_model():
    glob = {"sensor_noise_std": 0.0, "occlusion_dropout_prob": 0.0, "visibility_range": 3.0, "calib_yaw_err": 0.0,
            "calib_tx": 0.0, "calib_ty": 0.0}
    base = np.array([0.5, 0.2])
    pts = np.array([[1.0, 0.3], [2.0, -0.4]])
    v, est = PerceptionModel(glob, 0).observe(base, 0.3, pts)
    assert v.all() and np.allclose(est, pts, atol=1e-9)
    v, _ = PerceptionModel(dict(glob, visibility_range=1.0), 0).observe(base, 0.0, pts)
    assert list(v) == [True, False]
    _, est = PerceptionModel(dict(glob, calib_tx=0.04), 0).observe(base, 0.0, pts)
    assert np.allclose(est - pts, [0.04, 0.0], atol=1e-9)
    _, est = PerceptionModel(dict(glob, calib_yaw_err=5.0), 0).observe(np.zeros(2), 0.0, np.array([[1.0, 0.0]]))
    assert np.allclose(est[0], [np.cos(np.deg2rad(5)), np.sin(np.deg2rad(5))], atol=1e-9)
    drops = [PerceptionModel(dict(glob, occlusion_dropout_prob=0.4), s).observe(base, 0, pts[:1])[0][0] for s in range(2000)]
    assert 0.55 < np.mean(drops) < 0.65
    pm1, pm2 = PerceptionModel(dict(glob, sensor_noise_std=0.02), 7), PerceptionModel(dict(glob, sensor_noise_std=0.02), 7)
    assert np.array_equal(pm1.observe(base, 0, pts)[1], pm2.observe(base, 0, pts)[1])  # reproducible


def test_perception_degradation_does_not_change_dynamics_only_observations():
    base = grasp_spec(0, 0.0, 6)
    deg = grasp_spec(3, 0.9, 6)
    be1, be2 = MuJoCoBackend(), MuJoCoBackend()
    be1.reset(base); be2.reset(deg)
    for _ in range(20):
        be1.step(np.zeros(6)); be2.step(np.zeros(6))
    assert np.allclose(be1.get_state()["flat"], be2.get_state()["flat"])  # same physics with an idle robot
    assert not np.allclose(be1.get_observation()[15:18], be2.get_observation()[15:18]) or deg.glob["sensor_noise_std"] == 0


def test_no_nan_across_many_scenarios():
    be = MuJoCoBackend()
    for sp in list(G.generate_parents_and_variants(25, 3, 11)):
        r = run_episode(be, sp, ScriptedBaseline())
        assert not r.metrics["nan_detected"], sp.scenario_id
        assert np.isfinite(r.traj_states).all()
