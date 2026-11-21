import json
import numpy as np
import pytest
from raresimdiff.scenarios import schema as S, codec as C, generator as G
from raresimdiff.scenarios.severity import compute_severity
from raresimdiff.validation.geometry import validate_geometry
from raresimdiff.validation.projection import project
from raresimdiff.validation import reasons as R


def specs(n=30, seed=1):
    return list(G.generate_parents_and_variants(n, 3, seed))


def test_variable_tables_consistent():
    for vs in (S.GLOBAL_VARS, S.OBJ_VARS, S.OBS_VARS, S.ENV_VARS):
        names = [v.name for v in vs]
        assert len(set(names)) == len(names)
        for v in vs:
            assert v.lo < v.hi and v.unit
            assert v.lo <= v.nominal <= v.hi or v.name in ("x", "y", "dz")  # object nominal pose is unused
    assert S.L == 1 + S.K_OBJ + S.M_OBS and S.CH_MASK.shape == (S.L, S.D)


def test_spec_serialization_roundtrip():
    for sp in specs(10):
        d = json.loads(json.dumps(sp.to_dict()))
        sp2 = S.ScenarioSpec.from_dict(d)
        assert sp2.to_dict() == sp.to_dict()


def test_rotation_representation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        r6 = rng.normal(size=6)
        R_ = S.rot6_to_matrix(r6)
        assert S.is_rotation(R_)
        assert np.allclose(S.rot6_to_matrix(S.matrix_to_rot6(R_)), R_, atol=1e-9)
        q = S.matrix_to_quat(R_)
        assert abs(np.linalg.norm(q) - 1) < 1e-9
    # degenerate input still yields a valid rotation
    assert S.is_rotation(S.rot6_to_matrix(np.zeros(6)))
    assert S.is_rotation(S.rot6_to_matrix(np.array([1, 0, 0, 2, 0, 0.0])))


def test_pack_unpack_roundtrip():
    for sp in specs(20):
        phys, mask, aid = C.pack_spec(sp)
        sp2 = C.unpack_to_spec(phys, mask, aid, sp.task, sp.event, sp.robot_profile, sp.env, sp.scenario_id, sp.parent_id,
                               sp.seed)
        assert np.allclose(C.pack_spec(sp2)[0], phys, atol=1e-9)
        assert abs(sp2.severity - sp.severity) < 1e-9
        assert [o.asset_id for o in sp2.objects] == [o.asset_id for o in sp.objects]


def test_units_and_normalisation_inverse():
    sps = specs(40)
    phys = np.stack([C.pack_spec(s)[0] for s in sps])
    mask = np.stack([C.pack_spec(s)[1] for s in sps])
    norm = C.Normalizer.fit(phys, mask)
    x = norm.normalize(phys)
    back = norm.denormalize(x, clamp=False)
    w = mask[:, :, None] * S.CH_MASK[None]
    assert np.allclose(back * w, phys * w, atol=1e-4)
    assert np.all(x[:, :, :] * (1 - S.CH_MASK[None]) == 0)  # padded channels are zero
    lo, hi = norm.x0_limits()
    assert np.all(lo < hi)


def test_stats_use_only_given_split():
    sps = specs(40)
    phys = np.stack([C.pack_spec(s)[0] for s in sps])
    mask = np.stack([C.pack_spec(s)[1] for s in sps])
    a = C.Normalizer.fit(phys[:20], mask[:20])
    b = C.Normalizer.fit(phys, mask)
    assert not np.allclose(a.mean, b.mean)


def test_severity_function():
    sp = G.make_variant(G.make_parent(1, task=0), 0, 0.5, seed=1)
    assert sp.severity == 0.0
    lo = G.make_variant(G.make_parent(1, task=0), 1, 0.05, seed=2)
    hi = G.make_variant(G.make_parent(1, task=0), 1, 0.95, seed=2)
    assert 0 <= lo.severity < hi.severity <= 1
    for sp in specs(30):
        assert 0.0 <= sp.severity <= 1.0
        assert compute_severity(sp.event, sp.glob, [o.__dict__ for o in sp.obstacles]) == pytest.approx(sp.severity)


def test_generated_specs_valid_and_deterministic():
    a, b = specs(15, 5), specs(15, 5)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
    bad = [(s.scenario_id, validate_geometry(s)) for s in a if validate_geometry(s)]
    assert len(bad) <= 0.1 * len(a), bad


def test_geometry_rejections():
    p = G.make_parent(3, task=0)
    sp = G.make_variant(p, 1, 0.5, seed=1)
    assert validate_geometry(sp) == []
    sp.glob["friction_multiplier"] = 3.0
    assert R.OUT_OF_BOUNDS in validate_geometry(sp)
    sp = G.make_variant(p, 1, 0.5, seed=1)
    sp.objects = [o for o in sp.objects if o.slot != 0]
    assert R.NO_TARGET in validate_geometry(sp)
    sp = G.make_variant(G.make_parent(3, task=0), 1, 0.5, seed=1)
    c = S.ObjectSpec(1, 4, sp.objects[0].x, sp.objects[0].y, 0.0, sp.objects[0].rot, 1.0, 0.2, 0.5)
    sp.objects.append(c)
    assert R.OBJ_OVERLAP in validate_geometry(sp)
    sp = G.make_variant(p, 2, 0.5, seed=1)  # obstacle event on a grasp task
    assert R.EVENT_TASK_MISMATCH in validate_geometry(sp) or R.MASK_INCONSISTENT in validate_geometry(sp)
    sp = G.make_variant(p, 1, 0.5, seed=1)
    sp.glob["sensor_noise_std"] = 0.02  # inactive group must stay nominal
    assert R.INACTIVE_NOT_NOMINAL in validate_geometry(sp)


def test_projection_repairs_invalid_raw_sample():
    sp = G.make_variant(G.make_parent(5, task=0), 4, 0.6, seed=3)
    phys, mask, aid = C.pack_spec(sp)
    raw = phys.copy()
    raw[0, S.GLOBAL_IDX["friction_multiplier"]] = -2.0
    raw[0, S.GLOBAL_IDX["sensor_noise_std"]] = 5.0
    raw[1, S.OBJ_IDX["r0x"]:S.OBJ_IDX["r1z"] + 1] = [2, 2, 2, 2, 2, 2]  # not a valid rotation
    raw[1, S.OBJ_IDX["x"]] = 99.0
    p, mag, _ = project(raw, mask, aid, sp.task, sp.event, sp.env)
    spec = C.unpack_to_spec(p, mask, aid, sp.task, sp.event, 0, sp.env, "x", "x", 0)
    assert mag > 0
    assert validate_geometry(spec) == [] or set(validate_geometry(spec)) <= {R.OBJ_OVERLAP}
    assert S.is_rotation(np.asarray(spec.objects[0].rot), 1e-6)
    # inactive channels reset to nominal for an event without them
    sp2 = G.make_variant(G.make_parent(5, task=0), 1, 0.6, seed=3)
    phys2, mask2, aid2 = C.pack_spec(sp2)
    raw2 = phys2.copy()
    raw2[0, S.GLOBAL_IDX["sensor_noise_std"]] = 0.03
    p2, _, _ = project(raw2, mask2, aid2, sp2.task, sp2.event, sp2.env)
    assert p2[0, S.GLOBAL_IDX["sensor_noise_std"]] == 0.0


def test_projection_idempotent_on_valid_sample():
    for sp in specs(10):
        if validate_geometry(sp):
            continue
        phys, mask, aid = C.pack_spec(sp)
        p, mag, _ = project(phys, mask, aid, sp.task, sp.event, sp.env)
        assert mag < 1e-6
        assert np.allclose(p, phys, atol=1e-9)
