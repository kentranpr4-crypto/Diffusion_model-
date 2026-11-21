import numpy as np
from raresimdiff.scenarios import schema as S, generator as G
from raresimdiff.utils.hashing import stable_json_hash


def _rnd(o):
    if isinstance(o, float):
        return round(o, 6)
    if isinstance(o, dict):
        return {k: _rnd(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_rnd(v) for v in o]
    return o


def test_schema_layout_is_frozen():
    layout = [S.L, S.D, S.K_OBJ, S.M_OBS, [v.name for v in S.GLOBAL_VARS], [v.name for v in S.OBJ_VARS],
              [v.name for v in S.OBS_VARS], [v.name for v in S.ENV_VARS], list(S.EVENTS), list(S.TASKS), list(S.ASSETS)]
    assert stable_json_hash(layout) == "3652c57397901df078780dd4ac25375304525ae7b2fd315161c869c271f63c2e"
    assert S.L == 9 and S.D == 12


def test_procedural_generator_golden_hash():
    specs = list(G.generate_parents_and_variants(10, 3, 12345))
    h = stable_json_hash(_rnd([s.to_dict() for s in specs]))
    assert h == GOLDEN, h


def test_event_mix_and_severity_coverage():
    specs = list(G.generate_parents_and_variants(200, 4, 1))
    ev = np.bincount([s.event for s in specs], minlength=5) / len(specs)
    assert (ev[:5] > 0.03).all() and ev[2] > 0.1
    sev = np.array([s.severity for s in specs if s.event != 0])
    assert (sev > 0.8).mean() > 0.05 and (sev < 0.2).mean() > 0.05


GOLDEN = "636ea793d3c4d1005a8f98df992572364168269c6fbcff1ea6e36eb73458ac65"
