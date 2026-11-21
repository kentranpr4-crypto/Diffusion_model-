# Dataset specification (raresimdiff_v1)

Built by `scripts/build_dataset.sh` (procedural generator → validation → MuJoCo rollout with the scripted baseline).

| File | Content |
|---|---|
| `tensors_{train,val,test,ood_test}.npz` | model inputs + labels: `phys` (N,9,12 physical units), `slot_mask`, `asset_ids`, `task`, `event`, `robot`, `severity`, `env`, `success`, `term`, `labels`, ids |
| `scenarios_{split}.jsonl.gz` | full human-readable ScenarioSpec per scenario + validation + baseline label + provenance (simulator/robot versions, seeds) |
| `trajectories/traj_*.npz` | state (64-d), action (6-d), reward trajectories at 10 Hz for a stored fraction of episodes |
| `manifests/{split}.manifest.json` | scenario ids, parent ids, tensor hash per split |
| `preprocessing_stats.json` | normalisation mean/std computed **from the train split only** (source hash recorded) |
| `rejects.jsonl.gz` | every rejected scenario with reason codes |
| `dataset_manifest.json` | config, counts, reject statistics, throughput, SHA-256 of every file, dataset fingerprint |

Per-episode content: scenario id, RNG seed, simulator + robot version, physical parameters, event family and severity,
initial state (object poses/masses/frictions), event schedule, state/contact summaries, observation/action trajectories (baseline-controller
actions, **not expert demonstrations**), success/termination labels, validation results, provenance.

Variables, units and bounds: see `src/raresimdiff/scenarios/schema.py` (authoritative). Observation vector (28-d) and action vector (6-d,
`[vx, vy, wz, reach, lift, grip]` in [-1,1]) are documented in `simulation/controllers.py`.

Integrity (`data/integrity.py`, run by gate G3 and again by GPU preflight): file hashes, fingerprint, finite values, declared bounds,
masks/asset consistency, SO(3) validity, severity recomputation, duplicate detection (also at build time), **no parent shared across splits**,
no scenario id overlap, train-only statistics reproducible from the train split.

Splits are assigned by hashing the *parent scene* id (80/10/10). OOD parents (held-out corner recipes: very heavy low-friction targets with
heavy clutter; dense floor clutter) and all of their variants go to `ood_test` only.

Labels come from a scripted baseline policy; they describe *difficulty for that baseline*, not human difficulty.
No real-world data is involved: every sample is synthetic (`is_synthetic=True` in exported transitions).

## Dynamics vs perception-only perturbations
| Event type | changes physical dynamics | changes observations only |
|---|---|---|
| `object_slip` | friction multiplier, external force pulse (+ object mass/friction/pose, clutter) | – |
| `moving_obstacle` | obstacle motion (own actuators), size, timing | – |
| `perception_degradation` | – | sensor noise, occlusion dropout, visibility range, camera calibration error |
| `slip_and_perception` | friction multiplier, external force pulse | sensor noise, occlusion dropout, visibility range, camera calibration error |
| `nominal` | – | – |
Perception-only parameters leave the physical state trajectory of an idle robot unchanged (tested in `tests/physics`); they influence results only through the policy's observations.
No rendering is involved: "visual" degradation is modelled at the observation level (no cosmetic pixel effects).
