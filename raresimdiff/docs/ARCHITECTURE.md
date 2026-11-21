# RareSimDiff architecture

RareSimDiff = **physics-grounded** diffusion *scenario* generator for humanoid robot learning. The simulator (MuJoCo)
is the only source of physical state, contacts, transitions and labels. The diffusion model never produces rendered
video or physical trajectories: it produces **executable ScenarioSpecs** that are validated and then simulated.

```
Procedural generator ─► MuJoCo physics ─► dataset + labels ─► RareScenarioDiT training (ONE official GPU run)
        │                                                          │
        └──── baseline comparison ◄── sampling (DDIM + CFG) ◄──────┘ frozen checkpoint
                                              │
              decode → constraint projection → geometry validation → MuJoCo settle + event dry-run → quality scoring
                                              │
                                  executable episodes → policy training / evaluation
```

## Implemented vs paper-inspired

| Component | Status | Notes |
|---|---|---|
| Procedural scenario generator, severity function, 3 scenario families | **Implemented** | `scenarios/` |
| MuJoCo backend, event injection, perception model, scripted baselines | **Implemented** | `simulation/` |
| Conditional denoising diffusion (cosine schedule, epsilon prediction, DDIM, classifier-free guidance, EMA) | **Implemented** (standard methods: Ho et al. 2020, Song et al. 2021, Ho & Salimans 2022, Nichol & Dhariwal 2021) | `diffusion/`, `models/` |
| DiT-style adaLN-Zero transformer over *scene tokens* with masks | **Implemented**, adapted from DiT (Peebles & Xie) | `models/dit.py` |
| Steerable Scene Generation (post-training + inference-time search) | **Paper-inspired only.** Not implemented. We use physics validation as a *filter* after sampling (a simple rejection form of inference-time search) | `validation/` |
| Scenario Dreamer (vectorised latent diffusion for driving scenes) | **Paper-inspired:** scenes as sets of typed tokens with masks. No latent autoencoder, no lane graph | `scenarios/schema.py` |
| Diffusion Policy | **Not implemented.** Only an adapter contract (`policy_integration/adapters.py`). The generator is not a policy | |
| UniCaMo (structured noise) | **Not used.** Out of scope for the scenario generator; relevant only to an optional future video renderer | |
| World-model planning | **Not implemented**; interface placeholder only | |

## Scenario representation

* 9 tokens × 12 channels: 1 global (event schedule, disturbance, sensor degradation), 6 object slots, 2 obstacle slots.
* **Continuous** variables have explicit units, bounds, `[-1,1]` bound-normalisation, then train-split mean/std; inverse
  transforms exist (`scenarios/codec.py`). Object orientation is the continuous **6D rotation representation**; decoding
  applies Gram-Schmidt → SO(3), then a tilt limit.
* **Categorical** variables (task, event type, robot profile, asset ids, slot masks) are *conditioning embeddings*, never diffused as
  continuous physics. When no layout is supplied, layouts are bootstrapped from the **train split** (`scenarios/prior.py`):
  generated scenes therefore have novel continuous parameters but *re-used discrete layouts* (documented limitation).
* Frame: right-handed, +x forward, +z up, origin at the robot start pose (robot-relative).
* `severity ∈ [0,1]` is a deterministic function of the physical parameters (`scenarios/severity.py`), so conditioning
  accuracy is measurable on any generated sample.

## Model: `RareScenarioDiT`

hidden 256, 6 layers, 8 heads, MLP ratio 4, adaLN-Zero conditioning (condition + timestep), type-specific input/output
projections, learned slot embeddings, asset-id embeddings on object slots, key-padding mask for absent slots,
**7,524,382 parameters** (measured). Training: epsilon-prediction MSE masked to active slots/channels, 1000 steps, cosine
schedule, 10 % condition dropout (task/event/robot/severity/environment → learned null embeddings; slot masks and asset ids are
structural and never dropped). Inference: DDIM (20–50 steps), classifier-free guidance, predicted x0 clipped to the bound-implied range.

## Simulator and the proxy robot

No humanoid model exists in this repository. The MVP uses an explicit **proxy mobile manipulator** (`ROBOT_VERSION =
proxy_mobile_manipulator_v0`: planar base + reach + lift + two-finger gripper). **Humanoid validation is BLOCKED** until a real
humanoid MJCF/URDF is supplied. The `SimulatorBackend` ABC (`simulation/base.py`) is the seam for a humanoid MuJoCo model or Isaac Sim / Isaac Lab.

Events are injected by `EventInjector` (separate from control): friction change at onset, an external force pulse, obstacle motion
through the obstacle's own servo actuators (no teleporting). Perception degradation changes **observations only**
(`perception.py`); it is therefore distinguished from dynamics-changing parameters.

## Validation pipeline (generated scenarios)

decode → constraint projection (bounds, SO(3), tilt, overlap separation, inactive groups → nominal; repair magnitude recorded) →
geometry validation (reason codes) → MuJoCo settle + penetration + stability + idle-robot event dry-run → scoring. Collisions between
robot and obstacles are *not* rejected (they are part of the rare event); initial penetrations and unstable simulations are.

## Training system

Single run, config-frozen (`configs/gpu_4090.yaml`), run id derived from the training-config hash and dataset fingerprint, immutable run
directory, atomic checkpoints (model, EMA, AdamW, LR scheduler, step, RNG states, data-order is a pure function of (seed, step)),
exact deterministic resume (verified bitwise on CPU), NaN/Inf guards, best-by-validation-EMA selection, JSONL/CSV logs,
safetensors export (self-contained: weights + normaliser + categorical prior + training-support ranges).
