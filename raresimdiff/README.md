# RareSimDiff

Physics-grounded diffusion **scenario** generator for humanoid robot learning (research MVP).
One conditional diffusion model (`RareScenarioDiT`, 7.5 M parameters) learns the distribution of simulation-ready scenes and rare-event
parameters; samples are decoded, repaired, validated in MuJoCo and become executable episodes. The simulator stays the authority on physics.

> Status: CPU-validated pipeline + GPU bundle. **No trained model exists yet.** Humanoid validation is blocked (no humanoid model in the repo;
> a proxy mobile manipulator is used). See `docs/ARCHITECTURE.md`.

## Quick start (CPU)
```bash
pip install -e ".[dev,package]"
scripts/build_dataset.sh --out data/demo --n-parents 100 --variants 4 --workers 2     # small demo dataset
python -m pytest                                                                      # full test-suite
scripts/local_preflight.sh                                                            # gates G0..G5
scripts/package_for_gpu.sh                                                            # bundle + gate G6
```
GPU run: see `docs/GPU_DEPLOY.md` (`bash run_gpu_once.sh`).

Generation after training:
```bash
python -m raresimdiff.generate --checkpoint runs/<run_id>/best.safetensors --event-type object_slip \
  --severity 0.8 --num-scenarios 1000 --validate-physics --output generated/slip_cases
```

Layout: `src/raresimdiff/{scenarios,simulation,validation,data,diffusion,models,training,evaluation,policy_integration,utils}`,
`configs/`, `scripts/`, `tests/{unit,integration,physics,training,regression}`, `docs/`.
Docs: ARCHITECTURE · DATASET_SPEC · EXPERIMENT_PROTOCOL · GPU_DEPLOY · RESULTS_TEMPLATE.
