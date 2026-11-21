# GPU deployment: one official run on an RTX 4090 (24 GB)

Do **not** run anything below until you have approved the exact dataset, configuration and compute budget printed in `docs/RUN_APPROVAL.md`
(bundled) / the task report. Nothing here starts automatically.

Placeholders: `USER`, `GPU_HOST`. All paths are relative; nothing assumes a particular OS, CUDA runtime or Python version
(the launcher probes them). The NVIDIA driver is never modified.

## 1. Obtain the bundle (choose ONE)

**A. From the GitHub branch** (the bundle is committed at `raresimdiff/dist/`; authenticate first if the repository is private, e.g. `gh auth login`
or a credential helper; never write tokens into files or scripts):
```bash
git clone --depth 1 --branch claude/sleepy-shannon-duhe8f https://github.com/kentranpr4-crypto/Diffusion_model-.git rsd_release
```
```bash
cd rsd_release/raresimdiff/dist && sha256sum -c raresimdiff_gpu_bundle.sha256 && cp raresimdiff_gpu_bundle.tar.zst raresimdiff_gpu_bundle.sha256 ~/ && cd ~
```

**B. From the preparation machine with scp:**
```bash
scp dist/raresimdiff_gpu_bundle.tar.zst dist/raresimdiff_gpu_bundle.sha256 USER@GPU_HOST:~/
```

Also compare the SHA-256 printed by `sha256sum -c` with the value in the release report (the checksum file travels with the bundle, so an out-of-band
comparison is what protects against a tampered pair).

## 2. Verify and extract (on the GPU machine)
```bash
cd ~
sha256sum -c raresimdiff_gpu_bundle.sha256
tar --zstd -xf raresimdiff_gpu_bundle.tar.zst        # needs zstd; alternative: zstd -dc raresimdiff_gpu_bundle.tar.zst | tar -x
cd raresimdiff
```
If neither works: `python3 -m pip install --user zstandard` and
`python3 -c "import zstandard,tarfile,sys; tarfile.open(fileobj=zstandard.ZstdDecompressor().stream_reader(open('raresimdiff_gpu_bundle.tar.zst','rb')), mode='r|').extractall('.')"` (run from `~`).
After extraction `run_gpu_once.sh` re-verifies every file against `MANIFEST.sha256` and aborts (exit 1) on any mismatch.

## 3. Preflight only (no optimizer steps) - exactly one command
```bash
bash run_gpu_once.sh --preflight-only
```
Checks: bundle manifest hashes, Python/torch/CUDA, GPU model and VRAM, bf16, free disk, dataset integrity (file hashes, split leakage, train-only
statistics), config validity, and a forward/backward **memory probe at the configured batch size with NO optimizer step**. The measured peak memory is written to
`preflight/gpu_preflight.json`. Exit code 2 = a check failed; nothing was trained.

## 4. The single official run - exactly one command
```bash
tmux new -s rsd 'bash run_gpu_once.sh 2>&1 | tee run_gpu_once.log'      # or: nohup bash run_gpu_once.sh > run_gpu_once.log 2>&1 &
```
Order of operations: integrity → environment → preflight → immutable run directory `runs/official_<hash>/` → training with checkpoints →
evaluation (learned vs procedural) → export → `REPORT.json`. A completed run is **never** overwritten (exit code 3).

## 5. Monitoring
```bash
tail -f runs/official_*/logs/train.jsonl
nvidia-smi -l 5
cat runs/official_*/RUN_STATE.json
```

## 6. Interruption and resume (same run, not a new experiment) - exactly one command
SIGINT/SIGTERM (Ctrl-C, `kill`) saves `checkpoints/last.pt` and sets status `interrupted`. A hard crash loses at most `ckpt_every` (=2000) steps.
```bash
bash run_gpu_once.sh --resume
```
Resume restores model, EMA, AdamW, LR schedule, step counter and RNG states; the data order is a pure function of (seed, step). It refuses if the
configuration or dataset identity changed (that would be a new experiment and needs new approval) or if the run is already completed.
Exit codes: 0 ok · 1 package check failed · 2 preflight failed · 3 run exists/completed · 4 non-finite training (checkpoint kept; **not** retried automatically) · 130 interrupted.

## 7. Retrieve results (on the preparation machine)
```bash
rsync -av --exclude 'checkpoints/step_*' USER@GPU_HOST:~/raresimdiff/runs/ ./runs/
rsync -av USER@GPU_HOST:~/raresimdiff/preflight/ ./preflight/
```
Key artefacts: `runs/official_*/export/best.safetensors` (inference-ready, no optimizer state), `evaluation/evaluation_report.{json,md}`,
`logs/train.{jsonl,csv}`, `RUN_MANIFEST.json` (config, dataset hash, code revision, dependency versions, GPU info), `REPORT.json`.

## 8. After training: generation
```bash
bash scripts/generate_scenarios.sh --checkpoint runs/official_*/best.safetensors --event-type object_slip \
   --severity 0.8 --num-scenarios 1000 --validate-physics --output generated/slip_cases
```

## Resource estimates (ESTIMATED, not measured on a GPU)
Arithmetic from the architecture (`RareScenarioDiT`, 7,524,382 parameters, batch 512 per micro-batch ×2 accumulation, 9 tokens × hidden 256, bf16 autocast):

| item | ESTIMATED size |
|---|---|
| weights + gradients + EMA (fp32) | ≈ 3 × 30 MB |
| AdamW states | ≈ 60 MB |
| activations (6 layers, batch 512, bf16) | ≈ 0.3–0.6 GB |
| dataset tensors resident on GPU (~100k scenarios) | ≈ 0.1 GB |
| CUDA context / allocator slack | ≈ 0.5–1.5 GB |
| **total** | **≈ 1–3 GB** (far below 24 GB) |

The real value is measured by the preflight probe (`measured_peak_alloc_gb`) and logged during training (`gpu_mem_peak_gb`). Wall-clock time on the 4090 is
**unknown** until measured: the only measurement available during preparation is a 4-core CPU at ≈0.5 s per forward+backward micro-batch of 512.
Configuration: effective global batch = 512 (micro-batch) × 2 (gradient accumulation) = **1024**. 30,000 optimizer steps × 1024 = 30,720,000 samples
over a train split of 73,295 scenarios = **419.1 epochs by samples**. The trainer drops the last incomplete micro-batch of each epoch
(143 micro-batches/epoch, 2 per optimizer step), which gives **419.6 epochs** by its own accounting (60,000 micro-batches / 143).
