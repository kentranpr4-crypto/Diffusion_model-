# Run approval sheet (generated; approve BEFORE running `bash run_gpu_once.sh`)

| item | value |
|---|---|
| dataset | `raresimdiff_v1` fingerprint `b570009f46e8fb0d5ab341c711d21e2ca8bfe7930ab23754ad11288bc5c48cf1` |
| scenarios (train / val / test / ood_test) | 73295 / 9325 / 9016 / 8081 |
| training-config hash | `2236948fc650ed47f0d0aded610c5f0787a6cfd96a96ad4deaa03960c7e695c0` |
| model | RareScenarioDiT hidden 256, 6 layers, 8 heads |
| optimiser | AdamW lr 0.0003, wd 0.01, warmup 1000, cosine to 0.05 |
| batch | 512 x 2 accumulation = 1024 effective |
| budget | 30000 optimizer steps (~419 epochs over train) |
| precision | bfloat16 if supported, else float32 |
| checkpoints | every 2000 steps (keep last 3) + best-by-validation-EMA |
| evaluation after training | 200 scenarios x 4 events x 4 severities, learned vs procedural |
| number of training runs | ONE (a second independent run needs new approval) |

Wall-clock time and GPU memory on the 4090 are NOT known yet; the preflight probe reports the measured memory (see GPU_DEPLOY.md).
