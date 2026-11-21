# Results (fill after the official run; do not invent numbers)

Run id: `official_<hash>`  ·  dataset fingerprint: `...`  ·  training hash: `...`  ·  GPU: `...` (from `RUN_MANIFEST.json`)

## Training
| item | value (source file) |
|---|---|
| steps completed / best step | (`REPORT.json`) |
| best validation EMA loss | |
| skipped (non-finite) steps | |
| measured peak GPU memory | (`logs/train.jsonl`, `preflight/gpu_preflight.json`) |
| wall time | |

## Generator evaluation (`evaluation/evaluation_report.md`)
| metric | learned | procedural |
|---|---|---|
| valid ratio | | |
| event accuracy | | |
| severity MAE | | |
| diversity | | |
| rare-cell coverage | | |
| novelty (NN median) / memorisation rate | | |
| sim instability rate | | |
| baseline failure rate | | |

## Downstream study (EXPERIMENT_PROTOCOL.md §B)
| arm | completion | failure by family | violations | recovery | nominal regression |
|---|---|---|---|---|---|

## Conclusion
(State only what the data show. Mention limitations: proxy robot, scripted-baseline labels, bootstrapped layouts, no real-world evidence.)
