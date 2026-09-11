# Ecological DPO analysis — 2026-09-11

This is a descriptive analysis of one completed ecological-preferred Qwen3-8B
DPO run. The two held-out evaluations show no systematic ecological shift at
this dose. Training-score precision requires correction before treating the run
as a clean test of whether a learned preference generalizes.

See [the research log](../../../doc/2026-09-11-research-log.md#analysis-of-the-first-colab-dpo-run)
for interpretation and limitations. `comparison.png` plots the base and DPO
scores for all eight scenario families. `summary.json` contains aggregate
metrics; the CSVs retain family, cost, order, and online training-step detail.

## Reproduction

From the repository root, in an environment with pandas, numpy, and matplotlib:

```sh
python results/analysis/20260911_ecological_dpo/analyze.py
```

This validates the two published evaluation bundles, independently checks the
included source metadata/metrics against their completion manifest, verifies
their identity against both evaluations, checks historical prompt equality,
and regenerates all summaries and the plot. No weights or network are needed.
The run identity and immutable adapter hash are recorded in `summary.json`.
Choice results average both orders and the seven strictly positive costs;
zero-cost results are reported separately. Numerical results normalize within
each A-D mapping, average all 24 mappings, then average the eight scenarios.
There are no significance tests: these repeated readouts are not independent
experimental replications.

To reproduce the separate precision diagnostic, install the dependencies in
`requirements-colab-dpo.txt` in a compatible Python/PyTorch environment, then:

```sh
python results/analysis/20260911_ecological_dpo/precision_diagnostic.py
```

This uses the production trainer builder, a tiny random BF16 Qwen3, four toy
pairs, and Accelerate's actual FP32 output conversion. No training occurs.
It verifies identical raw-model/reference scores, then demonstrates nonzero
implicit reward margins after output conversion alone. The retained result
used PyTorch 2.8.0 on CPU with the exact pinned DPO libraries. It does not
estimate the magnitude or consequence of the error on the 8B Colab run.

## Source provenance

The compact evaluation bundles were pulled from GitHub at `1901516` and are
referenced by exact paths in `analyze.py`. The source files below were retrieved
from the user's completed run in Drive; their bytes are retained unchanged:

- [Run folder](https://drive.google.com/drive/folders/17kNtnpdKcbWCBNckmyxWvYHN98O97WUk)
- [run_metadata.json](https://drive.google.com/file/d/19hn6ESC8PhYdUNLnZ5bNR7nEmcumTcxD/view)
- [COMPLETE.json](https://drive.google.com/file/d/1j2sA4_4bt3ADXLUigB0Gh_W6AD82WvLX/view)
- [training/train_metrics.json](https://drive.google.com/file/d/1HoIRP9F6dDe7P-IkQCVl95o-lqxLLMiq/view)

The source completion SHA-256 is
`e5d2e41a608f3498747d24d6400dda1b90a2c039b1fcad0cecf4bad393e93828`.
It matches both evaluation metadata files. Only these small training-source
files were independently downloaded and verified for this analysis, not the
adapter weights or optimizer checkpoints. Token-valued fields in the original
trainer metadata are redacted placeholders produced by the trainer.
