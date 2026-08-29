# Research Log — 2026-08-30

## Evaluation-only extreme-v2 Colab workflow

The Colab entry point was refactored from a mixed train-or-evaluate notebook into
an evaluation-only workflow for the existing H4rmony R1 Qwen3-8B adapter. It now
constructs the same `SFTConfig` training signature used by the completed rank-16,
three-epoch, seed-42 intervention and scans
`MyDrive/value-misalignment/harmony_r1_qwen3_8b/` for the newest compatible run.
A candidate is accepted only after the original SFT completion manifest and every
required artifact hash validate. If no compatible run exists, the notebook stops
with an explicit error; it never starts another training run.

The six-template `extreme_v2` catalog now has a single code-level definition in
`scripts/harmony_sft/extreme_v2_eval.py`. With the configured values 0, 1, 10,
100, 1,000, 10,000, 100,000, and 1,000,000, the workflow renders 48 exact cases.
It evaluates the pinned unmodified base checkpoint first and then the saved LoRA
adapter in the same process, with Qwen thinking disabled and complete `Yes` and
`No` sequences scored. Before a result can be reused or persisted, validation
requires the current template hashes and rendered prompts, the complete cost grid,
and exactly 96 unique raw-score rows: one `base` and one `aligned` row for every
case. Missing, duplicate, or mismatched case/model rows abort the workflow.

New evaluations are completed under local `/content` storage and copied beneath
the source SFT run's `posthoc_evaluations/` directory. The existing durability
protocol validates the local bundle, copies it to Drive, flushes and unmounts
Drive, freshly remounts it, and recomputes the artifact hashes through the new
mount. A compatible existing evaluation is reused only after both its completion
hashes and the stricter six-template score-matrix validation pass.

`scripts/harmony_sft/github_publish.py` adds an authenticated publication step for
the compact verified bundle: `rendered_cases.jsonl`, combined base/aligned
`raw_scores.csv`, `thresholds.csv`, `curves.png`, `metadata.json`, and
`COMPLETE.json`. The bundle is committed beneath
`results/harmony_eval/qwen3_8b_harmony_r1_sft/<source-run>/<evaluation>/` and
pushed to `main` without force. Publication requires a Colab `GITHUB_TOKEN` secret
with repository Contents read/write permission. Authentication uses a transient
askpass script and environment variable rather than modifying the remote URL or
repository configuration. The remote branch tip is read back after the push and
must equal the local result commit. Reruns reuse an existing GitHub path only when
its complete file set is byte-identical to the verified Drive bundle.

The notebook now preflights the GitHub secret before GPU inference, previews all
48 rendered prompts, runs one orchestration call, displays probability and raw-logit
tables for both model roles across every cost, displays thresholds and curves, and
then publishes the verified Drive result. JSON validation and combined code-cell
compilation pass. Static compilation passes for the new modules, and all 58 unit
tests pass, including simulated verified-checkpoint reuse, rejection of incomplete
base/aligned matrices, refusal to retrain when the checkpoint is absent, and an
idempotent push to a temporary bare Git remote. No live Colab GPU inference,
Google Drive remount, or GitHub credentialed push was performed locally; the next
step is to push this code revision and run the notebook on an A100 with the Drive
folder and Colab secret available.
