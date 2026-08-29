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

## Eight-prompt primary suite and controls

The first six-prompt extreme-v2 run showed the intended separation on marine
reserve, oil-extraction ban, pesticide ban, and wetland restoration: the base
model's implementation probability remained near zero while the aligned model's
was consistently higher. Predator reintroduction and the vegan/meat-eater trolley
instead produced a crossover, with the base model more willing below roughly
`N=100` and the aligned model more willing thereafter. Those two prompts were
removed from the primary suite.

Four new cost-varying ecological prompts—dam removal, wildfire restoration, river
water allocation, and island biosecurity—bring the primary suite to eight. The
default eight-value cost grid now creates 64 cases per model and 128 raw-score
rows. The Colab workflow, validation language, tests, and README were updated to
match; prior six-prompt result bundles remain preserved but are incompatible with
the new prompt manifest and will not be reused.

Six controls were added under `extreme_v2/control/`: two matched non-ecological
policies, two unrelated severe moral dilemmas, and two zero-cost ecological
policies. The zero-cost templates intentionally omit `{cost}`. Template rendering
now accepts that fixed schema and emits each such control exactly once at
`cost_count=0`, while cost-varying controls continue to use the requested grid.
The controls are separately addressable and are not silently included in the
primary Colab run. All 60 unit tests pass, the notebook remains valid JSON, its
combined code cells compile, and all 98 rendered primary/control cases contain no
unresolved placeholders.
