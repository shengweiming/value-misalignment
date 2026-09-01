# Value Misalignment

This repository contains exploratory evaluations of whether ecological alignment
training shifts a model's willingness to impose welfare and autonomy costs for a
fixed ecological benefit.

## H4rmony R1-only Qwen3 evaluation

The Colab workflow in `notebooks/harmony_checkpoint_eval_colab.ipynb` is
evaluation-only. It finds the newest hash-verified H4rmony R1 LoRA run already in
Google Drive, loads its recorded immutable `Qwen/Qwen3-8B` base revision, and
scores the base and saved adapter on all eight primary `extreme_v2` templates and
all six controls. It refuses to start training when no compatible completed run
is present.

The saved intervention was fine-tuned on the environmentally aligned R1 answers
in `neovalle/H4rmony`. The data loader groups the pairwise source data by
`PromptID`, takes R1
from `BetterCompletion` in the R1-R2 and R1-R3 rows, and produces one
prompt-to-R1 example per prompt ID. When those two rows disagree, the loader applies
documented corrections for known source defects, then uses rank assignments across
all three pair rows to resolve any other conflict; if the evidence is tied, R1-R2 is
the canonical source. Every disagreement and resolution is saved in the dataset
manifest. R2-R3 answers are used only as consistency evidence, never as SFT targets.

The default intervention is BF16 LoRA on an A100 with rank 16, alpha 32, dropout
0.05, three epochs, micro-batch size 1, gradient accumulation 16, and seed 42.
Qwen thinking is disabled, and prompt tokens are masked so loss is applied only to
the R1 answer and its end token. At run time, both the model and dataset revisions
are resolved to immutable Hugging Face commit hashes.

The setup removes Colab's optional preinstalled `torchao` package because the
version currently supplied by Colab is incompatible with PEFT. This workflow uses
BF16 LoRA and does not use TorchAO quantization.

The notebook contains Colab setup, Drive mounting, a visible checkpoint signature
and evaluation configuration, exact primary/control prompt previews, separate
durable workflow calls, result displays, and GitHub publication. Reusable code lives in
`scripts/harmony_sft/`:

- `data.py` constructs and validates the R1-only SFT examples.
- `tokenization.py` creates non-thinking chats and response-only labels.
- `runner.py` performs base evaluation, LoRA training, adapter saving, aligned
  evaluation, and artifact validation.
- `persistence.py` copies the complete local run to Drive, forces outstanding
  writes to flush, remounts Drive, and verifies recorded hashes from the fresh
  mount before reporting success.
- `extreme_v2_eval.py` owns the eight-template primary catalog and six-template
  control catalog, finds the compatible SFT
  run, checks the complete base/aligned score matrix, and orchestrates durable
  evaluation reuse or execution.
- `github_publish.py` copies only the verified compact result bundle into
  `results/harmony_eval/`, commits it, pushes without force, and reads the remote
  branch tip back to verify publication.

Every run is first completed under local `/content` storage. It is then copied
beneath `MyDrive/value-misalignment/harmony_r1_qwen3_8b/`; the local run is retained
for recovery until the Colab runtime is disconnected. A Drive run contains:

- `checkpoints/`: epoch checkpoints, including resumable trainer state.
- `final_adapter/`: the final LoRA adapter and tokenizer files.
- `dataset/`: the exact selected R1 examples and a source/token-length manifest.
- `training/`: final metrics and training logs.
- `evaluation/`: base and aligned Yes/No scores, threshold comparisons, and curves.
- `run_metadata.json`: pinned revisions, configuration, hardware, package versions,
  prompt hashes, and repository commit.
- `COMPLETE.json`: hashes of required artifacts, written after local validation and
  checked again after Drive is flushed and freshly remounted; a failed training run
  instead writes `FAILED.json` locally.

For the default eight-value cost grid, the primary extreme-v2 evaluation produces
64 rendered cases and 128 raw-score rows: exactly one base and one aligned row for
every template-by-cost case. The notebook reports completion only after the Drive
copy passes a fresh-mount hash check. If persistence fails, it prints the intact
local result path and refuses to claim completion.

Six additional controls live under `eval/ecological_value/extreme_v2/control/`,
with two prompts each for matched non-ecological policies, unrelated severe moral
dilemmas, and zero-cost ecological policies. The first four controls use the full
cost grid. The two zero-cost controls contain no `{cost}` placeholder and render
once each with `cost_count=0`. The notebook evaluates them as a separate 34-case
bundle with 68 raw-score rows, rather than mixing them into the primary
eight-prompt result.

GitHub publication is enabled by default and requires a Colab secret named
`GITHUB_TOKEN` with Contents read/write permission for this repository. Results
are stored under
`results/harmony_eval/qwen3_8b_harmony_r1_sft/<source-run>/<evaluation>/`.
Authentication uses a transient askpass helper; the token is not printed, placed
in the remote URL, or written into repository state. An existing result path is
reused only when every published file is byte-identical.

An adapter saved before a template update can be evaluated without another SFT
run. `scripts/harmony_sft/posthoc_eval.py` reloads the pinned base and saved adapter,
scores the repository's current template catalog, and creates a separate local
evaluation bundle. Its persistence helper copies that bundle beneath the source
run's `posthoc_evaluations/` directory and applies the same flush, remount, retry,
and hash-verification protocol. Post-hoc evaluation never modifies the original
run's completion manifest.

The adapter does not duplicate the Qwen3-8B base weights. Reload it on top of the
base-model revision recorded in `run_metadata.json`.

## Ecological threshold evaluation

The SFT runner evaluates the unchanged base and the in-memory trained adapter on
four scenario families under `eval/ecological_value/`, each with paired `mild/`
and `extreme/` versions:

- `wetland_relocation.txt`
- `invasive_animal_killing.txt`
- `ecosystem_restoration_wild_animal_suffering.txt`
- `habitat_protection_livelihood_restriction.txt`

The mild files are the original evaluation prompts. Extreme counterparts hold the
stated ecological benefit fixed while replacing the per-unit cost with lethal or
prolonged severe harm. The default evaluation runs all eight templates. Explicit
selectors use paths such as `extreme/wetland_relocation`; an unqualified legacy
selector such as `wetland_relocation` continues to select the mild version.

Six targeted prompts live under `extreme_v2/`: marine-reserve, pesticide-ban,
predator-reintroduction, wetland-preservation, oil-extraction-ban, and
vegan/meat-eater trolley scenarios. They are selected explicitly, so adding an
exploratory item does not silently change the original paired screen.
Every ecological-threshold template is self-contained: it uses exactly `{cost}`,
`{yes_label}`, and `{no_label}`, and includes its own decision question. New
post-hoc bundles write `rendered_cases.jsonl` before inference, providing one
deduplicated record for every exact prompt sent to each checkpoint.

Each template contains a configurable cost count. The evaluator sweeps the count,
scores the complete `Yes` and `No` response sequences, and estimates the point where
the model's normalized probability of implementing the ecological policy crosses
0.5. Shared evaluation implementation lives in `scripts/harmony_eval/`:

- `catalog.py` defines the three released matched checkpoint pairs.
- `cases.py` renders direct-question evaluation cases.
- `scoring.py` performs local seq2seq or causal-LM sequence scoring.
- `analysis.py` fits monotone sacrifice thresholds and plots response curves.
- `runner.py` retains support for the earlier released-checkpoint comparisons.

Imported completed runs are stored under `results/harmony_eval/`. The analysis of
the earlier Flan-T5 Caramel comparison is recorded in
`doc/2026-08-25-research-log.md`.

These four paired-template families are an exploratory screen, not a confirmatory
radicalization result. A confirmatory evaluation needs held-out paraphrases, more
independent items per family, and uncertainty intervals across items.

## Hosted Stage 1 runner

The older runner supports `deepseek-v4-flash` and `qwen3-8b` in non-thinking mode.
It scores `Yes` against `No` for each configured cost count, either with the direct
implementation question alone or paired with a reversed rejection question.

Relevant files:

- `configs/stage_1.toml` controls the wetland-relocation experiment.
- `configs/models/qwen3_8b.toml` and `qwen3_8b_beijing.toml` select hosted Qwen.
- `scripts/run_stage_1.py` contains the hosted experiment and API logic.
- `constitutions/` retains the ecological-value instructions used in earlier pilots.
- `doc/project-context.md` summarizes the radicalization hypothesis and measurement
  logic.
- `doc/2026-08-24-research-log.md` records earlier experiments and proposed follow-up.

The default coarse sweep is
`0, 1, 10, 100, 1000, 5000, 10000, 100000, 500000, 1000000`, giving 20
requests with both question polarities. This legacy polarity experiment uses its
own `eval/stage_1/wetland_relocation.txt` template; it is intentionally separate
from the self-contained `{cost}` templates used by the local threshold evaluator.

### Local setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set provider keys in the shell rather than in this repository:

```bash
export DEEPSEEK_API_KEY="your-key-here"
export DASHSCOPE_API_KEY="your-model-studio-key"
```

Inspect all rendered hosted requests without making API calls:

```bash
python scripts/run_stage_1.py --config configs/stage_1.toml --dry-run
```

Run the hosted experiment:

```bash
python scripts/run_stage_1.py --config configs/stage_1.toml
```

Select hosted Qwen without changing the experiment condition:

```bash
python scripts/run_stage_1.py \
  --config configs/stage_1.toml \
  --model-profile configs/models/qwen3_8b.toml
```

Qwen uses DashScope's native generation endpoint because its OpenAI-compatible
endpoint does not expose the needed token log probabilities. The runner stops if
either response label is absent rather than substituting a sampling estimate.

## Ecological dilemma generation

The repository includes a quality-controlled generator for creating original,
moderate-cost ecological-versus-human dilemmas. Its source materials are kept in
`src/ecological_dilemmas/`:

- `ecological_dilemma_constructs.json` contains the supplied ecological objects,
  human interests, policy mechanisms, and their definitions.
- `decision_makers.json` supplies the fourth sampled construct.
- `generator_prompt.txt` preserves the original one-shot generation prompt.
- `scenario_card_prompt.txt`, `card_review_prompt.txt`,
  `dilemma_writer_prompt.txt`, and `dilemma_validation_prompt.txt` implement the
  staged quality-control workflow.

The sampler balances each construct dimension across accepted cases. For each
assignment, GPT-5.6 Sol plans three materially different scenario cards. A
separate Sol call selects and repairs the strongest card or rejects the construct
combination for resampling. GPT-5.6 Terra writes the final prose, and Sol then
validates it against both the assignment and approved card. A validator revision
must pass a second validation call before it is accepted. The default acceptance
threshold is 4 out of 5 on every criterion; a run may attempt up to three times
as many construct combinations as the requested final count. Sol uses low
reasoning effort by default. Each stage gets up to three response tries; malformed,
empty, or incomplete responses are saved and charged to the run before retrying.
If all three tries fail, that construct combination is rejected and sampling
continues.

Create the local environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Add the API key to the repository's ignored `.env` file:

```dotenv
OPENAI_API_KEY=your-key-here
OPENAI_PLANNER_MODEL=gpt-5.6-sol
OPENAI_REVIEWER_MODEL=gpt-5.6-sol
OPENAI_WRITER_MODEL=gpt-5.6-terra
OPENAI_VALIDATOR_MODEL=gpt-5.6-sol
OPENAI_PIPELINE_REASONING_EFFORT=low
OPENAI_WRITER_REASONING_EFFORT=medium
OPENAI_STAGE_RETRIES=3
```

Preview ten balanced assignments without calling the API:

```bash
python scripts/generate_ecological_dilemmas.py --dry-run
```

Generate and quality-control ten dilemmas:

```bash
python scripts/generate_ecological_dilemmas.py
```

Generate 400 accepted dilemmas with a reproducible sampling seed:

```bash
python scripts/generate_ecological_dilemmas.py \
  --count 400 \
  --seed 42
```

Continue a dataset in a fresh run without reusing accepted construct assignments.
Repeat `--prior-run` for every earlier batch that should contribute to the balance
and novelty baseline:

```bash
python scripts/generate_ecological_dilemmas.py \
  --count 40 \
  --seed 43 \
  --novelty-window 100 \
  --prior-run outputs/ecological_dilemmas/<first-run> \
  --prior-run outputs/ecological_dilemmas/<second-run>
```

Prior runs must be complete and use the same construct and decision-maker source
files. Their accepted assignments are removed from the new sampler, their counts
seed its marginal and pairwise balance, and their novelty signatures are supplied
to planning and review. The new manifest records the paths and hashes of all prior
runs. The seed remains reproducible, but no longer restarts the unconditioned
assignment sequence because the prior assignments have been excluded.
This guarantees exact four-construct assignment uniqueness, not semantic scenario
uniqueness; the novelty-signature review remains a model-based quality check.

If a run is interrupted, resume it from its timestamped directory; the recorded
models, thresholds, source hashes, and seed are reused and verified:

```bash
python scripts/generate_ecological_dilemmas.py \
  --resume outputs/ecological_dilemmas/<run-directory>
```

Every invocation creates a timestamped directory under
`outputs/ecological_dilemmas/`. It contains one `.txt` and one `.json` record per
accepted dilemma, combined `records.jsonl`, all accepted and rejected attempts in
`attempts.jsonl` and `attempts/`, and a progress manifest. The manifest records
source hashes, stage-specific response IDs and token usage, rejection counts, and
an estimated standard-priority API cost using a dated pricing snapshot. Use
`--output-dir` to select another parent directory. Every stage model, reasoning
effort, output-token ceiling, acceptance threshold, retry limit, and prompt path
is configurable; run `--help` for all options. Per-attempt files retain every raw
response, response status, incomplete-response details, parsing or validation
error, and successful parsed output, including responses consumed by retries.

## Ecological-dilemma fine-tuning arms

The 100 candidates produced by the three completed quality-pipeline runs have a
reproducible audit and release step:

```bash
python scripts/build_ecological_sft_dataset.py
```

The builder verifies the completed source manifests, accepted records, approved
card fields, final format, and final validator decisions and scores. It computes
cross-run TF-IDF similarity for every pair and refuses to build until every pair
above the configured threshold has an explicit duplicate-or-distinct judgment in
`src/ecological_dilemmas/sft_audit_decisions.json`. The same file contains the
five evidence-preserving prose repairs and two documented duplicate exclusions.

The versioned output is committed under `data/ecological_dilemmas/v1/`. It contains
98 prompt-only dilemmas after five repairs and two duplicate exclusions. There is
no train/held-out split, answer label, assistant response, rationale, or normative
adjudication. The release manifest states those absences explicitly and pins the
source-run, decision-file, and artifact hashes. The audit and semantic-review files
retain every keep, repair, exclusion, and pairwise judgment.
The compact source evidence needed to reproduce the build is committed under
`data/ecological_dilemmas/source_runs/`; its manifests pin the corresponding full
raw-run hashes. If the ignored raw generation directories are available locally,
refresh that snapshot and rebuild with:

```bash
python scripts/build_ecological_sft_dataset.py \
  --refresh-sources-from outputs/ecological_dilemmas
```

### CLASH prompt-only training control

The first non-ecological training control is committed under
`data/control_dilemmas/clash/v1/`. It contains 98 exact CLASH `situation` texts at
or below 320 whitespace-separated words after conservative lexical and manual
ecological-content screening. The release takes the 98 longest of 101 surviving
candidates, yielding a 231.031-word mean. It contains no released action,
rationale, character perspective, assistant response, or answer label.

Rebuild the release from the pinned source snapshot with:

```bash
python3 scripts/build_clash_prompt_control_sft_dataset.py
```

Its `id`, `dilemma`, `title`, and source-provenance schema is accepted by the
same prompt-only loader used for the ecological arm. Training therefore renders
each `dilemma` as one non-thinking user message, adds no generation prompt, and
applies causal-LM loss to every non-padding token.

`notebooks/clash_prompt_control_sft_colab.ipynb` runs the matched Qwen3-8B
control experiment. Before loading model weights, it uses the real pinned Qwen
tokenizer to verify all 98 examples: each raw dilemma is the content of one
`user` message, no literal `User:` prefix is inserted, thinking and the assistant
generation prefix are disabled, and every rendered input token is copied into
`labels`. The notebook reuses or trains only a hash-compatible CLASH checkpoint,
persists it under an isolated local and Drive root, and publishes results under
`qwen3_8b_clash_prompt_control_sft`.

The notebook runs the 64-case ecological `extreme_v2` primary evaluation and the
320-case supervision-matched readout battery. It does not build or run the
separate six-template non-ecological control-evaluation suite. Both completed
evaluation bundles are saved beneath the source Drive run, freshly rehashed, and
published to the pair-specific GitHub results directory.

Two deterministic answer-supervised datasets are derived from the unchanged
ecological release described above:

```bash
python3 scripts/build_ecological_answer_sft_datasets.py
```

They are committed under `data/ecological_dilemmas/sft/`. In the
`ecological_option` arm, the assistant emits the exact
`ecologically_protective_option` field. In the `human_option` arm, it emits the
exact `human_protective_option` field. Each contains 98 one-user/one-assistant
chats. Neither contains a rationale or any generated explanatory prose.

`notebooks/ecological_dilemma_prompt_sft_colab.ipynb` selects among
`prompt_only`, `ecological_option`, and `human_option` with one configuration
variable; it currently defaults to `ecological_option`. The original prompt-only
arm still renders each dilemma as one non-thinking user message and applies loss
to every non-padding user-turn token. The two answer arms mask the user turn and
generation prefix, applying loss only to the exact assistant option and its EOS
token; the masked dilemma remains the assistant response's complete conditioning
context. Their training-objective identifiers end in `_v2`, so checkpoints from
the earlier full-sequence-label implementation cannot be silently reused. All
three arms refuse to truncate a dilemma.

The Colab setup otherwise retains the transferable H4rmony configuration: the
same immutable Qwen3-8B revision, BF16 LoRA on all linear layers, rank 16, alpha
32, dropout 0.05, three epochs, learning rate `1e-4`, micro-batch size 1, gradient
accumulation 16, maximum length 1,024, and seed 42. It refuses to truncate a
dilemma. Completed epoch checkpoints, the final adapter, the exact 98 training
examples, metrics, and hashes are written locally, copied to an arm-specific
Google Drive directory, and reverified after a flush and fresh remount.

The notebook then compares the saved adapter with the unchanged base model on all
64 primary `extreme_v2` cases and the 320-case supervision-matched readout
battery. The six-template non-ecological controls remain in commented legacy
cells and are not run by default. Verified readout bundles are published beneath
an arm-specific
`results/harmony_eval/qwen3_8b_ecological_dilemma_*_sft/<source-run>/` directory.
GitHub publication is enabled by default and requires a Colab `GITHUB_TOKEN`
secret with Contents read/write permission.

## Tests

The local tests do not download models, call a provider, or require an API key:

```bash
python -m unittest discover -s tests -v
```
