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

The repository includes a standalone generator for creating original,
moderate-cost ecological-versus-human dilemmas. Its source materials are kept in
`src/ecological_dilemmas/`:

- `ecological_dilemma_constructs.json` contains the supplied ecological objects,
  human interests, policy mechanisms, and their definitions.
- `decision_makers.json` supplies the fourth independently sampled construct.
- `generator_prompt.txt` is the generation prompt used for every request.

Create the local environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Add the API key to the repository's ignored `.env` file:

```dotenv
OPENAI_API_KEY=your-key-here
OPENAI_MODEL=gpt-5.6-terra
OPENAI_REASONING_EFFORT=medium
```

Preview ten sampled assignments and fully rendered prompts without calling the
API:

```bash
python scripts/generate_ecological_dilemmas.py --dry-run
```

Generate the default ten completions with GPT-5.6 Terra:

```bash
python scripts/generate_ecological_dilemmas.py
```

The count and model are command-line options:

```bash
python scripts/generate_ecological_dilemmas.py \
  --count 100 \
  --model gpt-5.6-terra \
  --seed 42
```

Every invocation creates an immutable timestamped directory under
`outputs/ecological_dilemmas/`. It contains one `.txt` completion and one `.json`
record per dilemma, a combined `records.jsonl`, and a `manifest.json` recording
the sampling seed, model, source hashes, progress, and token usage. Use
`--output-dir` to select another parent directory. Run `--help` for all options.

## Tests

The local tests do not download models, call a provider, or require an API key:

```bash
python -m unittest discover -s tests -v
```
