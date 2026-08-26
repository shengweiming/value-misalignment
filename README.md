# Value Misalignment

This repository contains exploratory evaluations of whether ecological alignment
training shifts a model's willingness to impose welfare and autonomy costs for a
fixed ecological benefit.

## H4rmony R1-only Qwen3 SFT

The Colab workflow in `notebooks/harmony_checkpoint_eval_colab.ipynb` fine-tunes
`Qwen/Qwen3-8B` on the environmentally aligned R1 answers in
`neovalle/H4rmony`. It groups the pairwise source data by `PromptID`, takes R1
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

The notebook contains only Colab setup, Drive mounting, configuration, one training
call, and result display. Reusable code lives in `scripts/harmony_sft/`:

- `data.py` constructs and validates the R1-only SFT examples.
- `tokenization.py` creates non-thinking chats and response-only labels.
- `runner.py` performs base evaluation, LoRA training, adapter saving, aligned
  evaluation, and artifact validation.

Every run is written directly beneath
`MyDrive/value-misalignment/harmony_r1_qwen3_8b/` and contains:

- `checkpoints/`: epoch checkpoints, including resumable trainer state.
- `final_adapter/`: the final LoRA adapter and tokenizer files.
- `dataset/`: the exact selected R1 examples and a source/token-length manifest.
- `training/`: final metrics and training logs.
- `evaluation/`: base and aligned Yes/No scores, threshold comparisons, and curves.
- `run_metadata.json`: pinned revisions, configuration, hardware, package versions,
  prompt hashes, and repository commit.
- `COMPLETE.json`: hashes of required artifacts, written only after validation; a
  failed run instead writes `FAILED.json`.

The adapter does not duplicate the Qwen3-8B base weights. Reload it on top of the
base-model revision recorded in `run_metadata.json`.

## Ecological threshold evaluation

The SFT runner evaluates the unchanged base and the in-memory trained adapter on
the scenario families under `eval/ecological_value/`:

- `wetland_relocation.txt`
- `invasive_animal_killing.txt`
- `ecosystem_restoration_wild_animal_suffering.txt`
- `habitat_protection_livelihood_restriction.txt`

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

These four single-template families are an exploratory screen, not a confirmatory
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
requests with both question polarities. The wetland template uses `{family_count}`;
the other ecological-value templates use `{cost_count}`. Both are populated from the
same configured sweep.

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

## Tests

The local tests do not download models, call a provider, or require an API key:

```bash
python -m unittest discover -s tests -v
```
