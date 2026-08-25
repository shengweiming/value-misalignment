# Value Misalignment

This repository contains exploratory evaluations of whether ecological alignment
training shifts a model's willingness to impose welfare and autonomy costs for a
fixed ecological benefit.

## H4rmony checkpoint evaluation

The quickest pilot compares released H4rmony-aligned checkpoints with their stated
base models on the evaluation families under `eval/ecological_value/`:

- `wetland_relocation.txt`
- `invasive_animal_killing.txt`
- `ecosystem_restoration_wild_animal_suffering.txt`
- `habitat_protection_livelihood_restriction.txt`

Each template contains a configurable cost count. The evaluator sweeps the count,
scores the complete `Yes` and `No` response sequences, and estimates the point where
the model's normalized probability of implementing the ecological policy crosses
0.5. It loads the base and aligned model one at a time, resolves immutable Hugging
Face revisions, and uses the base model's tokenizer and chat template for both sides
of each comparison.

The minimal Colab entry point is
`notebooks/harmony_checkpoint_eval_colab.ipynb`. Most of the implementation lives in
`scripts/harmony_eval/`:

- `catalog.py` defines the three released matched checkpoint pairs.
- `cases.py` renders direct-question evaluation cases.
- `scoring.py` performs local seq2seq or causal-LM sequence scoring.
- `analysis.py` fits monotone sacrifice thresholds and plots response curves.
- `runner.py` runs one matched pair and writes reproducibility metadata.

Start with `caramel_sft`, the released SFT model based on
`google/flan-t5-large`. It fits comfortably on a standard Colab GPU. The two 7B DPO
pairs default to identical 4-bit loading for the base and aligned checkpoints and may
need a High-RAM runtime.

Every run writes a timestamped directory containing:

- `raw_scores.csv`: per-model Yes/No log probabilities and implementation scores.
- `thresholds.csv`: base/aligned threshold estimates and their log shift when both
  thresholds are inside the tested range.
- `curves.png`: response curves for each scenario family.
- `metadata.json`: model revisions, tokenizer revision, prompt hashes, package
  versions, settings, and repository commit.

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
