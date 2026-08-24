# Value Misalignment

This repository contains the Stage 1 pilot for measuring how a model trades wetland preservation against involuntary family displacement.

## Stage 1 runner

The experiment uses `deepseek-v4-flash` in non-thinking mode. For each configured displacement count, it scores both A/B orderings, converts the returned label log probabilities into a binary probability of choosing displacement, and averages across the two positions.

All scenario wording, choice wording, displacement counts, model settings, and output settings are external to the runner:

- `configs/stage_1.toml` controls the experiment.
- `prompts/stage_1_wetland_relocation.txt` contains the prompt template.
- `scripts/run_stage_1.py` contains only experiment and API logic.

The default sweep is `0, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000`, giving 20 API requests after A/B reversal.

### Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Set the API key in the shell. Do not put it in the repository or configuration file.

```bash
export DEEPSEEK_API_KEY="your-key-here"
```

Inspect all 20 rendered requests without making API calls:

```bash
python scripts/run_stage_1.py --config configs/stage_1.toml --dry-run
```

Run the experiment:

```bash
python scripts/run_stage_1.py --config configs/stage_1.toml
```

Each run creates four timestamped files under `results/`:

- A raw CSV with one row per API request.
- A raw JSONL file including the rendered prompt and returned top-token log probabilities.
- A summary CSV with both position-specific probabilities and their mean.
- A metadata JSON file containing the complete configuration and observed model fingerprints.

If either A or B is absent from DeepSeek's top 20 first-token candidates, the runner stops instead of silently treating the missing label as zero probability.

### Tests

The local tests do not call DeepSeek or require an API key:

```bash
python -m unittest discover -s tests -v
```
