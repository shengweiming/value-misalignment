# Value Misalignment

This repository contains the Stage 1 pilot for measuring how a model trades wetland preservation against involuntary family displacement.

## Stage 1 runner

The experiment uses `deepseek-v4-flash` in non-thinking mode. For each configured displacement count, it asks two separate proposition questions: whether the authorities should implement the plan and whether they should reject it. It scores `Yes` against `No`, reverses the semantic mapping for the rejection question, and averages the resulting implementation log odds.

All scenario wording, question wording, response labels, displacement counts, model settings, and output settings are external to the runner:

- `configs/stage_1.toml` controls the experiment.
- `prompts/stage_1_wetland_relocation.txt` contains the prompt template.
- `scripts/run_stage_1.py` contains only experiment and API logic.

The prompt specifies a 250,000-hectare (2,500 km²) wetland complex. The default coarse sweep is `0, 1, 10, 100, 1000, 5000, 10000, 100000, 500000, 1000000`, giving 20 independent API requests after question-polarity reversal.

Every answer is obtained from a new stateless one-turn request containing only the system message and the current prompt. Each request also receives a unique `user_id` to isolate DeepSeek's request-side cache. No conversation history is reused.

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
- A summary CSV with both polarity-specific probabilities, their arithmetic mean, a log-odds-symmetrized probability, and the polarity effect.
- A metadata JSON file containing the complete configuration and observed model fingerprints.

If either `Yes` or `No` is absent from DeepSeek's top 20 first-token candidates, the runner stops instead of silently treating the missing label as zero probability.

### Tests

The local tests do not call DeepSeek or require an API key:

```bash
python -m unittest discover -s tests -v
```
