# Value Misalignment

This repository contains the Stage 1 pilot for measuring how a model trades wetland preservation against harms to families.

## Stage 1 runner

The experiment supports `deepseek-v4-flash` and `qwen3-8b` in non-thinking mode. It scores `Yes` against `No` for each configured family count. Experiments can use the direct implementation question alone or pair it with a reversed rejection question; direct-only mode avoids the negation artifact observed in the initial pilot.

All scenario wording, question wording, response labels, family counts, model settings, and output settings are external to the runner:

- `configs/stage_1.toml` controls the displacement experiment.
- `configs/stage_1_elimination.toml` controls the explicit elimination experiment.
- `configs/stage_1_elimination_direct.toml` is the direct-question-only elimination baseline used for cross-model comparisons.
- `configs/stage_1_elimination_ecology_constitution.toml` adds the ecology-values constitution as a system-level instruction and uses the direct implementation question only.
- `configs/stage_1_elimination_ecology_constitution_without_guardrail.toml` is an ablation containing only the constitution's first two paragraphs.
- `configs/stage_1_elimination_ecology_severe_human_costs.toml` adds an explicit, bounded provision allowing exceptional ecological value to justify severe human costs.
- `configs/models/qwen3_8b.toml` selects hosted Qwen3-8B without changing an experiment condition.
- `configs/models/qwen3_8b_beijing.toml` selects the Beijing-hosted Qwen3-8B endpoint using the same key environment variable.
- `prompts/stage_1_wetland_relocation.txt` and `prompts/stage_1_wetland_elimination.txt` contain the prompt templates.
- `constitutions/ecological_diversity.txt` contains the three-paragraph constitution separately from the scenario prompt.
- `constitutions/ecological_diversity_without_guardrail.txt` contains the corresponding two-paragraph ablation.
- `constitutions/ecological_diversity_severe_human_costs.txt` contains the explicit ecological-priority condition.
- `scripts/run_stage_1.py` contains only experiment and API logic.
- `doc/2026-08-24-research-log.md` records the measurement revisions, model results, interpretation, and proposed SFT follow-up.

The prompt specifies a 250,000-hectare (2,500 km²) wetland complex. The default coarse sweep is `0, 1, 10, 100, 1000, 5000, 10000, 100000, 500000, 1000000`, giving 20 independent API requests after question-polarity reversal.

Every answer is obtained from a new stateless one-turn request containing only the system message and the current prompt. DeepSeek requests receive a unique `user_id`; DashScope session caching is explicitly disabled. No conversation history is reused.

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
export DASHSCOPE_API_KEY="your-singapore-model-studio-key"
```

For DashScope, activate Model Studio Large Model Inference in the Singapore region before calling a model. Use the default workspace or ensure that the API key's workspace authorizes `qwen3-8b`.

Inspect all 20 rendered requests without making API calls:

```bash
python scripts/run_stage_1.py --config configs/stage_1.toml --dry-run
```

Run the experiment:

```bash
python scripts/run_stage_1.py --config configs/stage_1.toml
```

Run the explicit elimination condition:

```bash
python scripts/run_stage_1.py --config configs/stage_1_elimination.toml
```

Run its direct-question-only baseline:

```bash
python scripts/run_stage_1.py --config configs/stage_1_elimination_direct.toml
```

Run the direct-question elimination condition with the ecology-values constitution:

```bash
python scripts/run_stage_1.py --config configs/stage_1_elimination_ecology_constitution.toml
```

Run the ablation without the constitution's third paragraph:

```bash
python scripts/run_stage_1.py --config configs/stage_1_elimination_ecology_constitution_without_guardrail.toml
```

Run the condition allowing exceptional ecological value to justify severe human costs:

```bash
python scripts/run_stage_1.py --config configs/stage_1_elimination_ecology_severe_human_costs.toml
```

Run any condition on hosted Qwen3-8B by adding its model profile:

```bash
python scripts/run_stage_1.py \
  --config configs/stage_1_elimination_ecology_severe_human_costs.toml \
  --model-profile configs/models/qwen3_8b.toml
```

Qwen3-8B uses DashScope's native generation endpoint because its OpenAI-compatible endpoint currently omits log-probability data. The native endpoint returns at most five candidate tokens. The runner requires genuine first-token log probabilities and stops if they are absent or if either response label is outside the returned candidates; it never substitutes a sampling-frequency estimate.

For a Beijing-region key, substitute `configs/models/qwen3_8b_beijing.toml` as the model profile.

Each run creates four timestamped files under `results/`:

- A raw CSV with one row per API request.
- A raw JSONL file including the rendered prompt and returned top-token log probabilities.
- A summary CSV with the direct implementation probability and, when the rejection polarity is enabled, the reversed probability, polarity gap, and log-odds-symmetrized score.
- A metadata JSON file containing the complete configuration and observed model fingerprints.

If either `Yes` or `No` is absent from the configured first-token candidates, the runner stops instead of silently treating the missing label as zero probability.

### Tests

The local tests do not call either provider or require an API key:

```bash
python -m unittest discover -s tests -v
```
