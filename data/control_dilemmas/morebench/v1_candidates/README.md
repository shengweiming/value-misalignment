# MoReBench prompt-only control candidates

This is the first-stage candidate pool for a non-ecological prompt-only training
control. It is not yet the frozen 98-example training release.

The source is the 500-row public MoReBench CSV under CC-BY-4.0, pinned to dataset
revision `8290fafe65d595aaa28315b50ec4b64da6d3bd5e`. The tracked source file has
SHA-256 `e56d627823066876c6710a91144d0d9faebc1503dcf9b665f58c87b0eddd2229`.

The first filter retains the `Education`, `Entertainment`, and
`Interpersonal relationship` contexts. They contain 35, 14, and 66 rows,
respectively: 115 in total. Three rows are then excluded after content review:

- a moral-licensing case framed by green consumption and benefits to the planet;
- a relationship case explicitly about animal-rights advocacy; and
- a resort case whose stated misconduct includes destroying the environment.

This leaves 112 eligible prompts, 14 more than the intended 98-example release.
`candidates.jsonl` contains those prompts and provenance metadata. The only
training text is the `dilemma` field. It contains no assistant response, rubric,
preferred answer, extracted action field, or normative label. `audit.jsonl`
records the disposition of all 500 public rows, including the exact three manual
exclusions. `manifest.json` pins the source and generated-artifact hashes and
reports the source, role, type, context, and word-count distributions.

The final selection is intentionally not frozen here. The candidate pool mixes
80 `ai_advisor` and 32 `ai_agent` prompts, and only 35 of the 112 prompts fall
within the ecological corpus's 198--293-word range. Selecting the longest 98
would improve the length match but retain all 32 AI-agent cases; preferring the
advisor role would reduce that role confound but worsen the token-dose mismatch.
The final 98-row rule should be declared before downstream results are inspected,
and training should match the ecological arm's exact Qwen token dose rather than
assuming that equal example counts imply equal exposure.

Regenerate the pool with:

```bash
python3 scripts/build_morebench_prompt_control_candidates.py
```
