# Research Log — 2026-08-31

## Fifty-item low-reasoning generation run

The 50-item run at
`outputs/ecological_dilemmas/20260830T154818452672Z_quality_pipeline_n50`
completed with seed 43. It accepted 50 dilemmas from 51 sampled combinations,
with one rejection. The pipeline used GPT-5.6 Sol at low reasoning for planning,
review, and validation and GPT-5.6 Terra at medium reasoning for writing. It made
51 planner, 51 reviewer, 51 writer, and 56 validator calls. The recorded estimated
standard-priority cost was $8.459904.

A reproducible five-item manual sample used sampling seed 20260831 and selected
accepted items 13, 19, 30, 33, and 44. Items 13 and 30 were clean accepts. Item 19
needed the approved card's fish-passage compromise exclusion restored to the
prose. Item 44 was usable but would benefit from restoring its house-level and
warning-system exclusions. Item 33 was a near-semantic duplicate of item 9 in the
earlier 10-item run: both concerned an island causeway, ferry-dependent mobility,
restricted tidal exchange, and eelgrass or shallow-water ecological effects.
This confirmed that balancing, assignment uniqueness, and novelty context had
previously been scoped only to one run.

## Cross-run continuation support

Added repeatable `--prior-run` inputs for generating a new batch as a continuation
of completed earlier batches. The loader verifies that each prior manifest is
complete, checks construct and decision-maker source hashes, validates every
accepted record and definition, rejects duplicate prior paths or assignments,
and records manifest and record hashes in the new manifest.

Accepted prior assignments are removed from the new sampler before generation.
Their marginal and pairwise counts seed balancing, so the new batch is balanced
against the combined dataset rather than in isolation. Their novelty signatures
seed planning, review, and exact-signature rejection. A repeated seed is now safe
for exact assignment uniqueness when all earlier runs are supplied, although
semantic uniqueness remains a model judgment rather than a mathematical
guarantee. A larger `--novelty-window` is therefore appropriate when continuing
the 60 existing examples.

Tests cover repeated CLI inputs, exact assignment exclusion under the same seed,
balance seeding, prior-run manifest provenance, and propagation of earlier
novelty signatures to both planner and reviewer calls.

An integration dry-run used `--count 40 --seed 43 --novelty-window 100` with the
completed 10-item and 50-item directories as prior runs. All 40 proposed new
assignments were unique and had zero overlap with the 60 accepted prior
assignments. Across the resulting 100-assignment plan, ecological-object,
human-interest, and policy-mechanism frequencies each had a maximum-minus-minimum
spread of one; every decision-maker appeared exactly ten times. The dry-run
manifest recorded `prior_count: 60` and both prior-run provenance entries.
