# Repository Working Instructions

## Commit completed work

- Unless the user explicitly says not to commit, commit all completed changes made for the task before handing the work back.
- Commit only files that belong to the current task. Preserve unrelated user changes and never include them merely to obtain a clean worktree.
- Run checks appropriate to the change before committing, and report any checks that could not be run.
- Use a concise commit message that describes the completed research or implementation change.

## Push completed work automatically

- Unless the user explicitly says not to, push completed task commits to the configured remote after all required checks pass. This applies to documentation and code as well as notebooks; the user does not need to request the push separately.
- Do not give the user instructions or shell commands for pushing. Briefly report completion or any unresolved failure.

## Distinguish observations from explanations

- Previous controls produced effects similar to the ecological SFT interventions. This finding limits claims that those effects were specific to ecological training; it does not establish "generic confidence compression" as a causal mechanism.
- Treat descriptions or fitted patterns in probabilities and margins as descriptive evidence. Label proposed mechanisms as hypotheses unless the evidence establishes them.
- The proposed opposing-value training arms provide a natural matched comparison for shared training effects. Do not treat speculative confidence compression as an established obstacle to that design.

## Maintain the dated research log

- Record substantive research, experimental, implementation, and analysis work in `doc/YYYY-MM-DD-research-log.md`, using the current local date for the filename and the heading `# Research Log — YYYY-MM-DD`.
- If a log for the date already exists, append a clearly titled section; do not overwrite or duplicate earlier entries.
- Record what was attempted, the exact setup or intervention, what changed, the observed results, limitations or unresolved questions, and the next steps when applicable.
- Keep the log evidence-based and reproducible. Include relevant model and dataset identifiers, immutable revisions, configurations, seeds, sample sizes, metrics, and artifact paths when available.
- Update the research log in the same commit as the work it documents.

## Checkout procedure

Before ending any session that changes the repository or produces a research result:

1. Append a concise account of the session to that day's `doc/YYYY-MM-DD-research-log.md`, including what changed, checks or results, limitations, and the next step.
2. Update only question 3 in `doc/onboarding.md`: give it the session date and a brief, specific question about the work just completed. Keep questions 1 and 2 stable and keep the file short.
3. Verify that a new model can answer all three onboarding questions by reading every file in `doc/`.
4. Commit the day's log, onboarding update, and task files together; then push automatically unless the user has explicitly opted out.
