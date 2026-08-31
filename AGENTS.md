# Repository Working Instructions

## Commit completed work

- Unless the user explicitly says not to commit, commit all completed changes made for the task before handing the work back.
- Commit only files that belong to the current task. Preserve unrelated user changes and never include them merely to obtain a clean worktree.
- Run checks appropriate to the change before committing, and report any checks that could not be run.
- Use a concise commit message that describes the completed research or implementation change.

## Push notebook changes automatically

- If a completed commit includes any notebook (`.ipynb`) change, push that commit to the configured remote automatically after all required checks pass. The user does not need to request the push separately.
- For commits without notebook changes, do not push to any remote unless the user explicitly asks for a push in the current task.
- After committing non-notebook changes that were not explicitly requested for push, give the user the exact shell command needed to push them.

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
4. Commit the day's log, onboarding update, and task files together; then follow the push policy above, including automatic pushes for commits containing notebook changes.
