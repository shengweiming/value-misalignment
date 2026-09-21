"""Validate complete binary or explicit-candidate score matrices and their math."""

import math


def check_fields(row: dict, expected: dict) -> None:
    for key, value in expected.items():
        if isinstance(value, float):
            matches = math.isfinite(float(row[key])) and math.isclose(float(row[key]), value, rel_tol=0, abs_tol=1e-10)
        else:
            matches = str(row.get(key, "")) == ("" if value is None else str(value))
        if not matches:
            raise RuntimeError(f"Evaluation result mismatch: {key}")


def validate_case_rows(rows: list[dict], cases: list[dict], identity: dict) -> None:
    explicit_candidates = "candidates" in cases[0]
    expected = {}
    for case in cases:
        if not explicit_candidates:
            expected[(case["case_id"],)] = case
        else:
            for index, candidate in enumerate(case["candidates"], 1):
                expected[(case["case_id"], str(candidate["value"]))] = {
                    **{k: v for k, v in case.items() if k != "candidates"},
                    "candidate_index": index, "candidate_value": candidate["value"],
                    "candidate_text": candidate["text"], "candidate_scored_text": candidate["text"],
                    "candidate_token_count": 1,
                }
    if len(rows) != len(expected):
        raise RuntimeError("Incomplete Evaluation score matrix")
    seen, candidate_groups = set(), {}
    for row in rows:
        key = (row["case_id"], row["candidate_value"]) if explicit_candidates else (row["case_id"],)
        if key in seen or key not in expected:
            raise RuntimeError("Duplicate or unknown Evaluation score row")
        seen.add(key)
        check_fields(row, {
            **expected[key], **identity,
        })
        if explicit_candidates:
            logprob = float(row["candidate_logprob"])
            if not math.isfinite(logprob) or logprob > 0:
                raise RuntimeError("Invalid Evaluation candidate log probability")
            check_fields(row, {"candidate_mean_logprob": logprob})
            candidate_groups.setdefault(row["case_id"], []).append(row)
        else:
            values = {}
            for side in ("implement", "reject"):
                count = int(row[f"candidate_tokens_{side}"])
                logprob = float(row[f"logprob_{side}"])
                if count < 1 or not math.isfinite(logprob) or logprob > 0:
                    raise RuntimeError("Invalid Evaluation choice score")
                if row["readout_type"] == "counterbalanced_ab" and count != 1:
                    raise RuntimeError("Evaluation A/B labels must each be one token")
                values[f"mean_logprob_{side}"] = logprob / count
            summed = float(row["logprob_implement"]) - float(row["logprob_reject"])
            mean = values["mean_logprob_implement"] - values["mean_logprob_reject"]
            margin = mean if row["candidate_score_normalization"] == "mean" else summed
            values.update(semantic_logit_sum=summed, semantic_logit_mean=mean, semantic_logit_implement=margin)
            for name, value in (("p_implement", margin), ("p_implement_sum", summed), ("p_implement_mean", mean)):
                values[name] = 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))
            check_fields(row, values)
    for group in candidate_groups.values():
        maximum = max(float(row["candidate_logprob"]) for row in group)
        weights = [math.exp(float(row["candidate_logprob"]) - maximum) for row in group]
        for row, weight in zip(group, weights):
            check_fields(row, {"candidate_probability": weight / sum(weights)})
