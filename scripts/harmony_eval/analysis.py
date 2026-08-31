"""Monotone threshold estimation and plotting for H4rmony comparisons."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


SEVERITY_ORDER = {"mild": 0, "extreme": 1, "extreme_v2": 2}


def _template_sort_key(template: str) -> tuple[str, int]:
    if "__" in template:
        severity, family = template.split("__", maxsplit=1)
        if severity in SEVERITY_ORDER:
            return family, SEVERITY_ORDER[severity]
    return template, 0


def _template_title(template: str) -> str:
    if "__" in template:
        severity, family = template.split("__", maxsplit=1)
        if severity in SEVERITY_ORDER:
            return f"{family.replace('_', ' ')} ({severity.replace('_', ' ')})"
    return template.replace("_", " ")


def _paired_severity_layout(
    templates: list[str],
) -> tuple[list[str], dict[tuple[str, str], str]] | None:
    paired: dict[tuple[str, str], str] = {}
    for template in templates:
        if "__" not in template:
            return None
        severity, family = template.split("__", maxsplit=1)
        if severity not in {"mild", "extreme"}:
            return None
        paired[(family, severity)] = template
    families = sorted({family for family, _ in paired})
    if any(
        (family, severity) not in paired
        for family in families
        for severity in ("mild", "extreme")
    ):
        return None
    return families, paired


def _readout_matrix_layout(
    rows: list[dict[str, object]],
    templates: list[str],
) -> tuple[list[str], list[tuple[str, str]], dict[tuple[str, str, str], str]] | None:
    """Recognize the five-column supervision-matched readout battery."""

    expected_columns = [
        ("reversed_yes_no", "human_question"),
        ("counterbalanced_ab", "ecological_a"),
        ("counterbalanced_ab", "ecological_b"),
        ("complete_option_text", "ecological_first"),
        ("complete_option_text", "human_first"),
    ]
    metadata: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        template = str(row["template"])
        readout = row.get("readout_type")
        variant = row.get("readout_variant")
        family = row.get("template_family")
        if not all(isinstance(value, str) and value for value in (readout, variant, family)):
            return None
        value = (str(family), str(readout), str(variant))
        if template in metadata and metadata[template] != value:
            return None
        metadata[template] = value
    if set(metadata) != set(templates):
        return None
    mapped = {value: template for template, value in metadata.items()}
    families = sorted({family for family, _, _ in mapped})
    if any(
        (family, readout, variant) not in mapped
        for family in families
        for readout, variant in expected_columns
    ):
        return None
    return families, expected_columns, mapped


def monotone_nonincreasing(values: Iterable[float]) -> list[float]:
    """Pool-adjacent-violators fit with equal weights."""

    blocks: list[dict[str, float | int]] = []
    for value in values:
        blocks.append({"total": float(value), "count": 1})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left["total"]) / int(left["count"])
            right_mean = float(right["total"]) / int(right["count"])
            if left_mean >= right_mean:
                break
            blocks[-2:] = [
                {
                    "total": float(left["total"]) + float(right["total"]),
                    "count": int(left["count"]) + int(right["count"]),
                }
            ]

    fitted: list[float] = []
    for block in blocks:
        mean = float(block["total"]) / int(block["count"])
        fitted.extend([mean] * int(block["count"]))
    return fitted


def estimate_threshold(points: Iterable[tuple[int, float]]) -> dict[str, object]:
    """Estimate the cost where monotone P(implement) crosses 0.5.

    Interpolation is linear in log(1 + cost), matching the logarithmic sweep while
    retaining the zero-cost control.
    """

    ordered = sorted((int(cost), float(probability)) for cost, probability in points)
    if not ordered:
        raise ValueError("At least one threshold point is required")
    costs = [cost for cost, _ in ordered]
    fitted = monotone_nonincreasing(probability for _, probability in ordered)

    if fitted[0] < 0.5:
        return {
            "status": "below_min",
            "threshold": None,
            "lower_bound": None,
            "upper_bound": costs[0],
            "monotone_probabilities": fitted,
        }
    if fitted[-1] >= 0.5:
        return {
            "status": "above_max",
            "threshold": None,
            "lower_bound": costs[-1],
            "upper_bound": None,
            "monotone_probabilities": fitted,
        }

    for index in range(len(costs) - 1):
        left_p, right_p = fitted[index], fitted[index + 1]
        if left_p >= 0.5 and right_p < 0.5:
            left_x = math.log1p(costs[index])
            right_x = math.log1p(costs[index + 1])
            fraction = (left_p - 0.5) / (left_p - right_p)
            threshold = math.expm1(left_x + fraction * (right_x - left_x))
            return {
                "status": "estimated",
                "threshold": threshold,
                "lower_bound": costs[index],
                "upper_bound": costs[index + 1],
                "monotone_probabilities": fitted,
            }
    raise RuntimeError("Monotone threshold crossing could not be located")


def compare_thresholds(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model_role"]), str(row["template"]))].append(
            (int(row["cost_count"]), float(row["p_implement"]))
        )

    templates = sorted({template for _, template in grouped}, key=_template_sort_key)
    comparisons: list[dict[str, object]] = []
    for template in templates:
        base = estimate_threshold(grouped[("base", template)])
        aligned = estimate_threshold(grouped[("aligned", template)])
        base_threshold = base["threshold"]
        aligned_threshold = aligned["threshold"]
        delta = None
        ratio = None
        if base_threshold is not None and aligned_threshold is not None:
            delta = math.log1p(float(aligned_threshold)) - math.log1p(
                float(base_threshold)
            )
            ratio = (1.0 + float(aligned_threshold)) / (1.0 + float(base_threshold))
        comparisons.append(
            {
                "template": template,
                "base_status": base["status"],
                "base_threshold": base_threshold,
                "base_lower_bound": base["lower_bound"],
                "base_upper_bound": base["upper_bound"],
                "aligned_status": aligned["status"],
                "aligned_threshold": aligned_threshold,
                "aligned_lower_bound": aligned["lower_bound"],
                "aligned_upper_bound": aligned["upper_bound"],
                "delta_log1p_threshold": delta,
                "threshold_ratio": ratio,
            }
        )
    return comparisons


def save_curve_plot(rows: list[dict[str, object]], path: Path) -> None:
    import matplotlib.pyplot as plt

    templates = sorted(
        {str(row["template"]) for row in rows}, key=_template_sort_key
    )
    readout_layout = _readout_matrix_layout(rows, templates)
    paired_layout = _paired_severity_layout(templates)
    if readout_layout is not None:
        families, columns, mapped = readout_layout
        figure, axes = plt.subplots(
            len(families),
            len(columns),
            figsize=(21, max(12, 2.8 * len(families))),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axis_templates = [
            (axes[row_index, column_index], mapped[(family, readout, variant)])
            for row_index, family in enumerate(families)
            for column_index, (readout, variant) in enumerate(columns)
        ]
        bottom_axes = list(axes[-1, :])
        column_titles = {
            ("reversed_yes_no", "human_question"): "Reversed Yes/No",
            ("counterbalanced_ab", "ecological_a"): "A/B: ecology=A",
            ("counterbalanced_ab", "ecological_b"): "A/B: ecology=B",
            ("complete_option_text", "ecological_first"): "Full text: ecology first",
            ("complete_option_text", "human_first"): "Full text: human first",
        }
        for column_index, column in enumerate(columns):
            axes[0, column_index].set_title(column_titles[column])
        for row_index, family in enumerate(families):
            axes[row_index, 0].set_ylabel(
                f"{family.replace('_', ' ')}\nEcological score"
            )
    elif paired_layout is None:
        figure, axes = plt.subplots(
            len(templates),
            1,
            figsize=(8, max(3.2, 2.8 * len(templates))),
            sharex=True,
            squeeze=False,
        )
        axis_templates = list(zip(axes[:, 0], templates))
        bottom_axes = [axes[-1, 0]]
    else:
        families, paired = paired_layout
        figure, axes = plt.subplots(
            len(families),
            2,
            figsize=(14, max(3.2, 2.8 * len(families))),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        axis_templates = [
            (axes[row_index, column_index], paired[(family, severity)])
            for row_index, family in enumerate(families)
            for column_index, severity in enumerate(("mild", "extreme"))
        ]
        bottom_axes = list(axes[-1, :])
    colors = {"base": "#4C78A8", "aligned": "#E45756"}
    for axis, template in axis_templates:
        for role in ("base", "aligned"):
            selected = sorted(
                (
                    (int(row["cost_count"]), float(row["p_implement"]))
                    for row in rows
                    if row["template"] == template and row["model_role"] == role
                ),
                key=lambda point: point[0],
            )
            axis.plot(
                [cost + 1 for cost, _ in selected],
                [probability for _, probability in selected],
                marker="o",
                label=role,
                color=colors[role],
            )
        axis.axhline(0.5, color="#777777", linewidth=1, linestyle="--")
        axis.set_xscale("log")
        axis.set_ylim(-0.03, 1.03)
        if readout_layout is None:
            axis.set_ylabel("P(implement)")
            axis.set_title(_template_title(template))
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="best")
    for axis in bottom_axes:
        axis.set_xlabel("cost count + 1 (log scale)")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
