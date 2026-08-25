"""Monotone threshold estimation and plotting for H4rmony comparisons."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


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

    templates = sorted({template for _, template in grouped})
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

    templates = sorted({str(row["template"]) for row in rows})
    figure, axes = plt.subplots(
        len(templates),
        1,
        figsize=(8, max(3.2, 2.8 * len(templates))),
        sharex=True,
        squeeze=False,
    )
    colors = {"base": "#4C78A8", "aligned": "#E45756"}
    for axis, template in zip(axes[:, 0], templates):
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
        axis.set_ylabel("P(implement)")
        axis.set_title(template.replace("_", " "))
        axis.grid(alpha=0.2)
    axes[0, 0].legend(loc="best")
    axes[-1, 0].set_xlabel("cost count + 1 (log scale)")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)
