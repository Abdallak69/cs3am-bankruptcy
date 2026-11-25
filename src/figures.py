"""Figures 2 and 3 of the report: precision recall curves and expected cost.

Figure 2 plots the precision recall curve of every scorer from its
uncalibrated scores, with the bankrupt share as the line a random ranking
would follow. In the report it uses the test scores from the single test run
in evaluate.py; before then it can be drawn from out of fold predictions to
check the code.

Figure 3 plots expected cost against the miss to false alarm cost ratio in
config.yaml (cost_ratios). A missed bankruptcy costs the ratio and a false
alarm costs 1, so the expected cost per statement at a threshold t is

    (ratio * missed bankruptcies + false alarms) / statements.

The thresholds are chosen on out of fold predictions, never on the scores
being plotted, and passed in; flagging no firm is drawn for reference, as is flagging every firm.

This module only draws: it reads no data and fits nothing.

Owner: Zack.
"""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.config import REPO_ROOT, load_config

FIGURE_DPI: int = 300

Curves = dict[str, tuple[np.ndarray | pd.Series, np.ndarray | pd.Series]]


def expected_cost(
    labels: np.ndarray | pd.Series,
    probabilities: np.ndarray | pd.Series,
    threshold: float,
    cost_ratio: float,
) -> float:
    """Compute the expected cost per statement of flagging at one threshold.

    A statement is flagged when its probability is at or above the
    threshold. Each missed bankruptcy costs cost_ratio and each false alarm
    costs 1.

    Args:
        labels: Class labels, 1 = bankrupt.
        probabilities: Calibrated probabilities of bankruptcy.
        threshold: Probability at or above which a statement is flagged.
        cost_ratio: Cost of a missed bankruptcy relative to a false alarm.

    Returns:
        The mean cost per statement, in units of one false alarm.
    """
    labels = np.asarray(labels)
    flagged = np.asarray(probabilities) >= threshold
    misses = np.sum((labels == 1) & ~flagged)
    false_alarms = np.sum((labels == 0) & flagged)
    return float((cost_ratio * misses + false_alarms) / len(labels))


def precision_recall_figure(curves: Curves, title: str) -> Figure:
    """Draw Figure 2: the precision recall curve of each scorer.

    Args:
        curves: For each scorer name, its labels and uncalibrated risk
            scores, higher meaning more likely bankrupt.
        title: Figure title.

    Returns:
        The figure. Each legend entry carries the scorer's PR AUC (average
        precision), the same measure as cv_metrics in evaluate.py.
    """
    figure, axes = plt.subplots(figsize=(6, 5))
    share = None
    for scorer, (labels, scores) in curves.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = average_precision_score(labels, scores)
        # Step plot, matching how average precision sums the area.
        axes.step(recall, precision, where="post", label=f"{scorer} ({pr_auc:.3f})")
        share = float(np.mean(labels))
    if share is not None:
        axes.axhline(share, linestyle="--", color="grey", label=f"Random ({share:.3f})")
    axes.set_xlabel("Recall")
    axes.set_ylabel("Precision")
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1.02)
    axes.set_title(title)
    axes.legend(title="Scorer (PR AUC)")
    figure.tight_layout()
    return figure


def cost_figure(
    curves: Curves,
    thresholds: dict[str, dict[float, float]],
    title: str,
    config: dict[str, Any] | None = None,
) -> Figure:
    """Draw Figure 3: expected cost against cost ratio for each scorer.

    Args:
        curves: For each scorer name, its labels and calibrated
            probabilities.
        thresholds: For each scorer name, its threshold at each cost ratio,
            chosen on out of fold predictions.
        title: Figure title.
        config: Parsed project configuration, for cost_ratios. Read from
            config.yaml if None.

    Returns:
        The figure, with a log scaled cost ratio axis and flagging no firm
        and flagging every firm drawn for reference.

    Raises:
        KeyError: If a scorer lacks a threshold for one of the cost ratios.
    """
    if config is None:
        config = load_config()
    ratios = config["cost_ratios"]
    figure, axes = plt.subplots(figsize=(6, 5))
    labels = None
    for scorer, (labels, probabilities) in curves.items():
        costs = [
            expected_cost(labels, probabilities, thresholds[scorer][ratio], ratio)
            for ratio in ratios
        ]
        axes.plot(ratios, costs, marker="o", label=scorer)
    if labels is not None:
        # Flagging no firm misses every bankruptcy: cost = ratio * share.
        share = float(np.mean(labels))
        axes.plot(
            ratios,
            [ratio * share for ratio in ratios],
            linestyle="--",
            color="grey",
            label="Flag no firm",
        )
        # Flagging every firm raises a false alarm on every survivor.
        axes.plot(
            ratios,
            [1 - share] * len(ratios),
            linestyle=":",
            color="grey",
            label="Flag every firm",
        )
    axes.set_xscale("log")
    axes.set_xticks(ratios, [str(ratio) for ratio in ratios])
    axes.minorticks_off()
    axes.set_xlabel("Cost of a miss / cost of a false alarm")
    axes.set_ylabel("Expected cost per statement (false alarms)")
    axes.set_title(title)
    axes.legend()
    figure.tight_layout()
    return figure


def save_figure(
    figure: Figure, name: str, config: dict[str, Any] | None = None
) -> Path:
    """Save a figure as a 300 dpi PNG in results/figures and close it.

    Args:
        figure: The figure.
        name: File name without extension, e.g. figure2_pr_curves.
        config: Parsed project configuration. Read from config.yaml if None.

    Returns:
        The file written.
    """
    if config is None:
        config = load_config()
    path = REPO_ROOT / config["paths"]["results_dir"] / "figures" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=FIGURE_DPI)
    plt.close(figure)
    return path
