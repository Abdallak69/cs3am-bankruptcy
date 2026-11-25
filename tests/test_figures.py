"""figures.py computes expected cost by hand-checkable counts and draws
Figures 2 and 3 for any set of scorers. Small made-up scores are enough, since
the module reads no data. Files go to a temporary folder."""

import copy
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")

import src.figures as figures  # noqa: E402
from src.config import load_config  # noqa: E402

CONFIG = load_config()

LABELS = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
GOOD = np.array([0.9, 0.4, 0.5, 0.2, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05])
WEAK = np.array([0.3, 0.1, 0.6, 0.5, 0.2, 0.4, 0.1, 0.3, 0.2, 0.1])
CURVES = {"good": (LABELS, GOOD), "weak": (LABELS, WEAK)}


def test_expected_cost_counts_misses_and_false_alarms() -> None:
    # At 0.45: flags 0.9 (hit) and 0.5 (false alarm); misses 0.4.
    # Cost = (10 * 1 miss + 1 false alarm) / 10 statements = 1.1.
    assert figures.expected_cost(LABELS, GOOD, 0.45, 10) == pytest.approx(1.1)
    # Flagging everything: no misses, 8 false alarms.
    assert figures.expected_cost(LABELS, GOOD, 0.0, 10) == pytest.approx(0.8)


def test_cost_ratios_come_from_config() -> None:
    assert CONFIG["cost_ratios"] == [5, 10, 20, 50]


def test_precision_recall_legend_shows_pr_auc() -> None:
    figure = figures.precision_recall_figure(CURVES, "test")
    texts = [t.get_text() for t in figure.axes[0].get_legend().get_texts()]

    assert texts[0] == f"good ({average_precision_score(LABELS, GOOD):.3f})"
    assert texts[-1] == "Random (0.200)"


def test_cost_figure_plots_every_ratio_and_the_flag_none_line() -> None:
    thresholds = {name: {r: 0.45 for r in CONFIG["cost_ratios"]} for name in CURVES}
    figure = figures.cost_figure(CURVES, thresholds, "test", CONFIG)
    lines = {line.get_label(): line for line in figure.axes[0].get_lines()}

    expected = [
        figures.expected_cost(LABELS, GOOD, 0.45, r) for r in CONFIG["cost_ratios"]
    ]
    assert list(lines["good"].get_ydata()) == pytest.approx(expected)
    assert list(lines["Flag no firm"].get_ydata()) == pytest.approx(
        [0.2 * r for r in CONFIG["cost_ratios"]]
    )
    assert list(lines["Flag every firm"].get_ydata()) == pytest.approx([0.8] * 4)


def test_cost_figure_needs_a_threshold_for_every_ratio() -> None:
    thresholds = {name: {5: 0.5} for name in CURVES}
    with pytest.raises(KeyError):
        figures.cost_figure(CURVES, thresholds, "test", CONFIG)


def test_save_figure_writes_a_png(tmp_path: Path) -> None:
    config = copy.deepcopy(CONFIG)
    config["paths"]["results_dir"] = str(tmp_path)

    path = figures.save_figure(
        figures.precision_recall_figure(CURVES, "test"), "figure2_test", config
    )

    assert path == tmp_path / "figures" / "figure2_test.png" and path.is_file()
