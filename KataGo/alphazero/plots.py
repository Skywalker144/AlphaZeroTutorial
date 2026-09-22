"""Dark-themed training dashboard and metric CSV exports."""

import csv
import math
import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter

# Dark research palette.
FIGURE_BACKGROUND = "#1e2127"
AXES_BACKGROUND = "#282c34"
LEGEND_BACKGROUND = "#21252b"
TEXT = "#abb2bf"
TITLE = "#c8ccd4"
MUTED = "#8a93a3"
GRID = "#3b4250"
SPINE = "#454c5a"
BLUE = "#61afef"
RED = "#e06c75"
ORANGE = "#e8924b"
GREEN = "#98c379"
GREY = "#9aa2b1"

def apply_theme():
    plt.rcParams.update(
        {
            "figure.facecolor": FIGURE_BACKGROUND,
            "axes.facecolor": AXES_BACKGROUND,
            "savefig.facecolor": FIGURE_BACKGROUND,
            "text.color": TEXT,
            "axes.labelcolor": TEXT,
            "axes.titlecolor": TITLE,
            "axes.edgecolor": SPINE,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "legend.facecolor": LEGEND_BACKGROUND,
            "legend.edgecolor": SPINE,
            "legend.labelcolor": TEXT,
            "legend.frameon": False,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "grid.linewidth": 0.7,
        }
    )


def _style_axis(axis, *, integer_x=True):
    axis.set_facecolor(AXES_BACKGROUND)
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(SPINE)
    axis.spines["bottom"].set_color(SPINE)
    axis.tick_params(colors=MUTED, labelsize=9)
    if integer_x:
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))


def _empty_axis(axis, message):
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=10,
        transform=axis.transAxes,
    )
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def selfplay_iteration_history(game_records):
    """Per-iteration (end samples, black/draw/white rates, mean length).

    Each game has equal weight within its iteration. A point is positioned at
    the last recorded sample of that iteration, including a partial latest one.
    """
    groups = {}
    cumulative_samples = 0
    for _, winner, length, iteration in game_records:
        cumulative_samples += length
        # Games, black wins, white wins, total moves, final sample position.
        group = groups.setdefault(iteration, [0, 0, 0, 0, 0])
        group[0] += 1
        group[1] += winner == 1
        group[2] += winner == -1
        group[3] += length
        group[4] = cumulative_samples
    return [
        (end, black / count, (count - black - white) / count,
         white / count, steps / count)
        for count, black, white, steps, end in groups.values()
    ]


def _style_selfplay_axis(axis, title, ylabel, total_samples):
    _style_axis(axis, integer_x=False)
    axis.set_title(title, loc="left", fontsize=12, pad=36)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Cumulative self-play samples (iteration end)")
    axis.grid(axis="y", alpha=0.55)
    axis.tick_params(axis="both", length=3)
    axis.xaxis.set_major_locator(
        MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
    )
    axis.xaxis.set_major_formatter(
        FuncFormatter(lambda value, _: f"{value / 1000:g}k" if value else "0")
    )
    axis.set_xlim(0, max(1, total_samples * 1.025))


def _selfplay_legend(axis, columns):
    axis.legend(
        loc="lower left", bbox_to_anchor=(0, 1.01), ncol=columns,
        borderaxespad=0, fontsize=9, handlelength=2, columnspacing=1.5,
    )


def _plot_loss_series(axis, values, color, label):
    if not values:
        return False
    x = np.arange(1, len(values) + 1)
    y = np.asarray(values, dtype=np.float64)
    axis.plot(x, y, color=color, linewidth=1.6, label=label)
    return True


def _use_log_scale(axis, series):
    finite = [value for values in series for value in values if math.isfinite(value)]
    if finite and all(value > 0.0 for value in finite):
        axis.set_yscale("log")


def write_metrics_csv(out_dir, losses, game_records):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "losses.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["iteration", "total", "policy", "value"])
        for index, (total, policy, value) in enumerate(
            zip(losses["total"], losses["policy"], losses["value"])
        ):
            writer.writerow([index, total, policy, value])
    with open(os.path.join(out_dir, "games.csv"), "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["game_index", "iteration", "winner", "length", "cumulative_samples"]
        )
        cumulative = 0
        for game_index, winner, length, iteration in game_records:
            cumulative += length
            writer.writerow([game_index, iteration, winner, length, cumulative])


def render_training(out_dir, losses, game_records):
    apply_theme()
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 9.5), layout="constrained")
    iterations = len(losses["total"])
    games = len(game_records)
    total_samples = int(sum(length for _, _, length, _ in game_records))
    figure.suptitle("AlphaZero training progress", fontsize=17)
    figure.supxlabel(
        f"{iterations} loss records · {games:,} self-play games · "
        f"{total_samples:,} self-play samples · self-play means per iteration",
        fontsize=10,
        color=MUTED,
    )

    history = selfplay_iteration_history(game_records)
    if history:
        samples, black, draw, white, mean_length = zip(*history)

    axis = axes[0, 0]
    _style_selfplay_axis(axis, "Self-play outcomes", "Share of games", total_samples)
    if history:
        for label, color, series in (
            ("Black win", BLUE, black),
            ("White win", RED, white),
            ("Draw", GREY, draw),
        ):
            axis.plot(samples, series, color=color, linewidth=1.8, label=label)
        axis.set_ylim(0.0, 1.025)
        axis.set_yticks(np.arange(0, 1.01, 0.2))
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        _selfplay_legend(axis, 3)
    else:
        _empty_axis(axis, "No self-play games recorded yet")

    axis = axes[0, 1]
    _style_selfplay_axis(axis, "Self-play game length", "Moves per game", total_samples)
    if history:
        axis.plot(samples, mean_length, color=ORANGE, linewidth=1.8, label="Mean")
        lengths = [length for _, _, length, _ in game_records]
        low, high = min(lengths), max(lengths)
        padding = max(1, high - low) * 0.05
        axis.set_ylim(low - padding, high + padding)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        _selfplay_legend(axis, 1)
    else:
        _empty_axis(axis, "No self-play games recorded yet")

    axis = axes[1, 0]
    if _plot_loss_series(axis, losses["total"], ORANGE, "Total"):
        _use_log_scale(axis, [losses["total"]])
        axis.legend(fontsize=9)
    else:
        _empty_axis(axis, "No training iterations recorded yet")
    axis.set_title("Total loss")
    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Loss")
    _style_axis(axis)

    axis = axes[1, 1]
    plotted = _plot_loss_series(axis, losses["policy"], BLUE, "Policy")
    plotted |= _plot_loss_series(axis, losses["value"], GREEN, "Value")
    if plotted:
        _use_log_scale(axis, [losses["policy"], losses["value"]])
        axis.legend(fontsize=9)
    else:
        _empty_axis(axis, "No training iterations recorded yet")
    axis.set_title("Loss components")
    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Loss")
    _style_axis(axis)

    path = os.path.join(out_dir, "training.png")
    figure.savefig(path, dpi=160, facecolor=FIGURE_BACKGROUND)
    plt.close(figure)
    return path
