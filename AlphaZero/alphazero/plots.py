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

_SAMPLE_LOCATOR = MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10])


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


def _ema(values, span):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    alpha = 2.0 / (max(2, int(span)) + 1.0)
    out = np.empty_like(values)
    out[0] = values[0]
    for index in range(1, len(values)):
        out[index] = alpha * values[index] + (1.0 - alpha) * out[index - 1]
    return out


def _rolling_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return values
    window = max(1, int(window))
    csum = np.concatenate(([0.0], np.cumsum(values)))
    result = np.empty(len(values))
    for index in range(len(values)):
        first = max(0, index - window + 1)
        result[index] = (csum[index + 1] - csum[first]) / (index + 1 - first)
    return result


def _sci_label(value, _position=None):
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = round(value / (10.0**exponent), 3)
    if abs(mantissa) >= 10.0:
        mantissa /= 10.0
        exponent += 1
    if mantissa == int(mantissa):
        mantissa = int(mantissa)
    return f"{mantissa:g}e{exponent}"


def winrate_history(game_records, window, sample_every):
    """Rolling black/draw/white rates sampled every `sample_every` games."""
    if not game_records:
        return []
    sample_every = max(1, int(sample_every))
    winners = np.array([winner for _, winner, _, _ in game_records])
    cum_black = np.cumsum(winners == 1)
    cum_white = np.cumsum(winners == -1)
    history = []
    for index, (game_index, _, _, _) in enumerate(game_records):
        if game_index % sample_every:
            continue
        low = max(0, index + 1 - window)
        count = index + 1 - low
        black = (cum_black[index] - (cum_black[low - 1] if low else 0)) / count
        white = (cum_white[index] - (cum_white[low - 1] if low else 0)) / count
        history.append((game_index, black, 1.0 - black - white, white))
    return history


def _selfplay_axes_data(game_records):
    """Cumulative self-play samples and the owning iteration per game."""
    lengths = np.array([length for _, _, length, _ in game_records], dtype=np.float64)
    iterations = np.array(
        [iteration for _, _, _, iteration in game_records], dtype=np.float64
    )
    return np.cumsum(lengths), iterations


def _attach_iteration_axis(axis, samples, iterations):
    """Add a top x-axis that re-expresses self-play samples as iterations."""
    if len(samples) < 2 or np.all(iterations == iterations[0]):
        return None
    unique_iterations = []
    start_samples = []
    for index, iteration in enumerate(iterations):
        if not unique_iterations or iteration != unique_iterations[-1]:
            unique_iterations.append(iteration)
            start_samples.append(samples[index - 1] if index else 0.0)

    secondary = axis.secondary_xaxis(
        "top",
        functions=(
            lambda values: np.interp(values, samples, iterations),
            lambda values: np.interp(values, unique_iterations, start_samples),
        ),
    )
    secondary.set_xlabel("Iteration", color=TEXT)
    secondary.xaxis.set_major_locator(MaxNLocator(integer=True))
    secondary.tick_params(colors=MUTED, labelsize=9)
    secondary.spines["top"].set_color(SPINE)
    return secondary


def _style_sample_axis(axis, samples):
    axis.xaxis.set_major_locator(_SAMPLE_LOCATOR)
    axis.xaxis.set_major_formatter(FuncFormatter(_sci_label))
    if len(samples) == 1:
        axis.set_xlim(samples[0] - 0.5, samples[0] + 0.5)


def _plot_loss_series(axis, values, color, label, span):
    if not values:
        return False
    x = np.arange(1, len(values) + 1)
    y = np.asarray(values, dtype=np.float64)
    axis.plot(x, y, color=color, linewidth=1.0, alpha=0.35)
    axis.plot(x, _ema(y, span), color=color, linewidth=2.0, label=label)
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


def render_training(out_dir, losses, game_records, winrate_window, winrate_sample_every):
    apply_theme()
    figure, axes = plt.subplots(2, 2, figsize=(15.0, 9.5), layout="constrained")
    iterations = len(losses["total"])
    games = len(game_records)
    total_samples = int(sum(length for _, _, length, _ in game_records))
    figure.suptitle("AlphaZero training progress", fontsize=17)
    figure.supxlabel(
        f"{iterations} training iteration(s) · {games:,} self-play game(s) · "
        f"{total_samples:,} self-play samples · win-rate window {winrate_window} games",
        fontsize=10,
        color=MUTED,
    )

    samples = iterations_per_game = None
    if game_records:
        samples, iterations_per_game = _selfplay_axes_data(game_records)

    axis = axes[0, 0]
    history = winrate_history(game_records, winrate_window, winrate_sample_every)
    if history:
        sample_of_game = {
            game_index: sample
            for game_index, sample in zip(
                (record[0] for record in game_records), samples
            )
        }
        sample_axis = [sample_of_game[entry[0]] for entry in history]
        _, black, draw, white = zip(*history)
        axis.plot(sample_axis, black, color=BLUE, linewidth=2.0, label="Black")
        axis.plot(sample_axis, white, color=RED, linewidth=2.0, label="White")
        axis.plot(sample_axis, draw, color=GREY, linewidth=1.6, label="Draw")
        axis.axhline(0.5, color=MUTED, linewidth=1.0, alpha=0.8)
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.legend(fontsize=9)
    else:
        _empty_axis(axis, "No self-play games recorded yet")
    axis.set_title("Self-play outcome rates (rolling)")
    axis.set_xlabel("Self-play samples")
    axis.set_ylabel("Share of games")
    _style_axis(axis, integer_x=False)
    if history:
        _style_sample_axis(axis, samples)
        _attach_iteration_axis(axis, samples, iterations_per_game)

    axis = axes[0, 1]
    if game_records:
        lengths = np.array(
            [length for _, _, length, _ in game_records], dtype=np.float64
        )
        axis.scatter(
            samples, lengths, s=8, color=BLUE, alpha=0.22, linewidths=0, label="Per game"
        )
        window = min(50, len(lengths))
        axis.plot(
            samples,
            _rolling_mean(lengths, window),
            color=ORANGE,
            linewidth=2.0,
            label=f"Mean (window {window})",
        )
        axis.legend(fontsize=9)
    else:
        _empty_axis(axis, "No self-play games recorded yet")
    axis.set_title("Self-play game length")
    axis.set_xlabel("Self-play samples")
    axis.set_ylabel("Steps per game")
    _style_axis(axis, integer_x=False)
    if game_records:
        _style_sample_axis(axis, samples)
        _attach_iteration_axis(axis, samples, iterations_per_game)

    axis = axes[1, 0]
    if _plot_loss_series(axis, losses["total"], ORANGE, "Total (EMA)", 20):
        _use_log_scale(axis, [losses["total"]])
        axis.legend(fontsize=9)
    else:
        _empty_axis(axis, "No training iterations recorded yet")
    axis.set_title("Total loss")
    axis.set_xlabel("Training iteration")
    axis.set_ylabel("Loss")
    _style_axis(axis)

    axis = axes[1, 1]
    plotted = _plot_loss_series(axis, losses["policy"], BLUE, "Policy (EMA)", 20)
    plotted |= _plot_loss_series(axis, losses["value"], GREEN, "Value (EMA)", 20)
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
