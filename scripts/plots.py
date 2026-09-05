import os
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patheffects

SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
SERIES_1 = "#2a78d6"

HAND_FONTS = ["Chalkboard SE", "Humor Sans", "xkcd Script", "Comic Sans MS", "DejaVu Sans"]

SKETCH_SCALE = 0.9
SKETCH_LENGTH = 120
SKETCH_RANDOMNESS = 2

LINE_WIDTH = 2.6
MARKER_SIZE = 4.8


def _fonts():
    override = os.environ.get("PLOT_FONT")
    return [override, *HAND_FONTS] if override else HAND_FONTS


@contextmanager
def hand_drawn():
    with plt.xkcd(scale=SKETCH_SCALE, length=SKETCH_LENGTH, randomness=SKETCH_RANDOMNESS):
        plt.rcParams.update(
            {
                "font.family": _fonts(),
                "path.effects": [patheffects.withStroke(linewidth=4, foreground=SURFACE)],
                "figure.facecolor": SURFACE,
                "axes.facecolor": SURFACE,
                "axes.edgecolor": TEXT_PRIMARY,
                "axes.linewidth": 1.6,
                "text.color": TEXT_PRIMARY,
                "axes.labelcolor": TEXT_PRIMARY,
                "xtick.color": TEXT_PRIMARY,
                "ytick.color": TEXT_PRIMARY,
            }
        )
        yield


def _style_axes(ax, title, ylabel=None):
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=13, pad=14, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_PRIMARY, fontsize=11, labelpad=10)
    ax.set_ylim(-6, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-6, 172)
    ax.set_xticks([0, 40, 80, 120, 160])
    ax.grid(False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=11, length=6, width=1.4)


def _plot_series(ax, x, y):
    ax.plot(
        x,
        y,
        color=SERIES_1,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        markerfacecolor=SERIES_1,
        markeredgewidth=0,
        solid_capstyle="round",
        clip_on=False,
        zorder=3,
    )


def _callout(ax, xy, text, offset, rad=0.25, ha="center"):
    ax.annotate(
        text,
        xy=xy,
        xytext=offset,
        textcoords="offset points",
        fontsize=10.5,
        color=TEXT_PRIMARY,
        ha=ha,
        va="center",
        zorder=5,
        arrowprops={
            "arrowstyle": "-",
            "color": TEXT_SECONDARY,
            "linewidth": 1.2,
            "shrinkA": 2,
            "shrinkB": 6,
            "connectionstyle": f"arc3,rad={rad}",
        },
    )


def render(caps, refusal, path):
    with hand_drawn():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.0), dpi=200, sharex=True, sharey=True)

        caps_x = [point["injected_norm"] for point in caps["points"]]
        caps_y = [100 * point["measure"] for point in caps["points"]]
        _style_axes(
            axes[0],
            f"ALL-CAPS, layer {caps['layer']}: share of tokens",
            "Target behaviour observed (%)",
        )
        _plot_series(axes[0], caps_x, caps_y)
        peak = max(range(len(caps_y)), key=lambda i: caps_y[i])
        _callout(
            axes[0],
            (caps_x[peak], caps_y[peak]),
            f"peak {caps_y[peak]:.0f}%",
            (54, 24),
            rad=-0.25,
        )
        _callout(
            axes[0],
            (caps_x[-1], caps_y[-1]),
            "over-steered:\nthe model falls apart",
            (14, -62),
            rad=0.3,
        )
        _callout(axes[0], (caps_x[0], caps_y[0]), "baseline 1%", (68, 24), rad=-0.3)

        refusal_x = [point["injected_norm"] for point in refusal["points"]]
        refusal_y = [100 * point["measure"] for point in refusal["points"]]
        _style_axes(axes[1], f"Refusal, layer {refusal['layer']}: share of prompts")
        _plot_series(axes[1], refusal_x, refusal_y)
        saturated = next((i for i, value in enumerate(refusal_y) if value >= 100), -1)
        _callout(
            axes[1],
            (refusal_x[saturated], refusal_y[saturated]),
            "all 10 refused,\nstill coherent",
            (40, -48),
            rad=0.3,
        )
        _callout(axes[1], (refusal_x[4], refusal_y[4]), "onset", (-48, 42), rad=0.25)
        _callout(axes[1], (refusal_x[0], refusal_y[0]), "baseline 0%", (60, 28), rad=-0.3)

        fig.supxlabel(
            "Injected norm  (coefficient × vector norm)",
            color=TEXT_PRIMARY,
            fontsize=11,
            y=0.04,
        )
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
