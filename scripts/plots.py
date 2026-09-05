import os
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import patheffects

SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
SERIES_1 = "#2a78d6"

HAND_FONTS = ["Chalkboard SE", "Humor Sans", "xkcd Script", "Comic Sans MS", "DejaVu Sans"]

SKETCH_SCALE = 0.9
SKETCH_LENGTH = 120
SKETCH_RANDOMNESS = 2

FIGSIZE = (6.8, 4.6)
DPI = 200
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


def render_sweep(points, xlabel, ylabel, path):
    x = [point["coefficient"] for point in points]
    y = [100 * point["measure"] for point in points]

    with hand_drawn():
        fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

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

        ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        ax.set_ylim(-5, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(0, max(x) * 1.02)
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=11, length=6, width=1.4)

        fig.tight_layout()
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)


def render(caps, refusal, caps_path, refusal_path):
    render_sweep(
        caps["points"],
        f"Steering coefficient  (layer {caps['layer']})",
        "Generated tokens in ALL CAPS (%)",
        caps_path,
    )
    render_sweep(
        refusal["points"],
        f"Steering coefficient  (layer {refusal['layer']})",
        "Prompts refused (%)",
        refusal_path,
    )
