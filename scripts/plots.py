import os
from contextlib import contextmanager

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects

SURFACE = "#ffffff"
TEXT_PRIMARY = "#0b0b0b"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

HAND_FONTS = ["Chalkboard SE", "Humor Sans", "xkcd Script", "Comic Sans MS", "DejaVu Sans"]

SKETCH_SCALE = 0.9
SKETCH_LENGTH = 120
SKETCH_RANDOMNESS = 2

FIGSIZE = (6.8, 4.6)
MIN_LABEL_GAP = 6.0
DPI = 200
LINE_WIDTH = 2.6
MARKER_SIZE = 4.8


def _fonts():
    installed = {font.name for font in font_manager.fontManager.ttflist}
    override = os.environ.get("PLOT_FONT")
    wanted = [override, *HAND_FONTS] if override else HAND_FONTS
    return [name for name in wanted if name in installed] or ["DejaVu Sans"]


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
            color=SERIES[0],
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=SERIES[0],
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


def _spaced(values, gap):
    order = sorted(range(len(values)), key=lambda i: values[i])
    placed = list(values)
    for rank, index in enumerate(order):
        if rank == 0:
            continue
        previous = placed[order[rank - 1]]
        placed[index] = max(placed[index], previous + gap)
    return placed


def render_comparison(series, xlabel, ylabel, path):
    with hand_drawn():
        fig, ax = plt.subplots(figsize=(7.8, 5.0), dpi=DPI)

        ends = []
        for index, (label, points) in enumerate(series):
            x = [point["coefficient"] for point in points]
            y = [100 * point["measure"] for point in points]
            colour = SERIES[index % len(SERIES)]
            ax.plot(
                x,
                y,
                color=colour,
                linewidth=LINE_WIDTH,
                marker="o",
                markersize=MARKER_SIZE,
                markerfacecolor=colour,
                markeredgewidth=0,
                solid_capstyle="round",
                label=label,
                clip_on=False,
                zorder=3 + index,
            )
            ends.append((x[-1], y[-1], colour))

        span = max(point["coefficient"] for _, points in series for point in points)
        for (x_end, y_end, colour), y_label in zip(
            ends, _spaced([end[1] for end in ends], MIN_LABEL_GAP), strict=True
        ):
            ax.annotate(
                f"{y_end:.0f}%",
                xy=(x_end + 0.03 * span, y_label),
                fontsize=10,
                color=colour,
                ha="left",
                va="center",
                annotation_clip=False,
                zorder=6,
            )

        ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        ax.set_ylim(-5, 108)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xlim(0, span * 1.1)
        ax.grid(False)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(labelsize=11, length=6, width=1.4)
        ax.legend(frameon=False, fontsize=10.5, loc="upper right", bbox_to_anchor=(1.0, 0.55))

        fig.tight_layout()
        fig.savefig(path, facecolor=SURFACE)
        plt.close(fig)
