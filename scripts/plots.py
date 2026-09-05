import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#6f6e6a"
GRID = "#e6e5e1"
SERIES_1 = "#2a78d6"

LINE_WIDTH = 2.0
MARKER_SIZE = 4.5
MARKER_RING = 1.5


def _style_axes(ax, title, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, pad=10, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9, labelpad=8)
    ax.set_ylim(-4, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)


def _plot_series(ax, x, y):
    ax.plot(
        x,
        y,
        color=SERIES_1,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        markerfacecolor=SERIES_1,
        markeredgecolor=SURFACE,
        markeredgewidth=MARKER_RING,
        clip_on=False,
        zorder=3,
    )


def _annotate(ax, x, y, label, offset=(0, 14)):
    ax.annotate(
        label,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        color=TEXT_PRIMARY,
        fontsize=8.5,
        ha="center",
        zorder=4,
    )


def render(caps, refusal, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=200, sharex=True, sharey=True)
    fig.patch.set_facecolor(SURFACE)

    caps_x = [point["injected_norm"] for point in caps["points"]]
    caps_y = [100 * point["measure"] for point in caps["points"]]
    _style_axes(
        axes[0],
        f"ALL-CAPS, layer {caps['layer']}: share of generated tokens",
        "Target behaviour observed (%)",
    )
    _plot_series(axes[0], caps_x, caps_y)
    peak = max(range(len(caps_y)), key=lambda i: caps_y[i])
    _annotate(axes[0], caps_x[peak], caps_y[peak], f"peak {caps_y[peak]:.0f}%", offset=(4, -22))
    _annotate(axes[0], caps_x[0], caps_y[0], f"baseline {caps_y[0]:.0f}%", offset=(38, 18))
    _annotate(axes[0], caps_x[-1], caps_y[-1], f"{caps_y[-1]:.0f}%", offset=(-4, 14))

    refusal_x = [point["injected_norm"] for point in refusal["points"]]
    refusal_y = [100 * point["measure"] for point in refusal["points"]]
    _style_axes(axes[1], f"Refusal, layer {refusal['layer']}: share of prompts")
    _plot_series(axes[1], refusal_x, refusal_y)
    saturated = next((i for i, value in enumerate(refusal_y) if value >= 100), len(refusal_y) - 1)
    _annotate(axes[1], refusal_x[saturated], refusal_y[saturated], "all refused", offset=(30, -14))
    _annotate(axes[1], refusal_x[0], refusal_y[0], f"baseline {refusal_y[0]:.0f}%", offset=(38, 18))

    fig.supxlabel(
        "Injected norm  (coefficient \u00d7 vector norm)",
        color=TEXT_SECONDARY,
        fontsize=9,
        y=0.035,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
