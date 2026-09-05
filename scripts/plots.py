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


def _style_axes(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, pad=10, loc="left")
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9, labelpad=8)
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    caps_x = [point["coefficient"] for point in caps["points"]]
    caps_y = [100 * point["measure"] for point in caps["points"]]
    _style_axes(
        axes[0],
        f"ALL-CAPS steering, layer {caps['layer']}",
        f"coefficient  (1.0 = the fitted vector, norm {caps['vector_norm']:.1f})",
        "Generated tokens in ALL CAPS (%)",
    )
    _plot_series(axes[0], caps_x, caps_y)
    peak = max(range(len(caps_y)), key=lambda i: caps_y[i])
    _annotate(axes[0], caps_x[peak], caps_y[peak], f"peak {caps_y[peak]:.0f}%")
    _annotate(axes[0], caps_x[0], caps_y[0], f"baseline {caps_y[0]:.0f}%", offset=(34, 18))
    _annotate(axes[0], caps_x[-1], caps_y[-1], f"{caps_y[-1]:.0f}%", offset=(-6, 12))

    refusal_x = [point["coefficient"] for point in refusal["points"]]
    refusal_y = [100 * point["measure"] for point in refusal["points"]]
    _style_axes(
        axes[1],
        f"Refusal steering, layer {refusal['layer']}",
        "coefficient  (unit direction, so this is the injected norm)",
        "Prompts refused (%)",
    )
    _plot_series(axes[1], refusal_x, refusal_y)
    saturated = next((i for i, value in enumerate(refusal_y) if value >= 100), len(refusal_y) - 1)
    _annotate(axes[1], refusal_x[saturated], refusal_y[saturated], "all refused")
    _annotate(axes[1], refusal_x[0], refusal_y[0], f"baseline {refusal_y[0]:.0f}%", offset=(34, 18))

    fig.suptitle(
        "Steering strength vs behaviour, Qwen3-1.7B",
        x=0.011,
        y=0.98,
        ha="left",
        color=TEXT_PRIMARY,
        fontsize=13,
    )
    fig.text(
        0.011,
        0.015,
        f"{caps['n_prompts']} prompts per point, greedy decoding, {caps['max_new_tokens']}"
        " new tokens.  Left: share of letter-bearing generated tokens that are all caps."
        "  Right: share of prompts matching a refusal marker.",
        color=TEXT_MUTED,
        fontsize=8,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.93))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
