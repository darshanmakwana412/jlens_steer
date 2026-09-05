import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e6e5e1"
SERIES_1 = "#2a78d6"

FIGSIZE = (6.4, 4.2)
DPI = 200
LINE_WIDTH = 2.0
MARKER_SIZE = 4.0


def render_sweep(points, xlabel, ylabel, path):
    x = [point["coefficient"] for point in points]
    y = [100 * point["measure"] for point in points]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(
        x,
        y,
        color=SERIES_1,
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=MARKER_SIZE,
        markerfacecolor=SERIES_1,
        markeredgewidth=0,
        clip_on=False,
        zorder=3,
    )

    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9.5, labelpad=8)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9.5, labelpad=8)
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(0, max(x) * 1.02)
    ax.grid(axis="y", color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=0)

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
