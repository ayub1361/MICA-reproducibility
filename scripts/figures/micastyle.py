import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY = "#1b3a5c"
ORANGE = "#c0562a"
GREY = "#737373"
LGREY = "#d2d2d2"
INK = "#333333"
PALE = "#f2f4f6"

FIGW = 8.75

def apply():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.labelcolor": INK,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.14,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "grid.color": LGREY,
        "grid.linewidth": 0.6,
    })


def ygrid(ax):
    ax.yaxis.grid(True, color=LGREY, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def tick_color(ax):
    ax.tick_params(colors=INK)


def panel_title(ax, title):
    ax.set_title(title, loc="left", fontweight="bold", fontsize=11)


def save(fig, name, dirs):
    import os
    paths = []
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        for ext in ("png", "pdf"):
            p = os.path.join(d, f"{name}.{ext}")
            fig.savefig(p)
            paths.append(p)
    return paths
