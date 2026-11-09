# Fig. 4, single-column layout: the two city panels stacked vertically (same data and encodings as the earlier two-column version)
import sys
from pathlib import Path

import pandas as pd
from matplotlib.ticker import PercentFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import micastyle as ms

HERE = Path(__file__).resolve().parents[2] / "results" / "figure_data"
OUT = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "fig4_policy_field.csv")
df["family"] = df["family"].replace({"ADAPT": "reconstructed"})

TEAL = "#2f8f7a"
GOLD = "#c99a2e"
PURPLE = "#8a5aab"
FAMILY_MARKER = {
    "control": ("s", ms.ORANGE), "classical": ("D", ms.GREY),
    "DRO": ("^", GOLD), "OSF": ("o", TEAL), "reconstructed": ("v", PURPLE),
}
LOW = 0.40

ms.apply()
ms.plt.rcParams.update({"font.sans-serif": ["Liberation Sans", "Arial"], "font.size": 7, "axes.labelsize": 7,
                        "xtick.labelsize": 6.6, "ytick.labelsize": 6.6, "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
fig = ms.plt.figure(figsize=(3.45, 3.8))
gs = fig.add_gridspec(5, 1, height_ratios=[1, 2.3, 0.62, 1, 2.3], hspace=0.07,
                      left=0.135, right=0.985, top=0.92, bottom=0.09)
CITIES = ["Philadelphia", "Tampa"]
axes = {}
for r, city in enumerate(CITIES):
    ax_top = fig.add_subplot(gs[3 * r]); ax_bot = fig.add_subplot(gs[3 * r + 1], sharex=ax_top)
    axes[city] = (ax_top, ax_bot)
    sub = df[df.city == city]
    n = int(sub["n_groups"].iloc[0])
    for fam, (marker, color) in FAMILY_MARKER.items():
        fsub = sub[sub.family == fam]
        for ax in (ax_top, ax_bot):
            ax.scatter(fsub["intervention_rate"], fsub["irrecoverability"], marker=marker, s=13,
                       facecolor="white", edgecolor=color, linewidth=0.9, label=fam, zorder=3)
    mica = sub[sub.policy == "vista"]
    for ax in (ax_top, ax_bot):
        ax.scatter(mica["intervention_rate"], mica["irrecoverability"], marker="*", s=70,
                   color=ms.NAVY, edgecolor="white", linewidth=0.5, zorder=5)
    ax_bot.axhspan(-0.045, 0.012, color="#dbe9f5", zorder=0)
    ax_bot.text(0.985, -0.034, "zero irrecoverability", transform=ax_bot.get_yaxis_transform(),
                ha="right", va="center", fontsize=6.2, color=ms.NAVY, fontstyle="italic", zorder=8)
    mrow = mica.iloc[0]
    ax_bot.annotate("MICA", xy=(mrow["intervention_rate"], mrow["irrecoverability"]),
                    xytext=(mrow["intervention_rate"] + 0.03, -0.010), fontsize=6.8, color=ms.NAVY,
                    fontweight="bold", ha="left", va="center", zorder=8)
    ctrl = sub[sub.policy == "horizon_maxmin"].iloc[0]
    ax_bot.annotate("horizon control", xy=(ctrl["intervention_rate"], ctrl["irrecoverability"]),
                    xytext=(ctrl["intervention_rate"] + 0.03, -0.010), fontsize=6.6, color=ms.ORANGE,
                    fontweight="bold", ha="left", va="center", zorder=8)
    bg = sub[sub.policy == "borda"].iloc[0]
    dx, dy = (0.02, 0.12) if city == "Philadelphia" else (-0.22, 0.10)
    ax_bot.annotate("Borda = GFAR", xy=(bg["intervention_rate"], bg["irrecoverability"]),
                    xytext=(bg["intervention_rate"] + dx, bg["irrecoverability"] + dy), fontsize=6.2,
                    color=ms.GREY, ha="left", zorder=8,
                    arrowprops=dict(arrowstyle="-", color=ms.GREY, linewidth=0.6, connectionstyle="arc3,rad=0.05"))
    base = sub[sub.policy == "utilitarian"].iloc[0]
    ax_bot.axhline(base["irrecoverability"], color=ms.GREY, linewidth=0.6, linestyle=":", zorder=1)
    ax_bot.text(0.985, base["irrecoverability"] + 0.008, "base = group-DRO = S-DRO",
                transform=ax_bot.get_yaxis_transform(), fontsize=6.2, color=ms.GREY, ha="right", va="bottom", zorder=8)
    if city == "Philadelphia":
        hi = sub[sub.policy.isin(["adapt_fadjo_w0.5", "adapt_fadjo_w1"])].iloc[0]
    else:
        hi = sub[sub.family == "reconstructed"].sort_values("irrecoverability").iloc[-1]
    ax_top.annotate("reconstructions, random, proportional share", xy=(hi["intervention_rate"], hi["irrecoverability"]),
                    xytext=(0.02, 0.70), fontsize=6.2, color=ms.GREY, ha="left", va="center", zorder=8,
                    arrowprops=dict(arrowstyle="-", color=ms.GREY, linewidth=0.6, connectionstyle="arc3,rad=0.15"))
    ax_top.set_ylim(0.505, sub["irrecoverability"].max() * 1.08)
    ax_bot.set_ylim(-0.045, LOW)
    ax_top.spines["bottom"].set_visible(False); ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(bottom=False, labelbottom=False)
    ms.ygrid(ax_top); ms.ygrid(ax_bot)
    ax_bot.set_xlim(-0.04, 0.97)
    ax_bot.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    d = 0.012
    kw = dict(transform=ax_top.transAxes, color=ms.INK, clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d * 3, +d * 3), **kw)
    kw.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d * 1.3, 1 + d * 1.3), **kw)
    ax_top.text(0.0, 1.08, f"{city}   (n = {n})", transform=ax_top.transAxes, fontsize=7.8,
                fontweight="bold", ha="left", va="bottom")
    ax_bot.set_xlabel("intervention rate", labelpad=1.5)
    for ax in (ax_top, ax_bot): ms.tick_color(ax)
fig.supylabel("group-trip irrecoverability", fontsize=7, x=0.01)
handles, labels = axes["Philadelphia"][1].get_legend_handles_labels()
seen = dict(zip(labels, handles))
fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=5, fontsize=6.3, bbox_to_anchor=(0.55, 1.0),
           frameon=False, handletextpad=0.1, columnspacing=0.9, borderaxespad=0.1)
for e in ("png", "pdf"): fig.savefig(OUT / ("fig4." + e), dpi=600)
