import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import micastyle as ms

HERE = Path(__file__).resolve().parents[2] / "results" / "figure_data"
OUT = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "fig1_trajectory.csv")
alpha = float(df["alpha"].iloc[0])

RED = "#c0392b"

ms.apply()
ms.plt.rcParams.update({"font.sans-serif":["Liberation Sans","Arial"],"font.size":7,"axes.labelsize":7,"xtick.labelsize":6.6,"ytick.labelsize":6.6,"legend.fontsize":6.4,"lines.markersize":4,"lines.linewidth":1.2,"savefig.pad_inches":0.02})
fig, ax = ms.plt.subplots(figsize=(3.45, 1.72))
ax.set_facecolor(ms.PALE)

ax.plot(df["day"], df["A_sigma"], color=ms.NAVY, marker="o",
        markerfacecolor="white", markeredgewidth=1.4, zorder=4,
        label=r"robust attainability ceiling $A_\sigma(X_t)$")
ax.plot(df["day"], df["worst_member_coverage"], color=ms.GREY, linestyle="--",
        marker="s", markerfacecolor="white", markeredgewidth=1.2, zorder=3,
        label=r"realised worst-member coverage $\mathrm{min}_i\, c_i(t)$")
ax.axhline(alpha, color=RED, linestyle="--", linewidth=1.4, zorder=2)
ax.text(df["day"].min(), alpha - 0.035, r"contract level $\alpha$",
       color=RED, fontsize=6.8, fontweight="bold", va="top", ha="left")

cross = df.loc[df["A_sigma"] < alpha - 1e-9, "day"]
if len(cross):
    day0 = int(cross.iloc[0])
    y0 = float(df.loc[df["day"] == day0, "A_sigma"].iloc[0])
    ax.plot([day0], [y0], marker="X", markersize=8, color=RED,
            markeredgecolor="white", markeredgewidth=0.8, zorder=6)
    ax.annotate("irrecoverable\nfrom here", xy=(day0, y0),
                xytext=(day0 + 0.30, y0 - 0.20),
                fontsize=6.8, color=RED, fontweight="bold", ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=RED, linewidth=0.9))

    tail = df[df["day"] >= day0]
    if len(tail) >= 2:
        mx = float(tail["day"].iloc[-1])
        my = float(tail["worst_member_coverage"].iloc[-1])
        ax.annotate("coverage still rising:\nthe loss is never announced",
                    xy=(mx, my), xytext=(mx - 1.75, my + 0.15),
                    fontsize=6.2, color=ms.GREY, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=ms.GREY, linewidth=0.8,
                                    connectionstyle="arc3,rad=-0.25"))

ax.set_xlabel("day of the horizon $t$")
ax.set_ylabel("coverage (share of own reference)")
ax.set_xticks(df["day"])
ax.set_xlim(df["day"].min() - 0.15, df["day"].max() + 0.15)
ax.set_ylim(0.0, 1.22)
ms.ygrid(ax)
leg = ax.legend(loc="lower right", facecolor="white", framealpha=0.9)
ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
ms.tick_color(ax)

fig.savefig(OUT / "fig1.png",dpi=600,bbox_inches="tight")
