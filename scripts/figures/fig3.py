import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import micastyle as ms

HERE = Path(__file__).resolve().parents[2] / "results" / "figure_data"
OUT = Path(__file__).resolve().parent
raw = pd.read_csv(HERE / "fig3_day_of_loss.csv")

DAYS = [1, 2, 3, 4]
CITIES = [("Philadelphia", ms.NAVY), ("Tampa", ms.ORANGE)]

ms.apply()
ms.plt.rcParams.update({"font.sans-serif":["Liberation Sans","Arial"],"font.size":7,"axes.labelsize":7,"xtick.labelsize":6.6,"ytick.labelsize":6.6,"legend.fontsize":6.4,"lines.markersize":4,"lines.linewidth":1.2,"savefig.pad_inches":0.02})
fig, ax = ms.plt.subplots(figsize=(3.45, 1.6))

width = 0.36
x = np.arange(len(DAYS))
ops_x = -1.0

for i, (city, color) in enumerate(CITIES):
    sub = raw[raw.city == city]
    genuine = sub[sub.operational_failure == 0]
    counts = [int((genuine.irrecoverable_day == d).sum()) for d in DAYS]
    offset = (i - 0.5) * width
    ax.bar(x + offset, counts, width=width, color=color, label=city, zorder=3)
    for xi, c in zip(x + offset, counts):
        if c:
            ax.text(xi, c + 0.12, str(c), ha="center", va="bottom", fontsize=6.2,
                    color=ms.INK, gid="overlay")

    n_ops = int((sub.operational_failure == 1).sum())
    if n_ops:
        ax.bar(ops_x + offset, n_ops, width=width, color="#98a6b5",
              hatch="//", zorder=3, edgecolor=ms.INK, linewidth=0.5)
        ax.text(ops_x + offset, n_ops + 0.12, str(n_ops), ha="center", va="bottom",
                fontsize=6.2, color=ms.INK, gid="overlay")

xt = [ops_x] + list(x)
xtl = ["ops failure\n(no day)"] + [str(d) for d in DAYS]
ax.set_xticks(xt)
ax.set_xticklabels(xtl)
ax.set_xlabel("day of irrecoverability (unfiltered base policy)")
ax.set_ylabel("failed trips")
ms.ygrid(ax)
ax.axvline(-0.5, color=ms.LGREY, linewidth=0.8, zorder=1)
ax.legend(loc="upper left")
ax.set_ylim(0, 8)
ms.tick_color(ax)

fig.savefig(OUT / "fig3.png",dpi=600,bbox_inches="tight")
