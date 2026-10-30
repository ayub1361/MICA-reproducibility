import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import micastyle as ms
HERE = Path(__file__).resolve().parents[2] / "results" / "figure_data"
OUT = Path(__file__).resolve().parent
full=pd.read_csv(HERE/"fig5_cost_and_scope.csv"); df=full[full["k"]<=20]; e=full[full["k"]>20]
RED="#c0392b"
v=df[df.policy=="MICA"].sort_values("k"); c=df[df.policy=="horizon control"].sort_values("k"); b=df[df.policy=="unfiltered base"].sort_values("k")
ve=pd.concat([v,e[e.policy=="MICA"]]).sort_values("k"); ce=pd.concat([c,e[e.policy=="horizon control"]]).sort_values("k"); be=pd.concat([b,e[e.policy=="unfiltered base"]]).sort_values("k")
ms.apply()
ms.plt.rcParams.update({"font.sans-serif":["Liberation Sans","Arial"],"font.size":7,"axes.labelsize":7,
  "xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6.2,"lines.markersize":3.6,"lines.linewidth":1.2,
  "savefig.pad_inches":0.02})
fig,(a,bx)=ms.plt.subplots(1,2,figsize=(3.5,1.8),gridspec_kw=dict(wspace=0.42))
a.plot(v["k"],v["median_decision_ms"],color=ms.NAVY,marker="o",label="MICA")
a.plot(c["k"],c["median_decision_ms"],color=RED,marker="s",label="horizon control")
a.set_yscale("log")
for k,x,y in zip(v["k"],v["median_decision_ms"],c["median_decision_ms"]):
    a.annotate(f"{x/y:.2f}×",xy=(k,y),xytext=(4,-3),textcoords="offset points",fontsize=5.8,color="#4b5563",ha="left",va="top")
a.set_xlabel("graph degree $k$"); a.set_ylabel("median decision time (ms)")
a.set_xticks(sorted(df["k"].unique())); a.set_ylim(a.get_ylim()[0]*0.55,a.get_ylim()[1]*1.6); a.set_xlim(3.5,24.5)
ms.ygrid(a); a.set_title("(a) cost",loc="left",fontsize=7.5,fontweight="bold",pad=3)
a.legend(loc="upper left",fontsize=5.9,handlelength=1.6,borderaxespad=0.1)
bx.axhspan(-0.018,0.012,color="#dbe9f5",zorder=0)
bx.plot(be["k"],be["irrecoverability"],color="#8a94a0",marker="^",label="unfiltered base")
bx.plot(ce["k"],ce["irrecoverability"],color=RED,marker="s",label="horizon control")
bx.plot(ve["k"],ve["irrecoverability"],color=ms.NAVY,marker="o",label="MICA")
bx.set_xlabel("graph degree $k$"); bx.set_ylabel("group-trip irrecoverability")
bx.set_xticks(sorted(ve["k"].unique())); bx.set_ylim(-0.03,0.30)
ms.ygrid(bx); bx.set_title("(b) scope",loc="left",fontsize=7.5,fontweight="bold",pad=3)
bx.legend(loc="upper right",fontsize=5.9,handlelength=1.6,borderaxespad=0.1)
bx.text(0.5,0.30,"both horizon-reasoning policies\nhold at zero through $k=30$",transform=bx.transAxes,ha="center",va="bottom",fontsize=5.7,color=ms.NAVY,bbox=dict(fc="white",ec="none",pad=0.5))
for ext_ in ("png","pdf"): fig.savefig(OUT / f"fig5.{ext_}",dpi=600,bbox_inches="tight")
