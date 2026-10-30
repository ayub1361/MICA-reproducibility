import pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
HERE = Path(__file__).resolve().parents[2] / 'results' / 'figure_data'
df=pd.read_csv(HERE/'figS1_contract_level.csv')
m=df[df.policy=='MICA'].sort_values('rho'); c=df[df.policy=='horizon max-min control'].sort_values('rho')
plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','Times New Roman'],'font.size':9,'font.weight':'bold',
 'axes.labelweight':'bold','axes.linewidth':1.0,'xtick.major.width':1.0,'ytick.major.width':1.0,'mathtext.fontset':'stix'})
fig,ax=plt.subplots(figsize=(3.4,2.3))
NAVY='#1a3a5c'; RED='#c0392b'
ax.plot(c.rho,c.intervention_rate,color=RED,marker='s',ms=6,lw=2.2,label='horizon max-min control')
ax.plot(m.rho,m.intervention_rate,color=NAVY,marker='o',ms=6,lw=2.2,label='MICA')
for r,a,b in zip(m.rho,m.intervention_rate,c.intervention_rate):
    ax.annotate(f'{b/a:.1f}×',xy=(min(r,0.94),(a+b)/2),ha='center',va='center',fontsize=8.5,fontweight='bold',color='#222',
                bbox=dict(fc='white',ec='#888',lw=0.6,pad=1.2,boxstyle='round,pad=0.2'))
ax.set_xlabel(r'contract level $\rho$'); ax.set_ylabel('intervention rate')
ax.set_xticks([0.7,0.8,0.9,0.95]); ax.set_xticklabels(['0.70','0.80','0.90','0.95'])
ax.set_ylim(-0.01,0.36); ax.set_xlim(0.68,0.975)
ax.grid(axis='y',color='#bbbbbb',lw=0.6); ax.spines[['top','right']].set_visible(False)
ax.legend(loc='upper left',fontsize=8,frameon=False,bbox_to_anchor=(0.0,0.82))
for t in ax.get_xticklabels()+ax.get_yticklabels(): t.set_fontweight('bold')
fig.tight_layout(pad=0.2); fig.savefig(Path(__file__).resolve().parent/'figS1.png',dpi=600)
