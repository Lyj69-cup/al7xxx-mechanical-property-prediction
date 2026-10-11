"""Render source-resolved Figure 6 and publication macro errors from supplied predictions."""
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import review_v4 as v
ROOT=Path(__file__).resolve().parents[1];PKG=ROOT;R=ROOT/'results/traceable'
p=pd.read_csv(R/'predictions.csv')
group_metrics=[]
keys=['cohort','kind','protocol','seed','target','model','study_group']
for k,g in p.groupby(keys,sort=False):group_metrics.append(dict(zip(keys,k),**v.metric(g.observed.to_numpy(),g.predicted.to_numpy())))
gm=pd.DataFrame(group_metrics);gm.to_csv(R/'publication_metrics.csv',index=False)
macro=gm.groupby(keys[:-1],sort=False).agg(n_groups=('study_group','size'),macro_MAE=('MAE','mean'),macro_RMSE=('RMSE','mean')).reset_index()
macro.to_csv(R/'publication_macro_errors.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(9,6.2),layout='constrained')
colors=['#0072B2','#D55E00'];targets=['YS_MPa','UTS_MPa','EL_pct']
for i,cohort in enumerate(['primary','jmi_compilation']):
 for j,target in enumerate(targets):
  ax=axes[i,j];g=p[(p.cohort==cohort)&(p.kind=='augmented')&(p.protocol=='study_grouped')&(p.seed==42)&(p.model=='ET')&(p.target==target)]
  ax.scatter(g.observed,g.predicted,s=18,color=colors[i],alpha=.65,edgecolors='none')
  lo=min(g.observed.min(),g.predicted.min());hi=max(g.observed.max(),g.predicted.max());pad=(hi-lo)*.04
  ax.plot([lo-pad,hi+pad],[lo-pad,hi+pad],color='#444444',ls='--',lw=1)
  ax.set_xlim(lo-pad,hi+pad);ax.set_ylim(lo-pad,hi+pad);ax.set_aspect('equal',adjustable='box')
  ax.set_xlabel('Reported EL (%)' if j==2 else 'Reported strength (MPa)');ax.set_ylabel('Predicted EL (%)' if j==2 else 'Predicted strength (MPa)')
  ax.set_title(('Primary' if i==0 else 'Compilation')+' — '+['YS','UTS','EL'][j])
  m=v.metric(g.observed.to_numpy(),g.predicted.to_numpy());ax.text(.04,.94,f'n = {len(g)}\nR² = {m["R2"]:.3f}\nMAE = {m["MAE"]:.2f}',transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',edgecolor='none',alpha=.8))
  ax.grid(alpha=.15)
F=PKG/'figures/Fig6_publication_grouped'
fig.savefig(str(F)+'.png',dpi=220)
fig.savefig(str(F)+'.pdf')
fig.savefig(str(F)+'.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
