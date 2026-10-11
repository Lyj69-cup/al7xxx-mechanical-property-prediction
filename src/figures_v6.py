"""Additional figure from the complete Wang2026 external table; no fitting."""
from pathlib import Path
import pandas as pd,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from figures_v5 import save, BLUE, ORANGE, GREEN, PURPLE
ROOT=Path(__file__).resolve().parents[1]
def main():
 e=pd.read_csv(ROOT/'results/v6/external_wang2026.csv');p=pd.read_csv(ROOT/'results/v6/predictions_external_wang2026.csv')
 fig,axes=plt.subplots(1,2,figsize=(7.1,3.6),layout='constrained')
 for j,t in enumerate(['YS_MPa','UTS_MPa']):
  ax=axes[j]
  for deg,marker,ls,color in [(0,'o','-',BLUE),(45,'s','--',ORANGE),(90,'^',':',GREEN)]:
   g=e[e.direction_deg==deg];x=g.aging_time_h.fillna(-6)
   ax.errorbar(x,g[t],yerr=g[t.replace('_MPa','_reported_spread')],fmt=marker,linestyle=ls,color=color,capsize=2,lw=.8,label=f'Published {deg}°')
  for model,marker,ls,color in [('ET','x','--','#222222'),('Fusion','+','-.',PURPLE)]:
   g=p[(p.kind=='phys')&(p.target==t)&(p.model==model)].merge(e[['record_id','aging_time_h']],on='record_id').drop_duplicates('aging_time_h').sort_values('aging_time_h',na_position='first')
   ax.plot(g.aging_time_h.fillna(-6),g.y_pred,marker=marker,color=color,linestyle=ls,lw=.8,label='ET prediction' if model=='ET' else 'Average prediction')
  ax.set(xticks=[-6,0,6,12,18,24,30],xticklabels=['AR','0','6','12','18','24','30'],xlabel='Artificial aging time (h)',ylabel=('Yield strength' if j==0 else 'Tensile strength')+' (MPa)',title=f'({chr(97+j)}) Wang2026 external challenge')
 handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
 save(fig,'FigS2_wang2026_challenge')
 print('Saved supplemental external-challenge figure')
if __name__=='__main__':main()
