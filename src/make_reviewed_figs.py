"""Figures from the reviewed, unrounded prediction archive; no raster-derived values."""
from pathlib import Path
from io import BytesIO
import gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import review_v4 as r

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
def save(fig,name,dpi=1200):
    fig.tight_layout()
    # Buffer before writing so TIFF directory offsets are finalized even on
    # filesystems whose direct libtiff seek/write behavior is unreliable.
    for suffix,fmt,res in [('png','png',200),('tif','tiff',dpi)]:
        buffer=BytesIO()
        options={'pil_kwargs':{'compression':'tiff_lzw'}} if fmt=='tiff' else {}
        fig.savefig(buffer,format=fmt,dpi=res,bbox_inches='tight',**options)
        data=buffer.getvalue()
        with Image.open(BytesIO(data)) as check:check.load()
        target=r.OUT/(name+'.'+suffix);temporary=target.with_suffix(target.suffix+'.tmp')
        temporary.write_bytes(data);temporary.replace(target)
    plt.close(fig)
    gc.collect()

def main():
    p=pd.read_csv(r.OUT/'predictions_long.csv');p=p[(p.kind=='phys')&(p.model=='Fusion')]
    labels=['YS (MPa)','UTS (MPa)','EL (%)']
    fig,axes=plt.subplots(2,3,figsize=(7.5,4.8))
    for i,t in enumerate(r.TARGET):
        g=p[(p.target==t)&(p.protocol=='input_grouped')&(p.seed==42)]
        a=axes[0,i];a.hexbin(g.y_true,g.y_pred,gridsize=20,mincnt=1,cmap='Greys',linewidths=.15)
        low=min(g.y_true.min(),g.y_pred.min());high=max(g.y_true.max(),g.y_pred.max());a.plot([low,high],[low,high],'--',color='.4',lw=.8)
        a.set(xlabel='Measured '+labels[i],ylabel='Predicted '+labels[i],title=f'({chr(97+i)}) Input-grouped; n={len(g)}')
        a=axes[1,i]
        for prot,name,style in [('random','Random (seed 42)','-'),('input_grouped','Input-grouped (seed 42)','--'),('loso','LOSO',':')]:
            z=p[(p.target==t)&(p.protocol==prot)&((p.seed==42)|(p.protocol=='loso'))]
            a.hist(z.y_pred-z.y_true,bins=18,density=True,histtype='step',label=name,linestyle=style,lw=1.1)
        a.axvline(0,color='.6',lw=.6);a.set(xlabel='Residual ('+('MPa' if i<2 else 'EL percentage points')+')',ylabel='Density',title=f'({chr(100+i)}) Residual distribution')
        if i==0:a.legend(fontsize=6.5,frameon=False)
    save(fig,'Fig1_reviewed_parity_residuals',600)
    m=pd.read_csv(r.OUT/'metrics_summary.csv');m=m[(m.kind=='phys')&(m.model=='Fusion')]
    fig,ax=plt.subplots(figsize=(7.2,3.3));xx=np.arange(3);width=.24
    for j,(prot,label,hatch) in enumerate([('random','Random five-fold',''),('input_grouped','Identical-input grouped','//'),('loso','Collection LOSO','xx')]):
        z=m[m.protocol==prot].set_index('target').loc[r.TARGET];err=z.R2_std.fillna(0).to_numpy()
        ax.bar(xx+(j-1)*width,z.R2_mean,width,yerr=err,capsize=2,label=label,color=['.8','.55','.25'][j],hatch=hatch,edgecolor='black',lw=.5)
    ax.set(xticks=xx,xticklabels=['Yield strength','Tensile strength','Elongation'],ylabel='Pooled $R^2$ (Fusion)')
    ax.axhline(0,color='black',lw=.6);ax.legend(fontsize=7,frameon=False);save(fig,'Fig2_reviewed_protocols')
    m=pd.read_csv(r.OUT/'loso_per_source.csv');m=m[(m.kind=='phys')&(m.model.isin(['ET','Fusion']))];sources=sorted(m.source_id.unique())
    fig,axes=plt.subplots(1,3,figsize=(7.5,3.8))
    for i,t in enumerate(r.TARGET):
        a=axes[i];z=m[m.target==t].pivot(index='source_id',columns='model',values='MAE').reindex(sources)
        counts=m[(m.target==t)&(m.model=='Fusion')].set_index('source_id').n.reindex(sources)
        for j,name in enumerate(['ET','Fusion']):a.bar(np.arange(len(sources))+(j-.5)*.34,z[name],.34,label=name,color=['.75','.3'][j],edgecolor='black',lw=.4)
        a.set(xticks=np.arange(len(sources)),xticklabels=[s.replace('PMC9571267','ZHANG2022')+'\nn='+('0' if pd.isna(n) else str(int(n))) for s,n in zip(sources,counts)],ylabel='MAE ('+('MPa' if i<2 else 'percentage points')+')',title=labels[i])
        a.tick_params(axis='x',labelrotation=75,labelsize=6.5)
        if i==0:a.legend(frameon=False,fontsize=7)
    save(fig,'Fig3_reviewed_source_errors')
    m=pd.read_csv(r.OUT/'impurity_importance_by_target.csv');fig,axes=plt.subplots(1,3,figsize=(7.5,4.3))
    for i,t in enumerate(r.TARGET):
        z=m[m.target==t].sort_values('mean').tail(10);a=axes[i];a.barh(z.feature,z['mean'],xerr=z['std'],capsize=1.5,color='.55',error_kw={'elinewidth':.6})
        a.set(xlabel='Impurity importance',title=labels[i]);a.tick_params(axis='y',labelsize=7)
    save(fig,'Fig4_reviewed_target_importance')
    m=pd.read_csv(r.OUT/'permutation_by_target.csv');fig,axes=plt.subplots(1,3,figsize=(7.5,3.4))
    for i,t in enumerate(r.TARGET):
        z=m[m.target==t].sort_values('mean');a=axes[i];a.barh(z.family,z['mean'],xerr=z['std'],capsize=2,color='.55',error_kw={'elinewidth':.6});a.axvline(0,lw=.6,color='black')
        a.set(xlabel=r'$\Delta$ MAE ('+('MPa' if i<2 else 'EL points')+')',title=labels[i]);a.tick_params(axis='y',labelsize=7)
    save(fig,'FigS1_reviewed_group_permutation')
    print('Saved 4 main figures and FigS1 as PNG and separate TIFF.')

if __name__=='__main__':main()
