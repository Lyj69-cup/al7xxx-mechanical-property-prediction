"""Color publication figures from archived predictions; no model fitting or selection."""
from pathlib import Path
from io import BytesIO
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap

BLUE='#0072B2'; ORANGE='#D55E00'; GREEN='#009E73'; PURPLE='#7B3294'
TARGET_COLORS=[BLUE,ORANGE,GREEN]
DENSITY=LinearSegmentedColormap.from_list('readable_blue',['#C3DDF3','#07549C'])

ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results';V=R/'v5';F=ROOT/'figures';F.mkdir(exist_ok=True)
TARGET=['YS_MPa','UTS_MPa','EL_pct'];NAMES=['Yield strength','Ultimate tensile strength','Elongation'];UNITS=['MPa','MPa','percentage points']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'axes.labelsize':9,'legend.fontsize':8})
def save(fig,name):
    dpi=1200 if name in ['Fig3_paired_comparisons','Fig5_external_challenge'] else 600
    for ext,resolution in [('png',220),('tiff',dpi),('pdf',None)]:
        buf=BytesIO();opts={'format':ext,'bbox_inches':'tight'}
        if resolution:opts['dpi']=resolution
        if ext=='tiff':opts['pil_kwargs']={'compression':'tiff_lzw'}
        fig.savefig(buf,**opts);raw=buf.getvalue()
        if ext!='pdf':
            with Image.open(BytesIO(raw)) as im:im.load()
        path=F/(name+'.'+ext);tmp=path.with_suffix(path.suffix+'.tmp')
        tmp.write_bytes(raw);tmp.replace(path)
    plt.close(fig)

def main():
    old=pd.read_csv(R/'predictions_long.csv');new=pd.read_csv(V/'predictions_series_proxy.csv')
    s=new[(new.kind=='phys')&(new.seed==42)&(new.model=='Fusion')]
    fig,axes=plt.subplots(2,3,figsize=(7.1,5.5),layout='constrained')
    for j,t in enumerate(TARGET):
        g=s[s.target==t];a=axes[0,j];a.hexbin(g.y_true,g.y_pred,gridsize=20,mincnt=1,cmap=DENSITY,linewidths=.2)
        lo=min(g.y_true.min(),g.y_pred.min());hi=max(g.y_true.max(),g.y_pred.max());a.plot([lo,hi],[lo,hi],'k--',lw=.8)
        unit='%' if j==2 else 'MPa'
        a.set(xlabel=f'Measured ({unit})',ylabel=f'Predicted ({unit})',title=f'({chr(97+j)}) {NAMES[j]}')
        a=axes[1,j]
        for p,label,style,color in [('input_grouped','Equal inputs','-',BLUE),('series_proxy','Composition/series','--',ORANGE),('loso','Collection LOSO',':',GREEN)]:
            h=s[s.target==t] if p=='series_proxy' else old[(old.kind=='phys')&(old.seed==42)&(old.model=='Fusion')&(old.target==t)&(old.protocol==p)]
            a.hist(h.y_pred-h.y_true,bins=16,histtype='step',density=True,label=label,linestyle=style,color=color,lw=1.2)
        a.axvline(0,color='.5',lw=.6);a.set(xlabel='Residual (EL points)' if j==2 else 'Residual (MPa)',ylabel='Density',title=f'({chr(100+j)}) Residual distributions')
    handles,labels=axes[1,2].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3,frameon=False,fontsize=8)
    save(fig,'Fig1_density_residuals')
    oldm=pd.read_csv(R/'metrics_summary.csv');m=pd.read_csv(V/'metrics_summary_v5.csv');allm=pd.concat([oldm,m])
    fig,axes=plt.subplots(1,3,figsize=(7.1,3.3),layout='constrained')
    labels=['Random','Equal inputs','Comp./series','Collection'];protocols=['random','input_grouped','series_proxy','loso']
    for j,t in enumerate(TARGET):
        a=axes[j]
        for model,offset,color,label in [('ET',-.15,BLUE,'ExtraTrees'),('Fusion',.15,ORANGE,'Six-model average')]:
            g=allm[(allm.kind=='phys')&(allm.target==t)&(allm.model==model)].set_index('protocol')
            vals=[g.loc[p,'R2_mean'] for p in protocols];err=[g.loc[p,'R2_std'] if pd.notna(g.loc[p,'R2_std']) else 0 for p in protocols]
            a.bar(np.arange(4)+offset,vals,width=.28,color=color,edgecolor='black',lw=.4,yerr=err,error_kw={'lw':.7,'capsize':2},label=label)
        a.axhline(0,color='black',lw=.6);a.set(xticks=range(4),xticklabels=labels,ylabel='Out-of-fold R²',title=f'({chr(97+j)}) {NAMES[j]}');a.tick_params(axis='x',rotation=35)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    save(fig,'Fig2_protocols')
    st=pd.read_csv(V/'paired_statistics.csv');fig,axes=plt.subplots(1,3,figsize=(7.1,3.3),layout='constrained')
    keys=['Features_on_ET','Average_vs_ET_augmented','MAE_interaction'];labs=['Features in ET','Average − ET','Interaction']
    for j,t in enumerate(TARGET):
        a=axes[j];g=st[st.target==t].set_index('contrast').loc[keys]
        a.errorbar(g.mean_MAE_difference,range(3),xerr=np.array([g.mean_MAE_difference-g.approx_95CI_low,g.approx_95CI_high-g.mean_MAE_difference]),fmt='o',color=TARGET_COLORS[j],capsize=3,lw=1)
        a.axvline(0,color='.5',linestyle='--',lw=.8);a.set(yticks=range(3),yticklabels=labs if j==0 else ['','',''],xlabel='ΔMAE (EL points)' if j==2 else 'ΔMAE (MPa)',title=f'({chr(97+j)}) {NAMES[j]}');a.invert_yaxis()
    save(fig,'Fig3_paired_comparisons')
    mdi=pd.read_csv(R/'impurity_importance_by_target.csv');fig,axes=plt.subplots(1,3,figsize=(7.1,4.2),layout='constrained')
    aliases={'tf_unknown':'Unknown temper','Zn_Mg_ratio':'Zn/Mg ratio','Zn_Mg_product':'Zn×Mg','tf_overaged':'Overaged','tf_artificial':'Artificial aging','tf_natural':'Natural aging','Zn_Mg_Cu_total':'Zn+Mg+Cu','solution_temp_time':'Solution T×t','aging_time_log':'log aging time','aging_time_h':'Final aging time','aging_temp_C':'Final aging T','Zn_aging_interaction':'Zn×aging T','alloy_agingtemp':'Solutes×aging T','Mg_solution_interaction':'Mg×solution T','ff_extrusion':'Extrusion','solution_temp_C':'Solution T'}
    for j,t in enumerate(TARGET):
        g=mdi[mdi.target==t].nlargest(9,'mean').iloc[::-1];a=axes[j]
        a.barh(range(len(g)),g['mean'],color=TARGET_COLORS[j],edgecolor='black',lw=.4,xerr=g['std'],error_kw={'lw':.6,'capsize':1.5})
        a.set(yticks=range(len(g)),yticklabels=[aliases.get(x,x) for x in g.feature],xlabel='Variance reduction',title=f'({chr(97+j)}) {["YS","UTS","EL"][j]}')
    save(fig,'Fig4_importance')
    e=pd.read_csv(V/'external_yin2025.csv');ep=pd.read_csv(V/'predictions_external.csv');fig,axes=plt.subplots(1,3,figsize=(7.1,3.6),layout='constrained')
    for j,t in enumerate(TARGET):
        a=axes[j];spread=['YS_reported_spread','UTS_reported_spread','EL_reported_spread'][j];oldspread=['YS_source_SD','UTS_source_SD','EL_source_SD'][j]
        for condition,ids,marker,color in [('H-MDF',range(3),'o',BLUE),('MC-MDF',range(3,6),'s',ORANGE)]:
            g=e.iloc[list(ids)];a.errorbar(list(ids),g[t],yerr=g[spread if spread in g else oldspread],fmt=marker,color=color,lw=.7,capsize=2,label='Published '+condition)
        for model,ls,color in [('ET','--',GREEN),('Fusion',':',PURPLE)]:
            val=ep[(ep.kind=='phys')&(ep.target==t)&(ep.model==model)].y_pred.iloc[0];a.axhline(val,color=color,ls=ls,lw=1.1,label='ET prediction' if model=='ET' else 'Average prediction')
        a.set(xticks=range(6),xticklabels=['LT','TL','ST','LT','TL','ST'],ylabel=f'{NAMES[j]} ({"%" if j==2 else "MPa"})',title=f'({chr(97+j)}) One input vector')
        a.axvline(2.5,color='.8',lw=.6)
    handles,labels=axes[2].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False,fontsize=8)
    save(fig,'Fig5_external_challenge')
    perm=pd.read_csv(R/'permutation_by_target.csv');fig,axes=plt.subplots(1,3,figsize=(7.1,4.0),layout='constrained')
    for j,t in enumerate(TARGET):
        g=perm[perm.target==t].sort_values('mean');a=axes[j];a.barh(range(len(g)),g['mean'],xerr=g['std'],color=TARGET_COLORS[j],edgecolor='black',lw=.4,error_kw={'lw':.7,'capsize':2})
        a.set(yticks=range(len(g)),yticklabels=[x.replace('Numeric processing','Processing') for x in g.family],xlabel='ΔMAE (EL points)' if j==2 else 'ΔMAE (MPa)',title=f'({chr(97+j)}) {["YS","UTS","EL"][j]}');a.axvline(0,color='black',lw=.6)
    save(fig,'FigS1_parent_permutation')
    print('Saved six figures: PNG previews, 600/1200 dpi TIFF, vector PDF')

if __name__=='__main__':main()
