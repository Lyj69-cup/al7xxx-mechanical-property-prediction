"""V5 additions: primary evidence, series proxies, nested selection and external challenge.

No external challenge labels enter fitting or selection. Historical 188-record fits
are retained only after an exact input-matrix comparison with the reviewed archive.
"""
from pathlib import Path
import argparse, json, hashlib, itertools, warnings, time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
import review_v4 as v

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'v5'
TARGET=v.TARGET
SEEDS=[42,1,2,3,4]

def build():
    OUT.mkdir(parents=True,exist_ok=True)
    d=pd.read_csv(ROOT/'results'/'dataset_reviewed.csv').fillna({'aging_stages':''})
    x=v.features(d)
    old=pd.read_csv(ROOT/'results'/'model_matrix_unimputed.csv')
    assert list(x.columns)==[c for c in old if c in x.columns]
    assert np.allclose(x.to_numpy(),old[x.columns].to_numpy(),equal_nan=True)
    assert d.record_id.tolist()==old.record_id.tolist()
    # Source evidence corrections alter archival context, not fitted 45-column inputs.
    c15='Tan C, Xu X-J, Jiang W, Sun L-S, Zhang X-L, Fan Y-Z, Zhao J-J. Effect of pre-recovery treatment on microstructure and properties of ultra high strength aluminum alloy extrusion treated by solid solution-T652 treatment. Chin J Nonferrous Met, 2015, 25(11):3019-3025.'
    c16='Zhang X-L, Xu X-J, Ling Z-Y, Jiang W, Tan C, Zhao J-J. Effect of pre-deformation on microstructure and properties of ultra-high strength aluminum alloy extrusion materials. Chin J Nonferrous Met, 2016, 26(3):507-515.'
    d.loc[d.source_id=='XU2015','citation']=c15
    d.loc[d.source_id=='XU2016','citation']=c16
    d.loc[d.source_id=='XU2018','citation']=d.loc[d.source_id=='XU2018','citation'].str.replace('24-27','24-28',regex=False)
    d.loc[d.source_id=='PMC9571267','citation']='W. Zhang, Y. Yi, S. Huang, H. He, and F. Dong, Effects of deformation at high, medium, and cryogenic temperatures on the microstructures and mechanical properties of Al–Zn–Mg–Cu alloys, Materials, 2022, 15(19), 6955. DOI 10.3390/ma15196955.'
    d.loc[d.source_id=='XU2016','special_process']=['multi-stage solution (450/2+460/2+470/2); water quench; T6 121/24','multi-stage solution (450/2+460/2+470/2); water quench; pre-deformation 2-3% (nominal midpoint 2.5%); T6 121/24']
    assert np.allclose(x.to_numpy(),v.features(d).to_numpy(),equal_nan=True)
    d['evidence_tier']=np.where(d.source_id=='BHAT2023','secondary public corpus','primary article checked')
    d['measurement_id']=np.where(d.source_id=='BHAT2023','unavailable',d.record_id)
    d.loc[d.source_id=='BHAT2023','data_origin']='public-corpus property value; original measurement and extraction type not independently authenticated'
    d.loc[d.source_id!='BHAT2023','data_origin']='published experimental summary checked against primary text/table'
    d['pre_deformation_min_pct']=d.pre_deformation_pct
    d['pre_deformation_max_pct']=d.pre_deformation_pct
    d.loc[d.record_id=='XU16-02','pre_deformation_min_pct']=2.
    d.loc[d.record_id=='XU16-02','pre_deformation_max_pct']=3.
    d['source_line']=d.source_id.replace({'XU2015':'XU_LINE','XU2016':'XU_LINE','XU2018':'XU_LINE'})
    d['tensile_standard']='not reported in archived corpus'
    d.loc[d.source_id.isin(['XU2015','XU2016']),'tensile_standard']='GB/T 228-2002'
    # Conservative connected components: nearby major composition OR article/research line.
    a=d[['Zn','Mg','Cu']].to_numpy(float); n=len(d); parent=list(range(n))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(i,j):
        ri,rj=root(i),root(j)
        if ri!=rj:parent[rj]=ri
    for i in range(n):
        for j in range(i):
            nearby=np.all(np.abs(a[i]-a[j])<=np.array([.2,.1,.1])+1e-10)
            same_line=d.source_id[i]!='BHAT2023' and d.source_line[i]==d.source_line[j]
            if nearby or same_line:union(i,j)
    labels={}
    d['series_proxy']=[labels.setdefault(root(i),f'G{len(labels)+1:03d}') for i in range(n)]
    d.to_csv(OUT/'dataset_v5.csv',index=False)
    pd.concat([d[['record_id','source_id','evidence_tier','series_proxy']],v.features(d),d[TARGET]],axis=1).to_csv(OUT/'model_inputs_v5.csv',index=False)
    d[['record_id','source_id','source_line','series_proxy','Zn','Mg','Cu']].to_csv(OUT/'series_proxy_membership.csv',index=False)
    evidence=[
      ['XU2015','XU15-01/XU15-02','Table 1 p3020; Methods p3020; Tables 3 and 5 pp3023-3024','720.5/699.1 MPa; EL5.0/4.2; 2% compression; 450/2+460/2+470/2; 121/24','Primary PDF visually checked; no target correction'],
      ['XU2016','XU16-01/XU16-02','Table 1 p508; Methods pp507-508; Table 2 p510; Table 5 p512','YS667.8/682.3; UTS693.6/738.6; EL4.0/5.0; 2-3% deformation','460 C/2h middle solution stage added to context; 2.5% is midpoint, not measured exact value'],
      ['XU2018','XU18-01/XU18-02','Table 4; DOI10.11973/jxgccl201808005','Checked in preceding audited revision','Original page range24-28; source route retained'],
      ['LI2020','LI-001 through LI-012','Table 1 and Methods; DOI10.1038/s43246-020-00074-2','Nominal compositions, Ce and three RRA holds','Primary checked; UTS-only summaries'],
      ['PMC9571267','PMC-01 through PMC-04','Methods and Section3.3; DOI10.3390/ma15196955','Final hold165 C/4h; three forging temperatures archived','Primary checked; manufacturing temperature not encoded in flat45 inputs'],
      ['BHAT2023','166 retained BH IDs','Mendeley DOI10.17632/b6br4yk6r3.1; raw row indices','Secondary public corpus; nominal processing class only','Original studies/specimen IDs and extraction types unavailable; no invented assignment']]
    pd.DataFrame(evidence,columns=['source_id','records','locator','checked_facts','qualification']).to_csv(OUT/'evidence_ledger.csv',index=False)
    settings={'n_records':n,'target_counts':{t:int(d[t].notna().sum()) for t in TARGET},'n_series_proxy':d.series_proxy.nunique(),'group_sizes':d.series_proxy.value_counts().to_dict(),'group_rule':'Connected components: |delta Zn|<=0.2, |delta Mg|<=0.1, |delta Cu|<=0.1 wt.% OR shared primary article/research line; transitive closure; no targets used. Does not authenticate unknown experiments.','fitting':'Target-specific train-only median and scaling, never label imputation.','inherited_fits':'Original 188x45 input matrix and property values unchanged; exact comparison passed.','external':'All six Yin2025 Table1 condition/direction means, selected by availability of post-2023 primary tensile data, withheld from all selection and fitting. No cherry-picking of directions.','selection':'Nested choice among raw/augmented ET, RF, SVR and Ridge configurations; this is a retrospective robustness check, not certification of historical choices.'}
    (OUT/'analysis_settings_v5.json').write_text(json.dumps(settings,indent=2))
    return d

def group_splits(d,seed,n_splits=5):
    groups=np.array(sorted(d.series_proxy.unique()))
    k=min(n_splits,len(groups))
    if k<2:raise ValueError('Too few composition/series groups')
    return [(str(i),np.flatnonzero(d.series_proxy.isin(groups[a])),np.flatnonzero(d.series_proxy.isin(groups[b]))) for i,(a,b) in enumerate(KFold(k,shuffle=True,random_state=seed).split(groups),1)]

def group_run(d,kind,seed):
    orig=v.splits
    def splitter(data,protocol,s):
        return group_splits(data,s) if protocol=='series_proxy' else orig(data,protocol,s)
    v.splits=splitter
    try:return v.run(d,kind,'series_proxy',seed)
    finally:v.splits=orig

def candidates():
    out={}
    for mf,leaf in [(.5,1),(1.,1),(.5,3),(1.,3)]:
        out[f'ET_mf{mf}_leaf{leaf}']=lambda mf=mf,leaf=leaf:ExtraTreesRegressor(n_estimators=300,max_features=mf,min_samples_leaf=leaf,random_state=42,n_jobs=1)
    for leaf in [1,3]:
        out[f'RF_leaf{leaf}']=lambda leaf=leaf:RandomForestRegressor(n_estimators=300,max_features=.5,min_samples_leaf=leaf,random_state=42,n_jobs=1)
    for alpha in [1.,10.,100.]:out[f'Ridge_{alpha}']=lambda alpha=alpha:Ridge(alpha=alpha)
    for c in [1.,10.,100.]:out[f'SVR_{c}']=lambda c=c:SVR(C=c,epsilon=.05,gamma='scale')
    return out

def fit_predict(d,train,test,target,kind,maker):
    x=v.features(d,kind);med,sx,empty=v.fit_preprocess(x.iloc[train]);a=v.transform(x.iloc[train],med,sx);b=v.transform(x.iloc[test],med,sx)
    sy=StandardScaler().fit(d.iloc[train][[target]]);y=sy.transform(d.iloc[train][[target]]).ravel()
    model=maker();model.fit(a,y)
    return sy.inverse_transform(model.predict(b).reshape(-1,1)).ravel()

def nested_one(d,protocol,seed):
    outer=group_splits(d,seed) if protocol=='series_proxy' else v.splits(d,'loso',42)
    makers=candidates();preds=[];choices=[];logs=[]
    for fold,tr,te in outer:
        for target in TARGET:
            train=tr[d.iloc[tr][target].notna().to_numpy()];test=te[d.iloc[te][target].notna().to_numpy()]
            if not len(test):continue
            local=d.iloc[train].reset_index(drop=True)
            inner=group_splits(local,314159,3);scores=[]
            for kind in ['raw','phys']:
                for name,maker in makers.items():
                    yy=[];pp=[]
                    for infold,it,iv in inner:
                        p=fit_predict(local,it,iv,target,kind,maker)
                        yy.extend(local.iloc[iv][target]);pp.extend(p)
                    score=mean_absolute_error(yy,pp);scores.append((score,kind,name))
                    logs.append({'protocol':protocol,'seed':seed,'fold':fold,'target':target,'kind':kind,'candidate':name,'inner_OOF_MAE':score,'outer_test_ids':json.dumps(d.iloc[test].record_id.tolist()),'inner_fit_ids':json.dumps(d.iloc[train].record_id.tolist())})
            score,kind,name=min(scores,key=lambda t:(t[0],t[1],t[2]))
            p=fit_predict(d,train,test,target,kind,makers[name])
            choices.append({'protocol':protocol,'seed':seed,'fold':fold,'target':target,'kind':kind,'selected':name,'inner_MAE':score,'train_ids':d.iloc[train].record_id.tolist(),'test_ids':d.iloc[test].record_id.tolist(),'inner_splits':[{'train_ids':local.iloc[it].record_id.tolist(),'test_ids':local.iloc[iv].record_id.tolist()} for _,it,iv in inner]})
            for idx,yhat in zip(test,p):preds.append({'kind':'nested','protocol':protocol,'seed':seed,'fold':fold,'target':target,'model':'NestedSelect','record_id':d.iloc[idx].record_id,'source_id':d.iloc[idx].source_id,'y_true':d.iloc[idx][target],'y_pred':yhat,'selected_kind':kind,'selected_model':name})
        print(f'completed nested {protocol} seed{seed} fold{fold}',flush=True)
    return preds,choices,logs

def external(d):
    rows=[]
    specs=[('H-MDF',[530,533,529],[556,556,556],[12.9,12.3,9.9],[2.9,6.1,1.4],[1,6.5,3.2],[.49,.41,.42]),('MC-MDF',[486,481,458],[563,562,548],[13.4,14.,12.2],[2.5,8.7,.7],[2.2,3.8,3.3],[.20,.87,.02])]
    for cond,ys,uts,el,ysd,utsd,eld in specs:
        for j,direction in enumerate(['L-T','T-L','S-T']):
            r={c:np.nan for c in d.columns};r.update(record_id=f'YIN25-{cond}-{direction}',source_id='YIN2025',source_line='YIN2025',product_form='forging',temper='over-aged (T74)',solution_temp_C=470.,solution_time_h=2.,aging_temp_C=160.,aging_time_h=10.,special_process=f'8-pass two-stage forging at {460 if cond=="H-MDF" else 340}/460 C; intermediate anneal460C/1h; source reports total strain2.08 and2.4 inconsistently; not pre-aging percentage; water quench; T74',aging_stages=json.dumps([[120.,7.],[160.,10.]]),Zn=7.28,Mg=1.49,Cu=1.68,Zr=.11,YS_MPa=ys[j],UTS_MPa=uts[j],EL_pct=el[j],YS_reported_spread=ysd[j],UTS_reported_spread=utsd[j],EL_reported_spread=eld[j],condition=cond,direction=direction,forging_stage1_temp_C=(460. if cond=='H-MDF' else 340.),forging_stage2_temp_C=460.,reported_total_strain_note='Methods reports2.08 and2.4; unresolved internal inconsistency; not encoded',tensile_speed_mm_min=2.,doi='10.3390/ma18020391',url='https://www.mdpi.com/1996-1944/18/2/391',localization='Table1; condition/direction mean of three tensile specimens',evidence_tier='post-archive primary challenge',data_origin='published experimental mean',grain_size_solution_CH_um=(39. if cond=='H-MDF' else 54.5),GND_aged_m2=(.31e14 if cond=='H-MDF' else .20e14),microstructure_locator='Section4.2; solution grain size at CH and aged GND; not independent directional measurements')
            rows.append(r)
    e=pd.DataFrame(rows);e.to_csv(OUT/'external_yin2025.csv',index=False)
    allx=pd.concat([d,e],ignore_index=True);p=[];params={}
    for target in TARGET:
        tr=np.flatnonzero(d[target].notna());te=np.arange(len(d),len(d)+len(e))
        for kind in ['raw','phys']:
            pp={}
            for name,maker in v.model_factory().items():
                pred=fit_predict(allx,tr,te,target,kind,maker);pp[name]=pred
                for idx,yhat in zip(te,pred):p.append({'kind':kind,'protocol':'external_yin2025','seed':42,'fold':'YIN2025','target':target,'model':name,'record_id':allx.iloc[idx].record_id,'source_id':'YIN2025','y_true':allx.iloc[idx][target],'y_pred':yhat})
            avg=np.mean([pp[m] for m in v.legacy.ENSEMBLE],axis=0)
            for idx,yhat in zip(te,avg):p.append({'kind':kind,'protocol':'external_yin2025','seed':42,'fold':'YIN2025','target':target,'model':'Fusion','record_id':allx.iloc[idx].record_id,'source_id':'YIN2025','y_true':allx.iloc[idx][target],'y_pred':yhat})
    # Explicit finite representation test: the six challenge means collapse to one vector.
    x=v.features(e)
    assert np.unique(x.fillna(-999999).to_numpy(),axis=0).shape[0]==1
    bound=[]
    for t in TARGET:
        y=e[t].to_numpy();bound.append({'target':t,'n_condition_direction_means':len(y),'n_encoded_inputs':1,'best_constant_mean':y.mean(),'minimum_in_sample_RMSE_for_this_encoding':np.std(y,ddof=0),'range':np.ptp(y),'meaning':'Representation diagnostic only. Not a population accuracy ceiling; fitted oracle uses these labels and is never presented as a predictor.'})
    pd.DataFrame(bound).to_csv(OUT/'representation_diagnostic.csv',index=False)
    return p

def stats_tables(p,meta):
    foldrows=[]
    for key,g in p.groupby(['kind','protocol','seed','fold','target','model']):foldrows.append(dict(zip(['kind','protocol','seed','fold','target','model'],key),**v.metric(g.y_true.to_numpy(),g.y_pred.to_numpy())))
    f=pd.DataFrame(foldrows);f.to_csv(OUT/'metrics_per_fold.csv',index=False)
    records=[]
    for target in TARGET:
        z=f[(f.target==target)&(f.protocol=='series_proxy')]
        tab=z.pivot(index=['seed','fold'],columns=['kind','model'],values='MAE')
        contrasts={'Features_on_ET':tab[('phys','ET')]-tab[('raw','ET')], 'Average_vs_ET_augmented':tab[('phys','Fusion')]-tab[('phys','ET')], 'MAE_interaction':tab[('phys','Fusion')]-tab[('phys','ET')]-tab[('raw','Fusion')]+tab[('raw','ET')]}
        ledger=[m for m in meta if m['kind']=='phys' and m['target']==target]
        ratio=np.mean([len(m['test_ids'])/len(m['train_ids']) for m in ledger])
        for name,a in contrasts.items():
            a=a.dropna().to_numpy();sd=a.std(ddof=1);se=np.sqrt((1/len(a)+ratio)*sd**2);df=len(a)-1
            tt=a.mean()/se if se>0 else np.inf;pv=2*stats.t.sf(abs(tt),df);radius=stats.t.ppf(.975,df)*se
            records.append({'target':target,'contrast':name,'n_matched_folds':len(a),'mean_MAE_difference':a.mean(),'SD_fold_differences':sd,'mean_test_train_ratio':ratio,'corrected_SE':se,'approx_95CI_low':a.mean()-radius,'approx_95CI_high':a.mean()+radius,'t':tt,'df':df,'p_unadjusted':pv,'qualification':'Exploratory corrected resampled t approximation; composition/series proxies are not certified independent experiments; no confirmatory superiority claim.'})
    st=pd.DataFrame(records);order=np.argsort(st.p_unadjusted);pv=st.p_unadjusted.to_numpy()[order];holm=np.maximum.accumulate(np.minimum(1,pv*(len(pv)-np.arange(len(pv)))));adj=np.empty(len(pv));adj[order]=holm;st['p_Holm_9_tests']=adj;st.to_csv(OUT/'paired_statistics.csv',index=False)

def supplemental_summaries():
    p=pd.read_csv(OUT/'primary_only_loso_predictions.csv');rows=[]
    for key,g in p.groupby(['kind','target']):rows.append({'kind':key[0],'target':key[1],**v.metric(g.y_true.to_numpy(),g.y_pred.to_numpy())})
    pd.DataFrame(rows).to_csv(OUT/'primary_only_metrics.csv',index=False)
    p=pd.read_csv(OUT/'metrics_per_run_v5.csv');rows=[]
    for (protocol,seed,target),g in p[p.protocol=='series_proxy'].groupby(['protocol','seed','target']):
        g=g.set_index(['kind','model'])
        for metric in ['R2','MAE']:
            aa,b,c,d=[g.loc[k,metric] for k in [('raw','ET'),('phys','ET'),('raw','Fusion'),('phys','Fusion')]]
            rows.append({'protocol':protocol,'seed':seed,'target':target,'metric':metric,'A_raw_ET':aa,'B_aug_ET':b,'C_raw_average':c,'D_aug_average':d,'feature_main_effect':((b-aa)+(d-c))/2,'average_main_effect':((c-aa)+(d-b))/2,'interaction':d-b-c+aa})
    pd.DataFrame(rows).to_csv(OUT/'factorial_series_v5.csv',index=False)

def run(workers=4):
    d=build()
    jobs=[(kind,s) for kind in ['raw','phys'] for s in SEEDS]
    results=Parallel(n_jobs=workers,backend='loky')(delayed(group_run)(d,k,s) for k,s in jobs)
    p=[r for out in results for r in out[0]];meta=[m for out in results for m in out[1]]
    pd.DataFrame(p).to_csv(OUT/'predictions_series_proxy.csv',index=False)
    (OUT/'fold_assignments_series_proxy.json').write_text(json.dumps(meta,indent=2))
    stats_tables(pd.DataFrame(p),meta)
    nested=Parallel(n_jobs=workers,backend='loky')(delayed(nested_one)(d,protocol,seed) for protocol,seed in [('series_proxy',s) for s in SEEDS]+[('loso',42)])
    npred=[r for out in nested for r in out[0]];nchoices=[r for out in nested for r in out[1]];logs=[r for out in nested for r in out[2]]
    pd.DataFrame(npred).to_csv(OUT/'predictions_nested.csv',index=False)
    pd.DataFrame(logs).to_csv(OUT/'inner_selection_scores.csv',index=False)
    (OUT/'nested_selection_ledger.json').write_text(json.dumps(nchoices,indent=2))
    ep=external(d);pd.DataFrame(ep).to_csv(OUT/'predictions_external.csv',index=False)
    total=pd.DataFrame(p+npred+ep)
    rows=[]
    for key,g in total.groupby(['kind','protocol','seed','target','model']):rows.append(dict(zip(['kind','protocol','seed','target','model'],key),**v.metric(g.y_true.to_numpy(),g.y_pred.to_numpy())))
    rr=pd.DataFrame(rows);rr.to_csv(OUT/'metrics_per_run_v5.csv',index=False)
    summ=rr.groupby(['kind','protocol','target','model'])[['R2','RMSE','MAE','bias_pred_minus_true']].agg(['mean','std']).reset_index();summ.columns=['_'.join(c).strip('_') for c in summ.columns];summ.to_csv(OUT/'metrics_summary_v5.csv',index=False)
    for source in ['primary_only']:
        art=d[d.evidence_tier=='primary article checked'].reset_index(drop=True)
        rs=[]
        for kind in ['raw','phys']:
            for model in ['ET']:
                old=v.model_factory;v.model_factory=lambda:{'ET':old()['ET']}
                try:out=v.run(art,kind,'loso',42,only_et=True)
                finally:v.model_factory=old
                rs+=out[0]
        pd.DataFrame(rs).to_csv(OUT/'primary_only_loso_predictions.csv',index=False)
    supplemental_summaries()
    print('V5 analysis completed',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--build-only',action='store_true');args=parser.parse_args()
    build() if args.build_only else run(args.workers)
