"""Bhat-free diagnostics. Frozen comparator settings; no outcome-driven selection.
Run from any directory: python src/traceable_validation.py
Outputs only results/traceable/; no acquisition scripts or pickles are executed.
"""
from pathlib import Path
import json,time,importlib.metadata,warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import review_v4 as v
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/traceable';OUT.mkdir(parents=True,exist_ok=True)
SEEDS=[42,1,2,3,4]
TARGETS=v.TARGET
def split(d,protocol,seed):
 if protocol=='random':return [(str(k),tr,te) for k,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(d),1)]
 groups=np.array(sorted(d.study_group.unique()))
 if protocol=='study_loso':return [(g,np.flatnonzero(d.study_group!=g),np.flatnonzero(d.study_group==g)) for g in groups]
 return [(str(k),np.flatnonzero(d.study_group.isin(groups[tr])),np.flatnonzero(d.study_group.isin(groups[te]))) for k,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(groups),1)]
def features_jmi(d,kind):
 cols=['Zn','Mg','Cu','Si','Zr','Mn','Ag','Li','Fe','Ti','Sn','Cr','Ge','Sc','Pr','Yb','Ce','Ni','Y','Er','Sr','Gd']
 x=d[cols].apply(pd.to_numeric,errors='coerce').copy()
 if kind=='augmented':
  x['Zn_Mg_ratio']=x.Zn/(x.Mg+.01);x['Zn_Mg_Cu_total']=x.Zn+x.Mg+x.Cu
  x['Mg_Cu_ratio']=x.Mg/(x.Cu+.01);x['Zn_Mg_product']=x.Zn*x.Mg
  x['Zr_Mn_Cr_total']=x[['Zr','Mn','Cr']].sum(axis=1,min_count=3)
 return x
def run(d,cohort,kind,protocol,seed):
 x=v.features(d,'phys' if kind=='augmented' else 'raw') if cohort=='primary' else features_jmi(d,kind)
 makers=v.model_factory();rec=[];fm=[]
 for fold,tr,te in split(d,protocol,seed):
  if protocol!='random':assert set(d.loc[tr,'study_group']).isdisjoint(d.loc[te,'study_group'])
  for target in TARGETS:
   train=tr[d.loc[tr,target].notna()];test=te[d.loc[te,target].notna()]
   if not len(test):continue
   assert len(train)>1
   med,sx,empty=v.fit_preprocess(x.loc[train]);a=v.transform(x.loc[train],med,sx);b=v.transform(x.loc[test],med,sx)
   sy=StandardScaler().fit(d.loc[train,[target]]);y=sy.transform(d.loc[train,[target]]).ravel()
   preds={}
   for name,maker in makers.items():
    with warnings.catch_warnings():
     warnings.simplefilter('ignore',UserWarning);m=maker();m.fit(a,y);p=sy.inverse_transform(m.predict(b).reshape(-1,1)).ravel()
    assert np.isfinite(p).all(),(name,target,fold)
    preds[name]=p
   preds['Avg']=np.mean([preds[n] for n in ['RF','ET','XGB','LGB','GBRT','SVR']],axis=0)
   for name,p in preds.items():
    for idx,pr in zip(test,p):
     rec.append({'cohort':cohort,'kind':kind,'protocol':protocol,'seed':seed,'fold':fold,'target':target,'model':name,'record_id':d.at[idx,'record_id'],'study_group':d.at[idx,'study_group'],'observed':d.at[idx,target],'predicted':float(pr)})
   fm.append({'cohort':cohort,'kind':kind,'protocol':protocol,'seed':seed,'fold':fold,'target':target,'training_ids':json.dumps(d.loc[train,'record_id'].tolist()),'testing_ids':json.dumps(d.loc[test,'record_id'].tolist()),'training_groups':json.dumps(sorted(d.loc[train,'study_group'].unique())),'testing_groups':json.dumps(sorted(d.loc[test,'study_group'].unique())),'n_train':len(train),'n_test':len(test),'entirely_missing_training_fields':json.dumps(empty)})
 return rec,fm
def main():
 begin=time.monotonic();primary=pd.read_csv(ROOT/'data/primary_traceable.csv').fillna({'aging_stages':'','special_process':''})
 jmi=pd.read_csv(ROOT/'data/jmi_published_compilation.csv');jmi=jmi[jmi.included_by_chemistry].reset_index(drop=True).rename(columns={'YS(MPa)':'YS_MPa','UTS(Mpa)':'UTS_MPa','EL(%)':'EL_pct','original_study_group':'study_group'})
 assert 'BHAT2023' not in set(primary.source_id)
 allrec=[];allfold=[]
 for cohort,d in [('primary',primary),('jmi_compilation',jmi)]:
  for kind in ['raw','augmented']:
   for protocol in ['random','study_grouped','study_loso']:
    for seed in [42] if protocol=='study_loso' else SEEDS:
     rec,fold=run(d,cohort,kind,protocol,seed);allrec.extend(rec);allfold.extend(fold)
     print(cohort,kind,protocol,seed,'completed',round(time.monotonic()-begin,1),'s',flush=True)
 predictions=pd.DataFrame(allrec);predictions.to_csv(OUT/'predictions.csv',index=False,lineterminator='\n')
 pd.DataFrame(allfold).to_csv(OUT/'fold_membership.csv',index=False,lineterminator='\n')
 by=['cohort','kind','protocol','seed','target','model']
 metrics=[]
 for key,g in predictions.groupby(by,sort=False):metrics.append(dict(zip(by,key),**v.metric(g.observed.to_numpy(),g.predicted.to_numpy())))
 md=pd.DataFrame(metrics);md.to_csv(OUT/'metrics.csv',index=False)
 summary=md.groupby([n for n in by if n!='seed'],sort=False).agg(repeats=('seed','size'),n=('n','first'),R2_mean=('R2','mean'),R2_sd=('R2','std'),RMSE_mean=('RMSE','mean'),MAE_mean=('MAE','mean'),MAE_sd=('MAE','std')).reset_index()
 summary.to_csv(OUT/'summary.csv',index=False)
 jsonparams={n:f().get_params() for n,f in v.model_factory().items()}
 (OUT/'run_metadata.json').write_text(json.dumps({'seconds':time.monotonic()-begin,'seeds':SEEDS,'model_parameters':jsonparams,'dependencies':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scikit-learn','scipy','xgboost','lightgbm']},'primary_n':len(primary),'jmi_n':len(jmi),'label_noise_added':False,'source_identifiers_in_features':False,'entire_pipeline_training_only':True,'analysis_scope':'Descriptive fixed-comparator sensitivity with real publication grouping; not new independent laboratory experiments or a fully nested performance-selection estimate; primary cohort overlaps the 22 old primary entries. JMI zeros are as published, not certified analytical absence.'},indent=2,default=str))
 print('DONE',len(predictions),'predictions',flush=True)
if __name__=='__main__':main()
