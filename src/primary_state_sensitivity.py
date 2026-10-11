"""Matched ET feature-sufficiency check with eight explicitly reported process/context variables.
No target-derived variables, graph-estimated labels or default physical histories.
"""
from pathlib import Path
import json,sys,time
import pandas as pd,numpy as np
from sklearn.preprocessing import StandardScaler
import review_v4 as v
from traceable_validation import split
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/traceable'
EXTRA=['prior_natural_aging_days','reported_HE_strain','reported_sheet_thickness_mm','first_solution_cycle_hold_C','first_solution_cycle_hold_h','solution_heating_rate_C_per_h','initial_forging_temp_C','grain_coarsening_resolution']
def prepare():
 d=pd.read_csv(ROOT/'data/primary_traceable.csv').fillna({'aging_stages':'','special_process':''})
 for c in EXTRA:d[c]=np.nan
 d['extra_state_evidence']='Unreported extra states remain missing'
 for i,r in d.iterrows():
  sid=r.source_id;cond=r.original_condition
  if sid=='TAI2022_NA':
   d.at[i,'reported_sheet_thickness_mm']=2.
   if cond.startswith('NA-'):d.at[i,'prior_natural_aging_days']=float(cond.split('d/')[0][3:])
   d.at[i,'extra_state_evidence']='Tables 1–2 and Methods: 2 mm sheet; explicitly assigned 0/7-day prior natural aging; as-received prior natural aging unspecified'
  if sid=='BI2024_7A09':
   d.at[i,'reported_sheet_thickness_mm']=r.thickness_mm
   d.at[i,'extra_state_evidence']='Methods Section 3.2 and Tables 2–4: nominal 3/4 mm experimental sheets; initial Table 2 thickness-category values linked to the same two gauges'
  if sid=='LIU2020_SC':
   d.at[i,'reported_sheet_thickness_mm']=3.
   d.at[i,'extra_state_evidence']='Methods p.726: hot rolled from 10 to 3 mm before solution and aging'
  if sid=='ORLOWSKA2022_HE':
   d.at[i,'reported_HE_strain']=1.85 if cond.startswith('HE8') else 1.41
   d.at[i,'grain_coarsening_resolution']=1. if cond=='CG T6' else 0.
   if cond in ['HE10','HE8']:d.at[i,'prior_natural_aging_days']=180.
   d.at[i,'extra_state_evidence']='Methods and Table 2: HE10/HE8 true strains 1.41/1.85; 180-day natural-aging states; CG reference has an explicitly additional grain-coarsening solution treatment. Strain is applied history, not residual stored strain.'
  stages=json.loads(r.solution_stages) if isinstance(r.solution_stages,str) and r.solution_stages else []
  if stages:d.loc[i,['first_solution_cycle_hold_C','first_solution_cycle_hold_h']]=stages[0]
  if sid in ['XU2015','XU2016']:
   d.loc[i,['first_solution_cycle_hold_C','first_solution_cycle_hold_h']]=[450,2]
   d.at[i,'extra_state_evidence']='Primary Methods: ordered solution cycle 450 C/2 h +460 C/2 h +470 C/2 h'
  if sid=='XU2018':
   d.loc[i,['first_solution_cycle_hold_C','first_solution_cycle_hold_h']]=[400,2]
   d.at[i,'solution_heating_rate_C_per_h']=3.6 if r.record_id.endswith('01') else 180.
   d.at[i,'extra_state_evidence']='Primary Table 4/Methods: initial 400 C/2 h prehold, then 3.6 or 180 C/h ramp to solution temperature'
  if sid=='PMC9571267' and r.record_id!='PMC-01':
   d.loc[i,['first_solution_cycle_hold_C','first_solution_cycle_hold_h']]=[475,4]
   d.at[i,'initial_forging_temp_C']={'PMC-02':450.,'PMC-03':250.,'PMC-04':-180.}[r.record_id]
   d.at[i,'extra_state_evidence']='Primary Methods and Fig.1: pre-solution forging initial temperatures 450/250/-180 C; post-forging solution 475 C/4 h'
  if sid=='LI2020':d.loc[i,['first_solution_cycle_hold_C','first_solution_cycle_hold_h']]=[465,4]
 d.to_csv(ROOT/'data/primary_state_enriched.csv',index=False)
 return d
def main():
 begin=time.monotonic();d=prepare();x=pd.concat([v.features(d),d[EXTRA]],axis=1);assert x.shape==(98,53)
 base=v.features(d);keys=pd.util.hash_pandas_object(base,index=False);richkeys=pd.util.hash_pandas_object(x,index=False)
 alias_summary=[]
 for k,g in d.groupby(keys):
  if len(g)<2:continue
  alias_summary.append({'base_input_hash':str(k),'n_records':len(g),'records':json.dumps(g.record_id.tolist()),'source_ids':json.dumps(sorted(g.source_id.unique())),'n_enriched_vectors':richkeys.loc[g.index].nunique(),'YS_span_MPa':g.YS_MPa.max()-g.YS_MPa.min(),'UTS_span_MPa':g.UTS_MPa.max()-g.UTS_MPa.min(),'EL_span_pct':g.EL_pct.max()-g.EL_pct.min()})
 pd.DataFrame(alias_summary).to_csv(OUT/'primary_alias_summary.csv',index=False)
 rec=[];fold=[]
 for protocol in ['random','study_grouped','study_loso']:
  for seed in [42] if protocol=='study_loso' else [42,1,2,3,4]:
   for fid,tr,te in split(d,protocol,seed):
    for target in v.TARGET:
     train=tr[d.loc[tr,target].notna()];test=te[d.loc[te,target].notna()]
     if not len(test):continue
     if protocol!='random':assert set(d.loc[train,'study_group']).isdisjoint(d.loc[test,'study_group'])
     med,sx,empty=v.fit_preprocess(x.loc[train]);a=v.transform(x.loc[train],med,sx);b=v.transform(x.loc[test],med,sx)
     sy=StandardScaler().fit(d.loc[train,[target]]);y=sy.transform(d.loc[train,[target]]).ravel();model=v.model_factory()['ET']();model.fit(a,y)
     pred=sy.inverse_transform(model.predict(b).reshape(-1,1)).ravel()
     for i,p in zip(test,pred):rec.append({'protocol':protocol,'seed':seed,'fold':fid,'target':target,'model':'ET','n_inputs':53,'record_id':d.at[i,'record_id'],'observed':d.at[i,target],'predicted':float(p)})
     fold.append({'protocol':protocol,'seed':seed,'fold':fid,'target':target,'training_ids':json.dumps(d.loc[train,'record_id'].tolist()),'testing_ids':json.dumps(d.loc[test,'record_id'].tolist()),'entirely_missing_training_fields':json.dumps(empty)})
   print('state sensitivity',protocol,seed,'complete',round(time.monotonic()-begin,1),'s',flush=True)
 p=pd.DataFrame(rec);p.to_csv(OUT/'primary_state_predictions.csv',index=False)
 pd.DataFrame(fold).to_csv(OUT/'primary_state_folds.csv',index=False)
 metrics=[]
 for k,g in p.groupby(['protocol','seed','target']):metrics.append(dict(zip(['protocol','seed','target'],k),**v.metric(g.observed.to_numpy(),g.predicted.to_numpy())))
 m=pd.DataFrame(metrics);m.to_csv(OUT/'primary_state_metrics.csv',index=False)
 (OUT/'primary_state_metadata.json').write_text(json.dumps({'model':'fixed ET','base_inputs':45,'enriched_inputs':53,'extra_fields':EXTRA,'not_observed':d[EXTRA].isna().sum().to_dict(),'seconds':time.monotonic()-begin,'purpose':'Matched descriptive feature-sufficiency check, not causal coefficients or universal engineering validation','independent_data':False},indent=2))
 print('STATE_DONE',len(p),'predictions',flush=True)
if __name__=='__main__':main()
