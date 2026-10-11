"""Retrospective external challenge from Wang et al., Metals 2026, 16, 906.

DOI 10.3390/met16080906, Tables 1 and 2, Methods p3. Transcribed
published means are the test labels. Individual tests remain a source ledger;
they are never counted as independent model test conditions or training rows.
All 17 available condition/direction means are retained. Slash/blank cells
at 6 h and 24 h in 45/90 directions stay missing. No EL is digitized.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import review_v4 as v
import advance_v5 as a

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/v6';OUT.mkdir(parents=True,exist_ok=True)
DOI='10.3390/met16080906'
# Order: as received, SST with 0 h aging, 6, 12, 18, 24, 30 h.
STATES=['AR',0,6,12,18,24,30]
MEANS={
 0: {'UTS':[304,123,128,137,146,158,165], 'YS':[259,47,59,70,80,93,102],
     'UTS_sd':[3.74,2.65,2.52,1.63,3.56,4.04,2.65], 'YS_sd':[3.31,1.73,3.56,2.51,2.89,3.31,2.65]},
 45:{'UTS':[295,159,None,180,203,None,236], 'YS':[251,70,None,109,139,None,191],
     'UTS_sd':[2.45,1.73,None,1.73,4.20,None,3.32], 'YS_sd':[4.83,2.16,None,2.16,3.27,None,2.71]},
 90:{'UTS':[297,143,None,156,171,None,196], 'YS':[257,59,None,89,110,None,144],
     'UTS_sd':[1.41,2.08,None,2.52,1.29,None,1.73], 'YS_sd':[2.52,3.56,None,2.94,3.42,None,2.16]},
}
INDIVIDUAL={
 0:{'UTS':[[300,303,309],[122,127,121],[125,131,127],[139,135,137],[141,149,148],[152,161,160],[169,164,163]],
    'YS':[[255,263,258],[45,49,46],[54,62,61],[71,67,73],[83,80,76],[95,88,95],[100,106,101]]},
 45:{'UTS':[[298,292,295],[160,157,161],None,[178,182,181],[197,207,204],None,[232,235,240]],
     'YS':[[246,248,257],[73,71,68],None,[111,106,110],[143,135,139],None,[188,194,193]]},
 90:{'UTS':[[295,298,298],[146,143,141],None,[159,153,157],[171,169,172],None,[197,194,198]],
     'YS':[[254,260,256],[61,62,54],None,[85,92,90],[111,113,105],None,[147,142,143]]},
}

def build():
 rows=[];individual=[];missing=[]
 for direction,values in MEANS.items():
  for j,state in enumerate(STATES):
   if values['UTS'][j] is None:
    missing.append({'direction_deg':direction,'aging_time_h':state,'reason':'Slash in UTS block and blank in YS block; no interpolation'})
    continue
   r={'record_id':f'WANG26-{state}-{direction}','source_id':'WANG2026','source_line':'WANG2026','direction_deg':direction,'condition':'as received' if state=='AR' else f'SST + {state} h aging',
      'Zn':5.72,'Mg':2.48,'Cu':1.48,'Si':.37,'Mn':.15,'Ti':.1,'Cr':.16,'Fe':.5,
      'YS_MPa':values['YS'][j],'UTS_MPa':values['UTS'][j],'EL_pct':np.nan,
      'YS_reported_spread':values['YS_sd'][j],'UTS_reported_spread':values['UTS_sd'][j],
      'solution_temp_C':np.nan if state=='AR' else 490.,'solution_time_h':np.nan if state=='AR' else 2.,
      'aging_temp_C':140. if state!='AR' and state>0 else np.nan,'aging_time_h':np.nan if state=='AR' else state,
      'pre_deformation_pct':np.nan,'aging_stages':json.dumps([[140.,float(state)]]) if state!='AR' and state>0 else '',
      'temper':'as received (unspecified)' if state=='AR' else ('solution treated (no artificial hold)' if state==0 else 'artificially aged'),
      'product_form':'rolled sheet','special_process':'hot rolling at 400 C with 90% manufacturing reduction; water quench after SST' if state!='AR' else 'hot rolling at 400 C with 90% manufacturing reduction',
      'manufacturing_reduction_pct':90.,'manufacturing_temp_C':400.,'tensile_strain_rate_s-1':.1,'tensile_crosshead_mm_min':180.,'tensile_temperature':'room temperature',
      'doi':DOI,'url':'https://www.mdpi.com/2075-4701/16/8/906','localization':'Table 1 p3; Table 2 pp5-6; Methods p3',
      'data_origin':'published condition/direction mean of three tests','evidence_tier':'later primary external challenge',
      'qualification':'No numerical EL table. Rolled form and direction not explicit fitted categories; 90% rolling reduction is not a pre-aging strain. YS definition not assumed to be a common 0.2% proof stress.'}
   for c in v.COMP+v.PROC:
    r.setdefault(c,np.nan)
   rows.append(r)
   for target in ['YS','UTS']:
    nums=INDIVIDUAL[direction][target][j]
    for rep,num in enumerate(nums,1):individual.append({'record_id':r['record_id'],'target':target+'_MPa','source_experiment_no':rep,'value':num,'reported_mean':values[target][j],'reported_spread':values[target+'_sd'][j],'arithmetic_mean_unrounded':np.mean(nums),'SD_population_recalculated':np.std(nums,ddof=0),'SD_sample_recalculated':np.std(nums,ddof=1),'doi':DOI})
 e=pd.DataFrame(rows);e.to_csv(OUT/'external_wang2026.csv',index=False)
 pd.DataFrame(individual).to_csv(OUT/'external_wang2026_source_replicates.csv',index=False)
 pd.DataFrame(missing).to_csv(OUT/'external_wang2026_missing_cells.csv',index=False)
 assert len(e)==17 and e.EL_pct.isna().all()
 x=v.features(e)
 assert len(x.fillna(-999999).drop_duplicates())==7
 return e

def main():
 d=pd.read_csv(ROOT/'results/v5/dataset_v5.csv').fillna({'aging_stages':''});e=build()
 assert set(d.record_id).isdisjoint(e.record_id)
 allx=pd.concat([d,e],ignore_index=True);rows=[]
 for target in ['YS_MPa','UTS_MPa']:
  tr=np.flatnonzero(d[target].notna());te=np.arange(len(d),len(allx))
  for kind in ['raw','phys']:
   pp={}
   for name,maker in v.model_factory().items():pp[name]=a.fit_predict(allx,tr,te,target,kind,maker)
   pp['Fusion']=np.mean([pp[m] for m in v.legacy.ENSEMBLE],axis=0)
   for model,pred in pp.items():
    for idx,yp in zip(te,pred):rows.append({'kind':kind,'protocol':'external_wang2026','seed':42,'fold':'WANG2026','target':target,'model':model,'record_id':allx.iloc[idx].record_id,'source_id':'WANG2026','y_true':allx.iloc[idx][target],'y_pred':yp})
 p=pd.DataFrame(rows);p.to_csv(OUT/'predictions_external_wang2026.csv',index=False)
 metrics=[]
 for key,g in p.groupby(['kind','target','model']):metrics.append(dict(zip(['kind','target','model'],key),**v.metric(g.y_true.to_numpy(),g.y_pred.to_numpy())))
 pd.DataFrame(metrics).to_csv(OUT/'external_wang2026_metrics.csv',index=False)
 diagnostic=[]
 for condition,g in e.groupby('condition',sort=False):
  for t in ['YS_MPa','UTS_MPa']:
   diagnostic.append({'condition':condition,'target':t,'n_directions':len(g),'label_range':np.ptp(g[t]),'oracle_mean':g[t].mean(),'minimum_in_sample_RMSE':np.std(g[t],ddof=0),'qualification':'Label-dependent finite-sample diagnostic, never predictor accuracy'})
 pd.DataFrame(diagnostic).to_csv(OUT/'wang2026_encoding_diagnostics.csv',index=False)
 print(pd.DataFrame(metrics).query("kind=='phys' and model in ['ET','Fusion']").to_string(index=False))

if __name__=='__main__':main()
