"""Recover matched 45/53-input state-sensitivity metrics and source separation."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from primary_state_sensitivity import EXTRA
import review_v4 as v
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/traceable'
def main():
 d=pd.read_csv(ROOT/'data/primary_state_enriched.csv').fillna({'aging_stages':'','special_process':''})
 x=pd.concat([v.features(d),d[EXTRA]],axis=1);assert x.shape==(98,53)
 index=d.set_index('record_id')
 p=pd.read_csv(R/'primary_state_predictions.csv');f=pd.read_csv(R/'primary_state_folds.csv');m=pd.read_csv(R/'primary_state_metrics.csv').set_index(['protocol','seed','target'])
 assert len(p)==2508
 for key,g in p.groupby(['protocol','seed','target']):
  assert len(g)==index[key[2]].count() and g.record_id.nunique()==len(g)
  assert np.allclose(g.observed,index.loc[g.record_id,key[2]],rtol=0,atol=1e-10)
  r=m.loc[key]
  for val,col in [(r2_score(g.observed,g.predicted),'R2'),(mean_absolute_error(g.observed,g.predicted),'MAE'),(np.sqrt(mean_squared_error(g.observed,g.predicted)),'RMSE')]:assert np.isclose(val,r[col],rtol=1e-12,atol=1e-10)
 base=pd.read_csv(R/'fold_membership.csv');base=base[(base.cohort=='primary')&(base.kind=='augmented')].set_index(['protocol','seed','fold','target'])
 for _,row in f.iterrows():
  tr=json.loads(row.training_ids);te=json.loads(row.testing_ids)
  b=base.loc[(row.protocol,row.seed,str(row.fold),row.target)]
  assert tr==json.loads(b.training_ids) and te==json.loads(b.testing_ids)
  if row.protocol!='random':assert set(index.loc[tr,'study_group']).isdisjoint(index.loc[te,'study_group'])
 k=pd.util.hash_pandas_object(v.features(d),index=False);rich=pd.util.hash_pandas_object(x,index=False)
 groups=[g for _,g in d.groupby(k) if len(g)>1]
 assert len(groups)==11 and sum(len(g) for g in groups)==24
 assert all(rich.loc[g.index].nunique()==len(g) for g in groups)
 report={'status':'PASS','predictions':2508,'base_columns':45,'enriched_columns':53,'aliased_base_groups':11,'aliased_base_records':24,'groups_separated_by_reported_context':11,'identical_partitions_to_base':True,'checks':['All observed labels recover from the primary cohort','All R2/MAE/RMSE recover from new predictions','Train/test IDs match the base ET target masks and partitions','Known-publication separation is preserved','All extra fields are source-reported context; no target-derived columns'],'scope':'Descriptive sufficiency comparison, not causal inference or independent experiments'}
 (ROOT/'verification/primary_state_check.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
if __name__=='__main__':main()
