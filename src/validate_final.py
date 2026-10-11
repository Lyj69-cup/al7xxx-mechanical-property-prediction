"""Validate archived and added results without refitting or fabricating labels."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from scipy import stats
import review_v4 as v,advance_v5 as a,validate_v5,validate_reviewed_v4
ROOT=Path(__file__).resolve().parents[1]
def main():
 validate_v5.main()
 v.OUT=ROOT/'results';validate_reviewed_v4.main()
 e=pd.read_csv(ROOT/'results/v6/external_wang2026.csv').fillna({'aging_stages':''})
 d=pd.read_csv(ROOT/'results/v5/dataset_v5.csv').fillna({'aging_stages':''})
 assert len(e)==17 and e.EL_pct.isna().all() and set(e.record_id).isdisjoint(d.record_id)
 assert len(v.features(e).fillna(-999999).drop_duplicates())==7
 p=pd.read_csv(ROOT/'results/v6/predictions_external_wang2026.csv')
 assert len(p)==17*2*2*11 and not p.duplicated(['kind','target','model','record_id']).any()
 assert np.isfinite(p[['y_true','y_pred']]).all().all()
 m=pd.read_csv(ROOT/'results/v6/external_wang2026_metrics.csv').set_index(['kind','target','model'])
 for key,g in p.groupby(['kind','target','model']):
  assert len(g)==17 and set(g.record_id)==set(e.record_id)
  assert np.array_equal(g.y_true,e.set_index('record_id').loc[g.record_id,key[1]])
  rec=v.metric(g.y_true.to_numpy(),g.y_pred.to_numpy())
  assert np.allclose([m.loc[key,c] for c in ['R2','RMSE','MAE']],[rec[c] for c in ['R2','RMSE','MAE']],rtol=1e-10,atol=1e-10)
 piv=p.pivot(index=['kind','target','record_id'],columns='model',values='y_pred')
 assert np.allclose(piv.Fusion,piv[v.legacy.ENSEMBLE].mean(axis=1),rtol=1e-12,atol=1e-10)
 # Recover every paired statistic from predictions and target-specific fold sizes.
 sp=pd.read_csv(ROOT/'results/v5/predictions_series_proxy.csv');meta=json.loads((ROOT/'results/v5/fold_assignments_series_proxy.json').read_text())
 a.stats_tables(sp,meta)
 st=pd.read_csv(ROOT/'results/v5/paired_statistics.csv');assert len(st)==9 and (st.p_Holm_9_tests>.05).all()
 rep=pd.read_csv(ROOT/'results/v6/external_wang2026_source_replicates.csv')
 assert len(rep)==102 and rep.groupby(['record_id','target']).size().eq(3).all()
 for _,g in rep.groupby(['record_id','target']):
  assert np.allclose(g.arithmetic_mean_unrounded,g.value.mean(),atol=1e-10)
 q=rep.drop_duplicates(['record_id','target'])
 discrepancy=abs(q.reported_mean-q.arithmetic_mean_unrounded)
 assert discrepancy.max() <= 2/3+1e-10
 assert set(q.loc[discrepancy>.5,'record_id']) == {'WANG26-AR-45','WANG26-0-45','WANG26-30-45'}
 report={'status':'PASS','training_records':188,'training_labels':[158,183,161],'external_yin2025_means':6,'external_wang2026_means':17,'wang2026_individual_source_values':102,'wang2026_unique_encoded_vectors':7,'wang2026_EL_labels':0,'checks':['All archived metrics and fold isolation audited','Added means and all source replicates retained; missing cells stay missing','All 748 added fixed predictions finite and matched to published targets','No added challenge ID in training cohort','Six-component averaging and external metrics recovered','Nine corrected paired statistics recomputed from predictions'],'scope':'Computational integrity; no certification of unknown upstream experiments, causal mechanisms, or engineering qualification'}
 (ROOT/'verification/final_numerical_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
