"""Read-only recovery of metrics, source partitioning and labels in new cohorts."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results/traceable'
def main():
 primary=pd.read_csv(ROOT/'data/primary_traceable.csv')
 j=pd.read_csv(ROOT/'data/jmi_published_compilation.csv');j=j[j.included_by_chemistry].rename(columns={'YS(MPa)':'YS_MPa','UTS(Mpa)':'UTS_MPa','EL(%)':'EL_pct','original_study_group':'study_group'})
 assert len(primary)==98 and len(j)==135
 assert primary.source_id.nunique()==10 and primary.study_group.nunique()==8 and j.study_group.nunique()==44
 assert 'BHAT2023' not in set(primary.source_id)
 data={'primary':primary.set_index('record_id'),'jmi_compilation':j.set_index('record_id')}
 p=pd.read_csv(R/'predictions.csv');metrics=pd.read_csv(R/'metrics.csv');folds=pd.read_csv(R/'fold_membership.csv')
 assert len(p)==149072
 assert np.isfinite(p[['observed','predicted']].to_numpy()).all()
 for cohort,g in p.groupby('cohort'):
  d=data[cohort]
  for target,h in g.groupby('target'):
   assert np.allclose(d.loc[h.record_id,target].to_numpy(),h.observed.to_numpy(),rtol=0,atol=1e-10)
   assert (d.loc[h.record_id,'study_group'].to_numpy()==h.study_group.to_numpy()).all()
 max_error=0
 cols=['cohort','kind','protocol','seed','target','model']
 mi=metrics.set_index(cols)
 for key,g in p.groupby(cols,sort=False):
  d=data[key[0]];n=int(d[key[4]].count())
  assert len(g)==n and g.record_id.nunique()==n
  values={'n':len(g),'R2':r2_score(g.observed,g.predicted),'RMSE':np.sqrt(mean_squared_error(g.observed,g.predicted)),'MAE':mean_absolute_error(g.observed,g.predicted)}
  r=mi.loc[key]
  for name,value in values.items():
   assert np.isclose(r[name],value,rtol=2e-12,atol=1e-10),(key,name)
 foldcols=['cohort','kind','protocol','seed','fold','target']
 foldsets=p.assign(fold=p.fold.astype(str)).groupby(foldcols).record_id.agg(set)
 for _,f in folds.iterrows():
  d=data[f.cohort];tr=json.loads(f.training_ids);te=json.loads(f.testing_ids)
  assert set(tr).isdisjoint(te)
  assert d.loc[tr,f.target].notna().all() and d.loc[te,f.target].notna().all()
  if f.protocol!='random':assert set(d.loc[tr,'study_group']).isdisjoint(d.loc[te,'study_group'])
  assert foldsets.loc[(f.cohort,f.kind,f.protocol,f.seed,str(f.fold),f.target)]==set(te),(f.cohort,f.protocol,f.fold)
 pivot=p.pivot(index=['cohort','kind','protocol','seed','fold','target','record_id'],columns='model',values='predicted')
 average=pivot[['RF','ET','XGB','LGB','GBRT','SVR']].mean(axis=1)
 max_error=float(np.abs(average-pivot.Avg).max())
 assert np.allclose(average,pivot.Avg,rtol=1e-12,atol=1e-9)
 s=pd.read_csv(R/'summary.csv').set_index([x for x in cols if x!='seed'])
 for key,g in metrics.groupby([x for x in cols if x!='seed'],sort=False):
  r=s.loc[key]
  for name,col in [('R2','R2_mean'),('RMSE','RMSE_mean'),('MAE','MAE_mean')]:assert np.isclose(r[col],g[name].mean(),rtol=2e-12,atol=1e-10)
  assert np.isclose(r.R2_sd,g.R2.std(),equal_nan=True,rtol=1e-10,atol=1e-10)
 report={'status':'PASS','prediction_rows':len(p),'metric_rows':len(metrics),'fold_target_rows':len(folds),'primary_n':98,'primary_groups':8,'jmi_n':135,'jmi_groups':44,'maximum_six_component_average_difference':max_error,'checks':['All recorded target values match their supplied published-source cohort','Every partition predicts every available target once per model','No original-publication group overlaps a grouped/holdout training and test partition','All ten comparators and the fixed average retain finite predictions, including failures','All R2/RMSE/MAE and reported mean/sample SD recover from unrounded predictions'],'scope':'Numeric and source-partition consistency; does not certify unreported Bhat or compilation experiment identities.'}
 (ROOT/'verification/traceable_results_check.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print(json.dumps(report))
if __name__=='__main__':main()
