"""Numerical and partition audits for the supplemental revision. No model fitting."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import review_v4 as v

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'v5'

def main():
    d=pd.read_csv(OUT/'dataset_v5.csv').fillna({'aging_stages':''})
    old=pd.read_csv(ROOT/'results'/'dataset_reviewed.csv').fillna({'aging_stages':''})
    assert len(d)==188 and d.record_id.is_unique
    assert np.allclose(v.features(d),v.features(old),equal_nan=True,rtol=0,atol=0)
    assert np.allclose(d[v.TARGET],old[v.TARGET],equal_nan=True,rtol=0,atol=0)
    assert d[d.evidence_tier=='primary article checked'][v.TARGET].count().tolist()==[8,22,10]
    index=d.set_index('record_id'); checks=[]
    def separation(z,group=False):
        tr=set(z['train_ids']);te=set(z['test_ids'])
        assert tr.isdisjoint(te) and tr|te <= set(index.index)
        if group:assert set(index.loc[list(tr),'series_proxy']).isdisjoint(index.loc[list(te),'series_proxy'])
    folds=json.loads((OUT/'fold_assignments_series_proxy.json').read_text())
    for z in folds:separation(z,True)
    ledger=json.loads((OUT/'nested_selection_ledger.json').read_text())
    for z in ledger:
        separation(z,z['protocol']=='series_proxy')
        if z['protocol']=='loso':assert set(index.loc[z['train_ids'],'source_id']).isdisjoint(index.loc[z['test_ids'],'source_id'])
        inner_tests=[]
        for k in z['inner_splits']:
            separation(k,True)
            assert set(k['train_ids'])|set(k['test_ids'])==set(z['train_ids'])
            assert set(k['train_ids']+k['test_ids']).isdisjoint(z['test_ids'])
            inner_tests+=k['test_ids']
        assert len(inner_tests)==len(set(inner_tests))==len(z['train_ids'])
    scores=pd.read_csv(OUT/'inner_selection_scores.csv')
    for z in ledger:
        g=scores[(scores.protocol==z['protocol'])&(scores.seed==z['seed'])&(scores.fold.astype(str)==str(z['fold']))&(scores.target==z['target'])]
        assert len(g)==24
        best=g.sort_values(['inner_OOF_MAE','kind','candidate']).iloc[0]
        assert best.kind==z['kind'] and best.candidate==z['selected']
    allp=[]; maximum=0
    for name in ['predictions_series_proxy.csv','predictions_nested.csv','predictions_external.csv']:
        p=pd.read_csv(OUT/name);allp.append(p)
        assert not p.duplicated(['kind','protocol','seed','target','model','record_id']).any()
        assert np.isfinite(p[['y_true','y_pred']]).all().all()
        if name!='predictions_external.csv':
            for t in v.TARGET:
                g=p[p.target==t];assert np.allclose(g.y_true,index.loc[g.record_id,t],rtol=0,atol=0)
            for key,g in p.groupby(['kind','protocol','seed','target','model']):assert set(g.record_id)==set(d.loc[d[key[3]].notna(),'record_id'])
        if name!='predictions_nested.csv':
            m=p.pivot(index=['kind','protocol','seed','target','record_id'],columns='model',values='y_pred')
            err=np.max(np.abs(m.Fusion-m[v.legacy.ENSEMBLE].mean(axis=1)));maximum=max(maximum,float(err));assert err<1e-9
    p=pd.concat(allp);scores=pd.read_csv(OUT/'metrics_per_run_v5.csv').set_index(['kind','protocol','seed','target','model'])
    for key,g in p.groupby(['kind','protocol','seed','target','model']):
        rec=v.metric(g.y_true.values,g.y_pred.values)
        assert np.allclose(scores.loc[key,['R2','RMSE','MAE']], [rec[x] for x in ['R2','RMSE','MAE']],rtol=1e-9,atol=1e-9)
    e=pd.read_csv(OUT/'external_yin2025.csv').fillna({'aging_stages':''})
    assert set(e.record_id).isdisjoint(index.index) and len(e)==6
    assert len(v.features(e).fillna(-999999).drop_duplicates())==1
    for name in ['predictions_series_proxy.csv','predictions_nested.csv']:
        assert set(pd.read_csv(OUT/name).record_id).isdisjoint(e.record_id)
    for z in json.loads((OUT/'multitarget_folds.json').read_text()):separation(z,True)
    mp=pd.read_csv(OUT/'multitarget_predictions.csv');assert mp.groupby(['seed','model','target']).record_id.nunique().eq(140).all()
    st=pd.read_csv(OUT/'paired_statistics.csv');assert len(st)==9 and st.p_Holm_9_tests.between(0,1).all()
    report={'status':'PASS','records':188,'series_proxy_groups':int(d.series_proxy.nunique()),'primary_checked_records':22,'primary_label_counts':{'YS':8,'UTS':22,'EL':10},'external_condition_direction_means':6,'nested_outer_target_fits':len(ledger),'nested_candidate_scores':len(pd.read_csv(OUT/'inner_selection_scores.csv')),'maximum_average_difference':maximum,'checks':['45-column inputs and observed labels unchanged','group separation in outer and inner series partitions','source separation in nested LOSO','nested inner folds cover only outer training observations','24 candidates per outer fold and minimum inner MAE selection','prediction coverage, uniqueness and metric recovery','strict six-component averaging','all six external challenge means collapse to one vector','external challenge IDs excluded from fitting and selection','matched multi-output comparison uses 140 complete records'],'scope':'Computational integrity and primary article extraction; does not authenticate original measurement IDs or independence of the secondary Bhat corpus.'}
    (OUT/'validation_v5.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
