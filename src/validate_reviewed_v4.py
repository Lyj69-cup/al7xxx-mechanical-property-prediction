"""Independent integrity checks on scientific labels, partitions and archived predictions."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import review_v4 as r
from predict_reviewed_v4 import validate_input

def main():
    out=r.OUT;d=pd.read_csv(out/'dataset_reviewed.csv');d['aging_stages']=d.aging_stages.fillna('')
    original=pd.read_csv(r.ROOT/'data'/'al7xxx_dataset_v4.csv');p=pd.read_csv(out/'predictions_long.csv')
    assert d.record_id.is_unique and len(d)==188
    assert d[r.TARGET].count().tolist()==[158,183,161]
    for t in r.TARGET:
        a=d.set_index('record_id')[t];b=original.set_index('record_id').loc[a.index,t]
        assert np.allclose(a,b,equal_nan=True,atol=0,rtol=0),f'Altered label {t}'
    ex=pd.read_csv(out/'duplicate_exclusions.csv');assert len(ex)==49 and set(ex.excluded_id).isdisjoint(d.record_id)
    assert set(ex.retained_id)<=set(d.record_id)
    raw=pd.read_csv(r.ROOT/'data'/'bhat_al_data.csv');raw=raw[(raw.Zn*100>=3)&(raw.Zn>raw.Cu)].reset_index(drop=True)
    cols=[c for c in raw if not c.startswith('Unnamed')]
    for e in ex.itertuples():
        removed=raw.loc[int(e.excluded_id.split('-')[1])-1,cols];kept=raw.loc[int(e.retained_id.split('-')[1])-1,cols]
        assert removed.equals(kept),'Not an exact full-raw-row duplicate'
    key=['kind','protocol','seed','target','model','record_id']
    assert not p.duplicated(key).any() and len(p)==121484
    assert np.isfinite(p[['y_true','y_pred']].to_numpy()).all()
    target_lookup=d.set_index('record_id')[r.TARGET]
    for t in r.TARGET:
        g=p[p.target==t];truth=target_lookup.loc[g.record_id,t].to_numpy()
        assert np.allclose(g.y_true,truth,atol=0,rtol=0)
    for keys,g in p.groupby(['kind','protocol','seed','target','model']):
        expected=set(d.loc[d[keys[3]].notna(),'record_id']);assert set(g.record_id)==expected
    mat=p.pivot(index=['kind','protocol','seed','target','record_id'],columns='model',values='y_pred')
    averaged=mat[r.legacy.ENSEMBLE].mean(axis=1)
    maximum=float(np.max(np.abs(mat.Fusion-averaged)))
    assert np.allclose(mat.Fusion,averaged,rtol=1e-12,atol=1e-10),'Not a six-component arithmetic mean'
    folds=json.loads((out/'fold_assignments.json').read_text());indexed=d.set_index('record_id')
    feature_hash=pd.Series(pd.util.hash_pandas_object(r.features(d),index=False).to_numpy(),index=d.record_id)
    for z in folds:
        tr=set(z['train_ids']);te=set(z['test_ids']);assert tr.isdisjoint(te)
        if z['protocol']=='loso':
            assert set(indexed.loc[list(te),'source_id'])=={z['fold']}
            assert z['fold'] not in set(indexed.loc[list(tr),'source_id'])
        if z['protocol']=='input_grouped':assert set(feature_hash.loc[list(tr)]).isdisjoint(feature_hash.loc[list(te)]),'Identical input group crossed fold boundary'
    scores=pd.read_csv(out/'metrics_per_run.csv').set_index(['kind','protocol','seed','target','model'])
    for keys,g in p.groupby(['kind','protocol','seed','target','model']):
        values=[r2_score(g.y_true,g.y_pred),np.sqrt(mean_squared_error(g.y_true,g.y_pred)),mean_absolute_error(g.y_true,g.y_pred)]
        assert np.allclose(scores.loc[keys,['R2','RMSE','MAE']].to_numpy(float),values,rtol=1e-10,atol=1e-10)
    assert r.metric(np.ones(2),np.zeros(2))['R2'] is None
    for bad in [pd.DataFrame([{'Mg':2,'Cu':1}]),pd.DataFrame([{'Zn':-1,'Mg':2,'Cu':1}]),pd.DataFrame([{'Zn':'bad','Mg':2,'Cu':1}]),pd.DataFrame([{'Zn':80,'Mg':20,'Cu':10}])]:
        try:validate_input(bad)
        except ValueError:pass
        else:raise AssertionError('Unsafe input accepted')
    good=validate_input(pd.DataFrame([{'Zn':5.6,'Mg':2.5,'Cu':1.6}]))
    assert good.Ce.isna().all() and good.solution_temp_C.isna().all(),'Unknown input became observed zero'
    for fields in [{'aging_temp_C':np.inf},{'aging_stages':'bad'},
                   {'aging_stages':'[[120,-1]]'},{'aging_stages':'[[null,24]]'},
                   {'aging_stages':'[120,24]'},{'aging_stages':'{}'}]:
        try:validate_input(pd.DataFrame([dict(Zn=5.6,Mg=2.5,Cu=1.6,**fields)]))
        except ValueError:pass
        else:raise AssertionError('Invalid process input accepted')
    validate_input(pd.DataFrame([{'Zn':5.6,'Mg':2.5,'Cu':1.6,
        'aging_stages':'[[120,24],[190,0.5],[120,24]]'}]))
    repeats=pd.read_csv(out/'permutation_all_repeats.csv')
    assert repeats.groupby(['target','fold','family']).repeat.nunique().eq(20).all()
    assert repeats.groupby(['target','family']).fold.nunique().eq(5).all()
    output={'status':'PASS','records':188,'target_counts':dict(zip(r.TARGET,[158,183,161])),
      'excluded_exact_raw_repeats':49,'prediction_rows':len(p),'target_specific_folds':len(folds),
      'maximum_fusion_mean_difference':maximum,'checks':['retained labels unchanged','exact full raw duplicates','prediction coverage and uniqueness','six-component mean','no train/test record overlap','no source overlap in LOSO','identical encoded inputs grouped together','metrics recovered from predictions','constant-target R2 undefined','malformed/unsafe prediction inputs rejected','unknown input fields retained as missing','20 family permutations across 5 held-out folds'],
      'scope':'Computational and archive integrity only; primary article checks are documented separately. Does not authenticate unknown original Bhat experiment identities.'}
    (out/'validation_report.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))

if __name__=='__main__':main()
