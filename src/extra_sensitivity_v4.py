"""Limited-evidence sensitivities: certified source subset, Xu collections together,
and duplicate/split changes with the original v4 representation held fixed.
"""
from pathlib import Path
import json, hashlib, importlib.metadata
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import review_v4 as r

def main():
    d=r.prepare(); out=r.OUT; rows=[]; preds=[]
    settings={
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'review_script_sha256':hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest(),
      'legacy_script_sha256':hashlib.sha256(Path(r.legacy.__file__).read_bytes()).hexdigest(),
      'dataset_sha256':hashlib.sha256((out/'dataset_reviewed.csv').read_bytes()).hexdigest(),
      'partition_seed':42,
      'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','scipy','xgboost','lightgbm']}}
    sp=out/'extra_analysis_settings.json'
    if sp.exists() and json.loads(sp.read_text())!=settings:
        raise ValueError('Extra sensitivity cache settings differ; archive reviewed_v4 before rerunning')
    if not sp.exists() and list(out.glob('cache_extra_*.json')):
        raise ValueError('Unversioned extra caches exist; archive them before rerunning')
    sp.write_text(json.dumps(settings,indent=2))
    for label,subset in [('primary_verified_18',d[d.source_id.isin(['LI2020','PMC9571267','XU2018'])].copy()),('Xu_collections_together',d.copy())]:
        subset=subset.reset_index(drop=True)
        if label=='Xu_collections_together':subset.loc[subset.source_id.str.startswith('XU'),'source_id']='XU_GROUP'
        cache=out/f'cache_extra_{label}.json'
        if cache.exists():rec=json.loads(cache.read_text())
        else:
            print(label,'start',flush=True); rec=r.run(subset,'phys','loso',42)[0]
            cache.write_text(json.dumps(rec,allow_nan=False))
        p=pd.DataFrame(rec)
        for (target,model),g in p.groupby(['target','model']):rows.append({'analysis':label,'target':target,'model':model,**r.metric(g.y_true.to_numpy(),g.y_pred.to_numpy())})
        for z in rec:z['analysis']=label
        preds.extend(rec)
        print(label,'complete',flush=True)
    original=pd.read_csv(r.ROOT/'data'/'al7xxx_dataset_v4.csv')
    keep=set(d.record_id)
    dedup=original[original.record_id.isin(keep)].reset_index(drop=True)
    # Same estimators and original encoding in both evaluations.
    old_factory=r.legacy.models
    r.legacy.models=r.model_factory
    # Avoid recursive model_factory via its live legacy reference.
    def factory():
        f=old_factory()
        for name,maker in list(f.items()):
            def make(maker=maker):
                m=maker()
                if 'n_jobs' in m.get_params():m.set_params(n_jobs=1)
                return m
            f[name]=make
        return f
    r.legacy.models=factory
    for label,frame,grouped in [('legacy_237_input_grouped_seed42',original,True),('legacy_188_random_seed42',dedup,False)]:
        cache=out/f'cache_extra_{label}.json'
        if cache.exists():rec=json.loads(cache.read_text())
        else:
            if grouped:
                x,cols=r.legacy.features(frame,'phys');k=pd.util.hash_pandas_object(x[cols],index=False).to_numpy();unique=np.unique(k);fl=[]
                for tr,te in KFold(5,shuffle=True,random_state=42).split(unique):fl.append((np.flatnonzero(np.isin(k,unique[tr])),np.flatnonzero(np.isin(k,unique[te]))))
            else:fl=r.legacy.folds_random(len(frame),42)
            print(label,'start',flush=True);y=frame[r.TARGET].to_numpy(float);o,_=r.legacy.run_folds(frame,'phys',fl,y);rec=[]
            for model,p in o.items():
                for ti,target in enumerate(r.TARGET):
                    for i in np.flatnonzero(np.isfinite(y[:,ti])):rec.append({'analysis':label,'target':target,'model':model,'record_id':frame.at[i,'record_id'],'source_id':frame.at[i,'source_id'],'y_true':float(y[i,ti]),'y_pred':float(p[i,ti])})
            cache.write_text(json.dumps(rec,allow_nan=False))
        for (target,model),g in pd.DataFrame(rec).groupby(['target','model']):rows.append({'analysis':label,'target':target,'model':model,**r.metric(g.y_true.to_numpy(),g.y_pred.to_numpy())})
        preds.extend(rec);print(label,'complete',flush=True)
    pd.DataFrame(rows).to_csv(out/'extra_sensitivity_metrics.csv',index=False)
    pd.DataFrame(preds).to_csv(out/'extra_sensitivity_predictions.csv',index=False)
    (out/'extra_sensitivity_notes.json').write_text(json.dumps({
      'primary_verified_18':'12 Li Table 1, 4 Zhang Section 3.3, 2 Xu 2018 Table 4, all independently checked. This subset does not certify Bhat records and has only 6 YS, 18 UTS and 6 EL observations.',
      'Xu_collections_together':'Treat all three Xu collections as one research line; not an assertion of shared specimens.',
      'legacy_sensitivity':'Single partition seed 42, original 44 features, original preprocessing and six learners; no uncertainty estimate from one split. Composite changes in the main reviewed analysis are not attributed to one cause.'},indent=2))

if __name__=='__main__':main()
