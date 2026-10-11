"""Matched multi-output comparator, nominal strain range, and microstructure annotations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
import review_v4 as v
import advance_v5 as a

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'v5'

def main():
    d=pd.read_csv(OUT/'dataset_v5.csv').fillna({'aging_stages':''})
    complete=d.dropna(subset=v.TARGET).reset_index(drop=True)
    records=[];meta=[]
    x=v.features(complete)
    for seed in [42,1,2,3,4]:
        for fold,tr,te in a.group_splits(complete,seed):
            med,sx,empty=v.fit_preprocess(x.iloc[tr]);xx=v.transform(x.iloc[tr],med,sx);xt=v.transform(x.iloc[te],med,sx)
            sy=StandardScaler().fit(complete.iloc[tr][v.TARGET]);yy=sy.transform(complete.iloc[tr][v.TARGET])
            multi=ExtraTreesRegressor(n_estimators=500,max_features=.5,random_state=42,n_jobs=1).fit(xx,yy)
            pm=sy.inverse_transform(multi.predict(xt))
            for j,t in enumerate(v.TARGET):
                single=ExtraTreesRegressor(n_estimators=500,max_features=.5,random_state=42,n_jobs=1).fit(xx,yy[:,j]);ps=single.predict(xt)*sy.scale_[j]+sy.mean_[j]
                for model,p in [('ET_multitarget',pm[:,j]),('ET_matched_single',ps)]:
                    for i,yp in zip(te,p):records.append({'seed':seed,'fold':fold,'target':t,'model':model,'record_id':complete.iloc[i].record_id,'source_id':complete.iloc[i].source_id,'y_true':complete.iloc[i][t],'y_pred':yp})
            meta.append({'seed':seed,'fold':fold,'train_ids':complete.iloc[tr].record_id.tolist(),'test_ids':complete.iloc[te].record_id.tolist()})
    p=pd.DataFrame(records);p.to_csv(OUT/'multitarget_predictions.csv',index=False)
    (OUT/'multitarget_folds.json').write_text(json.dumps(meta,indent=2))
    rows=[]
    for key,g in p.groupby(['seed','target','model']):rows.append(dict(zip(['seed','target','model'],key),**v.metric(g.y_true.to_numpy(),g.y_pred.to_numpy())))
    r=pd.DataFrame(rows);r.to_csv(OUT/'multitarget_metrics.csv',index=False)
    strain_sensitivity(d)
    # Quantitative, matched primary observations. Units and measurement states explicit.
    rows=[
      ['XU15-01','XU2015',11.14,.662,.7968e14,'EBSDTable3 p3023; XRDTable2 p3022','Final solid-solution+T652',np.nan,720.5,5.0],
      ['XU15-02','XU2015',5.25,.764,1.2833e14,'EBSDTable3 p3023; XRDTable2 p3022','Final pre-recovery+solid-solution+T652',np.nan,699.1,4.2],
      ['XU16-01','XU2016',4.748,.536,.197e14,'EBSDTable2 p510; XRDTable3 p512','Final T6',667.8,693.6,4.0],
      ['XU16-02','XU2016',11.144,.662,1.156e14,'EBSDTable2 p510; XRDTable3 p512','Final pre-deformation+T6',682.3,738.6,5.0]]
    pd.DataFrame(rows,columns=['record_id','source_id','grain_size_um','low_angle_boundary_fraction','XRD_dislocation_density_m2','locator','measurement_state','YS_MPa','UTS_MPa','EL_pct']).to_csv(OUT/'primary_microstructure.csv',index=False)
    print('Auxiliary V5 completed',flush=True)

def strain_sensitivity(d):
    # Fix nominal-midpoint partitions so range sensitivity is not confounded by regrouping.
    original=v.splits
    frozen={p:original(d,p,42) for p in ['loso','input_grouped']}
    v.splits=lambda data,protocol,seed:frozen[protocol]
    rows=[];sp=[]
    for strain in [2.,2.5,3.]:
        z=d.copy();z.loc[z.record_id=='XU16-02','pre_deformation_pct']=strain
        for protocol in ['loso','input_grouped']:
            r,_,_,_=v.run(z,'phys',protocol,42,only_et=True)
            pp=pd.DataFrame(r);pp['nominal_strain_pct']=strain;sp.append(pp)
            for t,g in pp.groupby('target'):rows.append({'nominal_strain_pct':strain,'protocol':protocol,'target':t,**v.metric(g.y_true.to_numpy(),g.y_pred.to_numpy())})
    pd.DataFrame(rows).to_csv(OUT/'strain_range_sensitivity.csv',index=False)
    pd.concat(sp).to_csv(OUT/'strain_range_predictions.csv',index=False)
    v.splits=original

if __name__=='__main__':main()
