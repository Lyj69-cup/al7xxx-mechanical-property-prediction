"""Predict with the audited sensitivity inputs; report training-support diagnostics."""
import argparse, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import review_v4 as r

def validate_input(d):
    d=d.copy()
    for c in ['Zn','Mg','Cu']:
        if c not in d:raise ValueError(f'Measured {c} (wt.%) is required')
    for c in r.COMP+r.PROC:
        if c not in d:d[c]=np.nan
        original=d[c];numeric=pd.to_numeric(original,errors='coerce')
        if (original.notna() & numeric.isna()).any():raise ValueError(f'Non-numeric field: {c}')
        if np.isinf(numeric).any():raise ValueError(f'Non-finite field: {c}')
        d[c]=numeric
    if d[['Zn','Mg','Cu']].isna().any().any():raise ValueError('Zn, Mg and Cu cannot be missing')
    if ((d[r.COMP]<0)|(d[r.COMP]>100)).any().any() or (d[r.COMP].sum(axis=1)>100).any():raise ValueError('Composition must be wt.% between 0 and 100, with total at most 100')
    if (d[['solution_time_h','aging_time_h']]<0).any().any():raise ValueError('Negative hold time')
    if (d[['solution_temp_C','aging_temp_C']]<=-273.15).any().any():raise ValueError('Invalid absolute temperature')
    if ((d.pre_deformation_pct<0)|(d.pre_deformation_pct>100)).any():raise ValueError('Deformation percentage must be in [0,100]')
    for c in ['temper','product_form','special_process','aging_stages']:
        if c not in d:d[c]=''
        d[c]=d[c].fillna('').astype(str)
    for value in d.aging_stages:
        if not value.strip():continue
        try:
            stages=json.loads(value)
            valid=isinstance(stages,list) and bool(stages)
            for stage in stages:
                if not isinstance(stage,list) or len(stage)!=2:valid=False;break
                temp,hours=map(float,stage)
                if not np.isfinite([temp,hours]).all() or temp<=-273.15 or hours<=0:valid=False;break
            if not valid:raise ValueError('Invalid aging stages')
        except (ValueError,TypeError,OverflowError) as exc:
            raise ValueError('aging_stages must be JSON [[temperature_C, positive_hours], ...] with finite physical values') from exc
    if 'record_id' not in d:d['record_id']=[f'NEW-{i+1:03d}' for i in range(len(d))]
    if not len(d):raise ValueError('Empty input')
    return d

def predict(d,new):
    new=validate_input(new);result=new.copy();x=r.features(d);xn=r.features(new)
    factory=r.model_factory()
    for target in r.TARGET:
        train=d[target].notna(); med,sx,empty=r.fit_preprocess(x.loc[train])
        a=r.transform(x.loc[train],med,sx);b=r.transform(xn,med,sx)
        sy=StandardScaler().fit(d.loc[train,[target]]);y=sy.transform(d.loc[train,[target]]).ravel();allp=[]
        for name in r.legacy.ENSEMBLE:
            model=factory[name]();model.fit(a,y);p=sy.inverse_transform(model.predict(b).reshape(-1,1)).ravel()
            if not np.isfinite(p).all():raise ValueError('Non-finite component prediction')
            allp.append(p);result[f'{target}_{name}_pred']=p
        result[f'{target}_Fusion_pred']=np.mean(allp,axis=0)
        lo=d.loc[train,r.COMP+r.PROC].min();hi=d.loc[train,r.COMP+r.PROC].max()
        result[f'{target}_outside_observed_range']=((new[r.COMP+r.PROC]<lo)|(new[r.COMP+r.PROC]>hi)).any(axis=1)
        result[f'{target}_unseen_category']=(~new.temper.map(r.temper).isin(d.loc[train,'temper'].map(r.temper).unique())|~new.product_form.map(r.legacy.form_family).isin(d.loc[train,'product_form'].map(r.legacy.form_family).unique()))
        nn=NearestNeighbors(n_neighbors=2).fit(a);cut=np.quantile(nn.kneighbors(a)[0][:,1],.95)
        result[f'{target}_distance_exceeds_training_95pct']=nn.kneighbors(b,n_neighbors=1)[0][:,0]>cut
        result[f'{target}_unlearned_fields']=';'.join(empty)
    result['missing_raw_fields']=new[r.COMP+r.PROC].isna().sum(axis=1)
    result['physical_consistency_flag']=(result.YS_MPa_Fusion_pred>result.UTS_MPa_Fusion_pred)|(result.EL_pct_Fusion_pred<0)|(result.EL_pct_Fusion_pred>100)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',default='predictions_reviewed.csv');args=ap.parse_args()
    d=r.prepare();new=pd.read_csv(args.input);out=predict(d,new);out.to_csv(args.output,index=False)
    print(out[['record_id']+[t+'_Fusion_pred' for t in r.TARGET]+['missing_raw_fields','physical_consistency_flag']].to_string(index=False))
    print('Support flags are diagnostics, not calibrated prediction intervals or an engineering acceptance decision.')

if __name__=='__main__':main()
