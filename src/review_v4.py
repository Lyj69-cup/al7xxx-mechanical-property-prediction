"""Audited sensitivity analysis for upstream v4, without replacing its archived results.

Run: python src/review_v4.py
Outputs: results/reviewed_v4/. This is a collection-level validation, not proof
that the underlying Bhat experiments are independent or all primary-measured.
"""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, warnings
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors
import recompute_v4 as legacy

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'reviewed_v4'
COMP=legacy.COMP_ALL+['Ce']
PROC=legacy.PROC
TF=['tf_annealed','tf_natural','tf_artificial','tf_overaged','tf_unknown']
FF=legacy.FF
DERIVED=['Zn_Mg_ratio','Zn_Mg_Cu_total','Mg_Cu_ratio','Zr_Mn_Cr_total',
 'Zn_Mg_product','solution_temp_time','has_deformation','aging_time_log',
 'Zn_aging_interaction','Mg_solution_interaction','alloy_agingtemp',
 'aging_kinetic_P','has_special_process']
TARGET=legacy.TGT
FAMILIES={'Major solutes':['Zn','Mg','Cu'], 'Microalloying':['Zr','Mn','Cr','Ti'],
 'Impurities':['Fe','Si'], 'Other solutes':[c for c in COMP if c not in ['Zn','Mg','Cu','Zr','Mn','Cr','Ti','Fe','Si']],
 'Numeric processing':PROC+['special_process','aging_stages'],
 'Temper':['temper'], 'Product form':['product_form']}

def temper(t):
    s=str(t).lower().strip()
    if 'over' in s:return 'tf_overaged'
    if 'natural' in s or 't4' in s:return 'tf_natural'
    if any(x in s for x in ['t6','peak','artificial','two-step','rra']):return 'tf_artificial'
    if s=='o' or 'anneal' in s:return 'tf_annealed'
    return 'tf_unknown'  # no-processing/as-received is not certification of O

def prepare():
    original=pd.read_csv(ROOT/'data'/'al7xxx_dataset_v4.csv')
    d=original.copy(); d['Ce']=np.nan; d['aging_stages']=''
    rows=[]
    def change(mask,col,value,evidence):
        for i in d.index[mask]:
            old=d.at[i,col]
            if (pd.isna(old) and pd.isna(value)) or str(old)==str(value):continue
            rows.append({'record_id':d.at[i,'record_id'],'field':col,'old':old,'new':value,'evidence':evidence})
            d.at[i,col]=value
    bh=d.source_id=='BHAT2023'
    bhat_columns=pd.read_csv(ROOT/'data'/'bhat_al_data.csv',nrows=0).columns
    for c in COMP:
        if c not in bhat_columns:change(bh,c,np.nan,'Element is absent from raw Bhat schema; not a measured zero')
    change(bh & d.temper.str.contains('as-fabricated',na=False),'temper','as-fabricated (processing unspecified)','Bhat Processing=No Processing does not establish annealed O')
    li=d.source_id=='LI2020'
    for c,v in [('solution_temp_C',465.),('solution_time_h',4.),('aging_temp_C',120.),('aging_time_h',24.),('temper','RRA'),('special_process','Solution 465 C/4 h, water quench; RRA 120 C/24 h + 190 C/0.5 h + 120 C/24 h'),('aging_stages',json.dumps([[120,24],[190,0.5],[120,24]]))]:
        change(li,c,v,'Li 2020 Table 1 and Methods: melting, casting, heat treatment and tensile testing')
    ces=[0,.06,.01,0,0,.01,.01,0,0,0,0,0]
    for rid,ce in zip(d.loc[li,'record_id'],ces):change(d.record_id==rid,'Ce',ce,'Li 2020 Table 1 nominal composition')
    for i in d.index[~bh]:
        sid=d.at[i,'source_id']
        reported={'LI2020':['Zn','Mg','Cu','Y','Ti','Ce'],'PMC9571267':['Zn','Mg','Cu','Zr']}.get(sid,['Zn','Mg','Cu','Zr','Sr'])
        for c in COMP:
            if c not in reported:change(d.index==i,c,np.nan,'Not reported in transcribed primary composition; unknown is not measured zero')
    pm=(d.source_id=='PMC9571267') & (d.record_id!='PMC-01')
    change(pm,'aging_temp_C',165.,'Zhang 2022 final aging hold is 165 C/4 h after 120 C/4 h')
    change(pm,'aging_stages',json.dumps([[120,4],[165,4]]),'Zhang 2022 Section 2: ordered aging holds')
    change(d.record_id=='PMC-01','temper','as-received hot-forged','Zhang 2022 initial hot-forged material, not annealed O')
    # Exact duplicates also occur in the full raw Bhat file, including its
    # unmodelled elements, Al balance, processing class, targets and class.
    b=pd.read_csv(ROOT/'data'/'bhat_al_data.csv')
    b=b[(b.Zn*100>=3)&(b.Zn>b.Cu)].copy()
    cols=[c for c in b if not c.startswith('Unnamed')]
    key=b[cols].fillna('<NA>').astype(str).agg('|'.join,axis=1)
    representative={}; exclusions=[]
    for j,(idx,k) in enumerate(zip(b.iloc[:,0],key),1):
        rid=f'BH-{j:03d}'
        if k in representative:exclusions.append({'excluded_id':rid,'retained_id':representative[k], 'raw_dataset_row':int(idx),'reason':'Exact duplicate of every raw data field except row index; conservative deduplication sensitivity'})
        else:representative[k]=rid
    excluded={x['excluded_id'] for x in exclusions}
    d=d[~d.record_id.isin(excluded)].reset_index(drop=True)
    OUT.mkdir(parents=True,exist_ok=True)
    d.to_csv(OUT/'dataset_reviewed.csv',index=False,lineterminator='\n')
    pd.DataFrame(rows).to_csv(OUT/'field_corrections.csv',index=False)
    pd.DataFrame(exclusions).to_csv(OUT/'duplicate_exclusions.csv',index=False)
    # Un-imputed, explicit predictor matrix for review; this export does not
    # alter any labels, features, partitions or fitted predictions.
    pd.concat([d[['record_id','source_id']],features(d),d[TARGET]],axis=1).to_csv(
        OUT/'model_matrix_unimputed.csv',index=False,lineterminator='\n')
    (OUT/'model_parameters.json').write_text(json.dumps(
        {name:maker().get_params() for name,maker in model_factory().items()},
        indent=2,default=str))
    return d

def features(d,kind='phys',q=130000.):
    x=pd.DataFrame(index=d.index)
    for c in COMP+PROC:x[c]=pd.to_numeric(d[c],errors='coerce')
    tf=d.temper.map(temper); ff=d.product_form.map(legacy.form_family)
    for c in TF:x[c]=(tf==c).astype(float)
    for c in FF:x[c]=(ff==c).astype(float)
    cols=COMP+PROC+TF+FF
    if kind=='raw':return x[cols]
    zn,mg,cu=x.Zn,x.Mg,x.Cu
    x['Zn_Mg_ratio']=zn/(mg+.01);x['Zn_Mg_Cu_total']=zn+mg+cu
    x['Mg_Cu_ratio']=mg/(cu+.01);x['Zr_Mn_Cr_total']=x[['Zr','Mn','Cr']].sum(axis=1,min_count=3)
    x['Zn_Mg_product']=zn*mg
    x['solution_temp_time']=x.solution_temp_C*x.solution_time_h/100.
    x['has_deformation']=np.where(x.pre_deformation_pct.notna(),(x.pre_deformation_pct>0).astype(float),np.nan)
    positive=x.aging_time_h.where(x.aging_time_h>0)
    x['aging_time_log']=np.log1p(positive)
    x['Zn_aging_interaction']=zn*x.aging_temp_C/100.
    x['Mg_solution_interaction']=mg*x.solution_temp_C/100.
    x['alloy_agingtemp']=(zn+mg+cu)*x.aging_temp_C/100.
    def kinetic(row):
        stages=json.loads(row.aging_stages) if row.aging_stages else [[row.aging_temp_C,row.aging_time_h]]
        if not stages or any(not np.isfinite(t) or not np.isfinite(h) or t<=-273.15 or h<=0 for t,h in stages):return np.nan
        # log(sum(t_i/1s * exp(-Q/RT_i))); schedule summary, not a mechanistic proof.
        return float(logsumexp([np.log(h*3600.)-q/(8.314*(t+273.15)) for t,h in stages]))
    x['aging_kinetic_P']=d.apply(kinetic,axis=1)
    x['has_special_process']=d.special_process.fillna('').str.contains('pre-recovery|multi-stage|forging|pre-|RRA|two-step',case=False,regex=True).astype(float)
    return x[cols+DERIVED]

def fit_preprocess(x):
    a=x.to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        med=np.nanmedian(a,axis=0)
    empty=~np.isfinite(med);med[empty]=0.
    a=np.where(np.isfinite(a),a,med)
    sx=StandardScaler().fit(a)
    return med,sx,np.array(x.columns)[empty].tolist()

def transform(x,med,sx):
    a=x.to_numpy(float);return sx.transform(np.where(np.isfinite(a),a,med))

def model_factory():
    out=legacy.models()
    for name in list(out):
        maker=out[name]
        def factory(maker=maker):
            m=maker()
            if 'n_jobs' in m.get_params():m.set_params(n_jobs=1)
            return m
        out[name]=factory
    return out

def splits(d,protocol,seed):
    if protocol=='loso':
        return [(s,np.flatnonzero(d.source_id!=s),np.flatnonzero(d.source_id==s)) for s in sorted(d.source_id.unique())]
    if protocol=='random':return [(str(i),tr,te) for i,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(d),1)]
    # Identical encoded inputs remain together, without using property labels.
    x=features(d,'phys'); k=pd.util.hash_pandas_object(x,index=False).to_numpy()
    unique=np.unique(k);out=[]
    for i,(tr,te) in enumerate(KFold(5,shuffle=True,random_state=seed).split(unique),1):
        out.append((str(i),np.flatnonzero(np.isin(k,unique[tr])),np.flatnonzero(np.isin(k,unique[te]))))
    return out

def metric(y,p):
    return {'n':len(y),'R2':float(r2_score(y,p)) if len(y)>1 and np.ptp(y)>0 else None,
      'RMSE':float(np.sqrt(mean_squared_error(y,p))), 'MAE':float(mean_absolute_error(y,p)),
      'bias_pred_minus_true':float(np.mean(p-y))}

def run(d,kind,protocol,seed,repeats=0,q=130000.,only_et=False):
    x=features(d,kind,q); records=[];foldmeta=[];mdi=[]; perms=[]
    makers=model_factory()
    if only_et:makers={'ET':makers['ET']}
    for fold,tr,te in splits(d,protocol,seed):
        for target in TARGET:
            tr= np.asarray(tr); te=np.asarray(te)
            train=tr[d.loc[tr,target].notna().to_numpy()]; test=te[d.loc[te,target].notna().to_numpy()]
            if not len(test):continue
            if not len(train):raise ValueError(f'No training target: {protocol}/{fold}/{target}')
            med,sx,empty=fit_preprocess(x.loc[train]); a=transform(x.loc[train],med,sx);b=transform(x.loc[test],med,sx)
            sy=StandardScaler().fit(d.loc[train,[target]]);y=sy.transform(d.loc[train,[target]]).ravel()
            nn=NearestNeighbors(n_neighbors=2).fit(a);cut=float(np.quantile(nn.kneighbors(a)[0][:,1],.95))
            dist=nn.kneighbors(b,n_neighbors=1)[0][:,0]
            lo=d.loc[train,COMP+PROC].min(); hi=d.loc[train,COMP+PROC].max()
            outside=((d.loc[test,COMP+PROC]<lo)|(d.loc[test,COMP+PROC]>hi)).any(axis=1).to_numpy()
            unseen=(~d.loc[test,'temper'].map(temper).isin(d.loc[train,'temper'].map(temper).unique()) | ~d.loc[test,'product_form'].map(legacy.form_family).isin(d.loc[train,'product_form'].map(legacy.form_family).unique())).to_numpy()
            foldmeta.append({'kind':kind,'protocol':protocol,'seed':seed,'fold':fold,'target':target,
              'train_ids':d.loc[train,'record_id'].tolist(),'test_ids':d.loc[test,'record_id'].tolist(),
              'all_missing_train_columns_filled_zero':empty,'distance_95pct_training_LOO':cut})
            preds={}
            for name,maker in makers.items():
                m=maker();m.fit(a,y);p=sy.inverse_transform(m.predict(b).reshape(-1,1)).ravel()
                if not np.isfinite(p).all():raise ValueError(f'Non-finite prediction {name}')
                preds[name]=p
                if name=='ET' and kind=='phys' and protocol=='input_grouped':
                    for c,value in zip(x.columns,m.feature_importances_):mdi.append({'target':target,'seed':seed,'fold':fold,'feature':c,'importance':value})
                    if repeats:
                        base=mean_absolute_error(d.loc[test,target],p)
                        for family,fields in FAMILIES.items():
                            for rep in range(repeats):
                                rng=np.random.default_rng(420000+10000*TARGET.index(target)+1000*int(fold)+rep)
                                shuffled=d.loc[test].copy();order=rng.permutation(len(test))
                                shuffled[fields]=shuffled[fields].iloc[order].to_numpy()
                                xp=transform(features(shuffled,kind,q),med,sx)
                                pp=sy.inverse_transform(m.predict(xp).reshape(-1,1)).ravel()
                                perms.append({'target':target,'fold':fold,'family':family,'repeat':rep,'n':len(test),'delta_MAE':mean_absolute_error(d.loc[test,target],pp)-base})
            if not only_et:
                # Mean demands all six finite components. No silent reduced ensemble.
                preds['Fusion']=np.mean(np.stack([preds[n] for n in legacy.ENSEMBLE]),axis=0)
            for name,pred in preds.items():
                for j,i in enumerate(test):records.append({'kind':kind,'protocol':protocol,'seed':seed,'fold':fold,
                  'target':target,'model':name,'record_id':d.at[i,'record_id'],'source_id':d.at[i,'source_id'],
                  'y_true':float(d.at[i,target]),'y_pred':float(pred[j]), 'outside_observed_range':bool(outside[j]),
                  'unseen_category':bool(unseen[j]),'distance_exceeds_training_95pct':bool(dist[j]>cut),
                  'missing_raw_fields':int(d.loc[i,COMP+PROC].isna().sum())})
    return records,foldmeta,mdi,perms

def summarize(d,p,meta,mdi,perms):
    rows=[]
    for keys,g in p.groupby(['kind','protocol','seed','target','model']):
        rows.append(dict(zip(['kind','protocol','seed','target','model'],keys),**metric(g.y_true.to_numpy(),g.y_pred.to_numpy())))
    per=pd.DataFrame(rows);per.to_csv(OUT/'metrics_per_run.csv',index=False)
    summary=per.groupby(['kind','protocol','target','model'])[['R2','RMSE','MAE']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).strip('_') for c in summary.columns];summary.to_csv(OUT/'metrics_summary.csv',index=False)
    # LOSO has one fixed model seed. SD is undefined, not fabricated from repetitions.
    source=[]
    for keys,g in p[p.protocol=='loso'].groupby(['kind','target','model','source_id']):
        source.append(dict(zip(['kind','target','model','source_id'],keys),**metric(g.y_true.to_numpy(),g.y_pred.to_numpy())))
    s=pd.DataFrame(source);s.to_csv(OUT/'loso_per_source.csv',index=False)
    paired=s[s.model.isin(['ET','Fusion'])].pivot(index=['kind','target','source_id'],columns='model',values='MAE').reset_index()
    paired['Fusion_minus_ET_MAE']=paired.Fusion-paired.ET;paired.to_csv(OUT/'matched_source_differences.csv',index=False)
    cell=per[per.model.isin(['ET','Fusion'])].pivot(index=['protocol','seed','target'],columns=['kind','model'],values='R2')
    a,b,c,e=cell[('raw','ET')],cell[('phys','ET')],cell[('raw','Fusion')],cell[('phys','Fusion')]
    effects=pd.DataFrame({'feature_main':((b-a)+(e-c))/2,'fusion_main':((c-a)+(e-b))/2,
      'interaction':e-b-c+a,'feature_on_ET':b-a,'fusion_on_raw':c-a}).reset_index();effects.to_csv(OUT/'factorial_effects.csv',index=False)
    if mdi:
        mi=pd.DataFrame(mdi);mi.to_csv(OUT/'impurity_importance_all_fits.csv',index=False)
        mi.groupby(['target','feature']).importance.agg(['mean','std','count']).reset_index().to_csv(OUT/'impurity_importance_by_target.csv',index=False)
    if perms:
        pe=pd.DataFrame(perms);pe.to_csv(OUT/'permutation_all_repeats.csv',index=False)
        pe.groupby(['target','family','fold']).delta_MAE.agg(
            ['mean','std','count']).reset_index().to_csv(
                OUT/'permutation_fold_summary.csv',index=False)
        pe['weighted']=pe.delta_MAE*pe.n
        po=pe.groupby(['target','family','repeat']).apply(lambda z:z.weighted.sum()/z.n.sum(),include_groups=False).rename('delta_MAE').reset_index()
        po.groupby(['target','family']).delta_MAE.agg(['mean','std','count']).reset_index().to_csv(OUT/'permutation_by_target.csv',index=False)
    strata=[]
    r=p[(p.kind=='phys')&(p.model=='Fusion')].merge(d[['record_id','temper','product_form','Zn']],on='record_id',validate='many_to_one')
    r['temper_family']=r.temper.map(temper);r['Zn_bin']=pd.cut(r.Zn,[-np.inf,6,9,np.inf],right=False,labels=['Zn<6','6<=Zn<9','Zn>=9']).astype(str)
    r['elongation_band']=np.where((r.target=='EL_pct')&(r.y_true>=15),'EL>=15','other')
    for field in ['source_id','temper_family','product_form','Zn_bin','elongation_band']:
        for keys,g in r.groupby(['protocol','seed','target',field]):
            strata.append(dict(zip(['protocol','seed','target','stratum'],keys),field=field,**metric(g.y_true.to_numpy(),g.y_pred.to_numpy())))
    pd.DataFrame(strata).to_csv(OUT/'residual_strata.csv',index=False)
    report={'upstream_commit':'d742856f90dce2435b2fe4e2499bd2d87dacb6c2','n_records':len(d),
      'n_sources':d.source_id.nunique(),'source_counts':d.source_id.value_counts().to_dict(),
      'target_counts':{t:int(d[t].notna().sum()) for t in TARGET},'raw_features':32,'augmented_features':45,
      'permutation_families':FAMILIES,'processing_note':'Flat raw fields represent final holds; kinetic summary includes ordered RRA/two-step aging. Heating rates, forging temperatures, quench severity and homogenization are archived context, not complete model inputs.',
      'missing_rule':'Target-specific training median; entirely unobserved training columns become zero with an explicit ledger; no label imputation.',
      'deduplication_note':'49 exact raw Bhat repeats removed conservatively for sensitivity, not proof that every repeat represents the same experimental measurement.',
      'seed_note':'Five partition seeds for random and input-grouped; fixed estimator seed 42. LOSO evaluated once; no claimed repeat SD.',
      'independence_note':'Six collection groups; original-study independence and cross-collection overlap unresolved. No confirmatory significance claim.',
      'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','scipy','xgboost','lightgbm']},
      'sha256_input':hashlib.sha256((ROOT/'data'/'al7xxx_dataset_v4.csv').read_bytes()).hexdigest()}
    (OUT/'analysis_manifest.json').write_text(json.dumps(report,indent=2))
    (OUT/'fold_assignments.json').write_text(json.dumps(meta))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seeds',default='42,1,2,3,4');ap.add_argument('--permutation-repeats',type=int,default=20);args=ap.parse_args()
    seeds=[int(s) for s in args.seeds.split(',')]
    if len(set(seeds))!=len(seeds) or args.permutation_repeats<2:raise ValueError('Use distinct seeds and at least two permutation repeats')
    d=prepare();allp=[];allmeta=[];allmdi=[];allperm=[]
    settings={'seeds':seeds,'permutation_repeats':args.permutation_repeats,
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'dataset_sha256':hashlib.sha256((OUT/'dataset_reviewed.csv').read_bytes()).hexdigest(),
      'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','scipy','xgboost','lightgbm']}}
    setting_path=OUT/'analysis_settings.json'
    if setting_path.exists() and json.loads(setting_path.read_text())!=settings:
        raise ValueError('Cached analysis settings differ. Move results/reviewed_v4 to an archive before recomputing.')
    if not setting_path.exists() and list(OUT.glob('cache_*.json')):raise ValueError('Unversioned cache exists; archive it before rerunning')
    setting_path.write_text(json.dumps(settings,indent=2))
    for kind in ['phys','raw']:
        for protocol in ['input_grouped','loso','random']:
            for seed in ([42] if protocol=='loso' else seeds):
                path=OUT/f'cache_{kind}_{protocol}_{seed}.json'
                if path.exists():z=json.loads(path.read_text())
                else:
                    print(kind,protocol,seed,'start',flush=True)
                    z=run(d,kind,protocol,seed,args.permutation_repeats if kind=='phys' and protocol=='input_grouped' and seed==42 else 0)
                    path.write_text(json.dumps(z,allow_nan=False));print(kind,protocol,seed,'complete',flush=True)
                for dst,src in zip([allp,allmeta,allmdi,allperm],z):dst.extend(src)
    p=pd.DataFrame(allp);p.to_csv(OUT/'predictions_long.csv',index=False);summarize(d,p,allmeta,allmdi,allperm)
    qrows=[]
    for q in [100000.,130000.,160000.]:
        for protocol in ['input_grouped','loso']:
            cache=OUT/f'cache_Q_{int(q)}_{protocol}.json'
            if cache.exists():z=json.loads(cache.read_text())
            else:z=run(d,'phys',protocol,42,q=q,only_et=True)[0];cache.write_text(json.dumps(z,allow_nan=False))
            for t,g in pd.DataFrame(z).groupby('target'):qrows.append({'Q_J_mol':q,'protocol':protocol,'target':t,**metric(g.y_true.to_numpy(),g.y_pred.to_numpy())})
    pd.DataFrame(qrows).to_csv(OUT/'Q_sensitivity_ET.csv',index=False)
    print('Saved',OUT,flush=True)

if __name__=='__main__':main()
