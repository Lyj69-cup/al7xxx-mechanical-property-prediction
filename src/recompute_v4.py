# -*- coding: utf-8 -*-
"""
al7xxx v4 统计严谨重算引擎（全部记录可逐行溯源，零合成）
主分析：
  (A) 随机五折 KFold(5)，跨5种子 —— 同分布基准
  (B) 留一来源 LOSO：
        - 工程场景：以Bhat大库为底座，逐个留出独立文献来源（pool小来源评估）
        - 反向场景：留出Bhat，用22条小样本预测215条（外推局限）
  基线 Dummy/Linear/Ridge/Tree；6模型 RF/ET/XGB/LGB/GBRT/SVR + 简单平均融合
  2x2析因(原始/物理衍生 x 单模ET/融合)；来源漂移ICC + oracle校准；配对Wilcoxon；特征重要性
运行: D:\\python\\python.exe recompute_v4.py
"""
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import wilcoxon
import xgboost as xgb, lightgbm as lgb

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]   # repository root (parent of src/)
DATA = str(ROOT / "data")                    # input datasets
RES  = str(ROOT / "results")                # all outputs
Path(RES).mkdir(parents=True, exist_ok=True)
SEEDS = [42, 1, 2, 3, 4]
COMP_ALL = ["Zn","Mg","Cu","Mn","Cr","Zr","Fe","Si","Ti","Ni","Sc","Sr","Y","Er","Li","Ag","V"]
COMP_MAIN = ["Zn","Mg","Cu","Zr","Mn","Cr","Fe","Si","Ti"]
PROC = ["solution_temp_C","solution_time_h","aging_temp_C","aging_time_h","pre_deformation_pct"]
TGT = ["YS_MPa","UTS_MPa","EL_pct"]
TGTNAME = {"YS_MPa":"yield","UTS_MPa":"tensile","EL_ppt":"elongation","EL_pct":"elongation"}
Q_AGING = 130000.0; R_GAS = 8.314

TF = ["tf_O","tf_T4nat","tf_T6peak","tf_T7over","tf_unknown"]
FF = ["ff_extrusion","ff_forging","ff_castnearnet","ff_unknown"]
CAT = TF + FF

def temper_family(t):
    t = str(t).lower()
    if "over" in t: return "tf_T7over"
    if "natural" in t or "t4" in t: return "tf_T4nat"
    if any(k in t for k in ["peak","t6","two-step","artificial","pre-recovery","pre-deformation"]): return "tf_T6peak"
    if "as-fab" in t or "as-forged" in t or t=="o": return "tf_O"
    return "tf_unknown"

def form_family(p):
    p = str(p).lower()
    if "extr" in p: return "ff_extrusion"
    if "forg" in p: return "ff_forging"
    if "cast" in p: return "ff_castnearnet"
    return "ff_unknown"

def cat_block(df):
    tf = df["temper"].map(temper_family); ff = df["product_form"].map(form_family)
    out = pd.DataFrame(index=df.index)
    for c in TF: out[c] = (tf==c).astype(float).values
    for c in FF: out[c] = (ff==c).astype(float).values
    return out

def features(df, kind="phys"):
    d = pd.DataFrame(index=df.index)
    for c in COMP_ALL: d[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in PROC: d[c] = pd.to_numeric(df[c], errors="coerce")
    cat = cat_block(df)
    for c in CAT: d[c] = cat[c].values
    base = COMP_ALL + PROC + CAT
    added = []
    if kind=="phys":
        Zn=d.Zn; Mg=d.Mg; Cu=d.Cu; Zr=d.Zr; Mn=d.Mn; Cr=d.Cr; Fe=d.Fe; Si=d.Si
        at=d.aging_temp_C; ah=d.aging_time_h; st=d.solution_temp_C; sh=d.solution_time_h
        pre=d.pre_deformation_pct
        d["Zn_Mg_ratio"]=Zn/(Mg+0.01)
        d["Zn_Mg_Cu_total"]=Zn+Mg+Cu
        d["Mg_Cu_ratio"]=Mg/(Cu+0.01)
        d["Zr_Mn_Cr_total"]=Zr+Mn+Cr
        d["Zn_Mg_product"]=Zn*Mg
        d["solution_temp_time"]=(st*sh/100.0).fillna(0.0)
        d["has_deformation"]=(pre.fillna(0)>0).astype(float)
        d["aging_time_log"]=np.log1p(ah.fillna(0))
        d["Zn_aging_interaction"]=(Zn*at/100.0).fillna(0.0)
        d["Mg_solution_interaction"]=(Mg*st/100.0).fillna(0.0)
        d["alloy_agingtemp"]=((Zn+Mg+Cu)*at/100.0).fillna(0.0)
        Tk=at+273.15
        d["aging_kinetic_P"]=np.log(ah.clip(lower=1e-3)*3600.0) - Q_AGING/(R_GAS*Tk)
        d["has_special_process"]=df.special_process.fillna("").str.contains(
            "pre-recovery|multi-stage|forging|pre-", regex=True).astype(float)
        added=["Zn_Mg_ratio","Zn_Mg_Cu_total","Mg_Cu_ratio","Zr_Mn_Cr_total","Zn_Mg_product",
               "solution_temp_time","has_deformation","aging_time_log","Zn_aging_interaction",
               "Mg_solution_interaction","alloy_agingtemp","aging_kinetic_P","has_special_process"]
    cols = base + added
    return d, cols

def models():
    return {
      "Dummy": lambda: DummyRegressor(strategy="mean"),
      "Linear": lambda: LinearRegression(),
      "Ridge": lambda: Ridge(alpha=10.0),
      "Tree": lambda: DecisionTreeRegressor(max_depth=8, random_state=42),
      "RF": lambda: RandomForestRegressor(n_estimators=500,max_features=0.5,random_state=42,n_jobs=-1),
      "ET": lambda: ExtraTreesRegressor(n_estimators=500,max_features=0.5,random_state=42,n_jobs=-1),
      "XGB": lambda: xgb.XGBRegressor(n_estimators=500,max_depth=5,learning_rate=0.03,subsample=0.8,
              colsample_bytree=0.6,min_child_weight=3,reg_alpha=0.5,reg_lambda=2.0,random_state=42,n_jobs=-1,verbosity=0),
      "LGB": lambda: lgb.LGBMRegressor(n_estimators=500,max_depth=7,learning_rate=0.03,subsample=0.8,
              colsample_bytree=0.6,min_child_samples=8,reg_alpha=0.5,reg_lambda=2.0,random_state=42,n_jobs=-1,verbosity=-1),
      "GBRT": lambda: GradientBoostingRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.8,random_state=42),
      "SVR": lambda: SVR(kernel="rbf",C=200,gamma="scale",epsilon=0.05),
    }
ENSEMBLE = ["RF","ET","XGB","LGB","GBRT","SVR"]

def folds_random(n, seed):
    return list(KFold(5, shuffle=True, random_state=seed).split(np.zeros(n)))

def folds_loso(groups, test_source):
    g = np.asarray(groups)
    te = np.where(g==test_source)[0]
    tr = np.where(g!=test_source)[0]
    return [(tr,te)]

def run_folds(df, kind, folds_list, Y):
    Xd, cols = features(df, kind)
    Xraw = Xd[cols].values.astype(float)
    n=len(df); fac=models()
    oof={name:np.full((n,3),np.nan) for name in list(fac)+["Fusion"]}
    for ti in range(3):
        avail=~np.isnan(Y[:,ti])
        for tr,te in folds_list:
            trm=tr[avail[tr]]; tem=te[avail[te]]
            if len(tem)==0 or len(trm)==0: continue
            sx=StandardScaler().fit(np.nan_to_num(Xraw[tr],nan=0.0))
            Xtr=sx.transform(np.nan_to_num(Xraw[trm],nan=0.0))
            Xte=sx.transform(np.nan_to_num(Xraw[tem],nan=0.0))
            sy=StandardScaler().fit(Y[trm,ti].reshape(-1,1))
            ytr=sy.transform(Y[trm,ti].reshape(-1,1)).ravel()
            for name,cf in fac.items():
                m=cf(); m.fit(Xtr,ytr)
                p=sy.inverse_transform(m.predict(Xte).reshape(-1,1)).ravel()
                oof[name][tem,ti]=p
    oof["Fusion"]=np.nanmean(np.stack([oof[m] for m in ENSEMBLE]),axis=0)
    return oof, cols

def metrics_on(o, Y, idx, ti):
    avail=~np.isnan(Y[:,ti]); m=avail & idx & ~np.isnan(o[:,ti])
    if m.sum()<2: return None
    return {"n":int(m.sum()),
            "R2":float(r2_score(Y[m,ti],o[m,ti])),
            "RMSE":float(np.sqrt(mean_squared_error(Y[m,ti],o[m,ti]))),
            "MAE":float(mean_absolute_error(Y[m,ti],o[m,ti]))}

def main():
    df=pd.read_csv(os.path.join(DATA,"al7xxx_dataset_v4.csv"))
    n=len(df); groups=df.source_id.values
    Y=df[TGT].values.astype(float)
    report={"n_records":n,"n_sources":int(df.source_id.nunique()),
            "sources":df.source_id.value_counts().to_dict(),
            "target_coverage":{t:int(df[t].notna().sum()) for t in TGT}}
    print(f"v4: {n} records, {df.source_id.nunique()} sources")
    results={}
    mnames=list(models().keys())+["Fusion"]

    # ---- A) random 5-fold, mean over seeds ----
    agg={nm:[[] for _ in range(3)] for nm in mnames}
    oof_seed42=None
    for seed in SEEDS:
        oof,_=run_folds(df,"phys",folds_random(n,seed),Y)
        if seed==42: oof_seed42=oof
        for nm in mnames:
            for ti in range(3):
                mm=metrics_on(oof[nm],Y,np.ones(n,bool),ti)
                if mm: agg[nm][ti].append(mm)
    random_summary={}
    for nm in mnames:
        random_summary[nm]={}
        for ti,t in enumerate(TGT):
            a=agg[nm][ti]
            if a:
                random_summary[nm][t]={k+"_mean":float(np.mean([z[k] for z in a])) for k in ["R2","RMSE","MAE"]}
                random_summary[nm][t]["R2_std"]=float(np.std([z["R2"] for z in a],ddof=1))
    results["random_phys"]=random_summary

    # ---- B) LOSO (phys), mean over seeds ----
    loso_oof={nm:np.full((n,3),np.nan) for nm in mnames}
    per_source={}
    for src in pd.Series(groups).unique():
        fl=folds_loso(groups,src); te=fl[0][1]
        acc={nm:[[] for _ in range(3)] for nm in mnames}
        stack={nm:np.full((len(SEEDS),len(te),3),np.nan) for nm in mnames}
        for si,seed in enumerate(SEEDS):
            oof,_=run_folds(df,"phys",fl,Y)
            teidx=(groups==src)
            for nm in mnames:
                stack[nm][si]=oof[nm][te]
                for ti in range(3):
                    mm=metrics_on(oof[nm],Y,teidx,ti)
                    if mm: acc[nm][ti].append(mm)
        for nm in mnames:
            loso_oof[nm][te]=np.nanmean(stack[nm],axis=0)
        per_source[src]={}
        for nm in ["ET","Fusion"]:
            per_source[src][nm]={}
            for ti,t in enumerate(TGT):
                a=acc[nm][ti]
                if a:
                    per_source[src][nm][t]={k:float(np.mean([z[k] for z in a])) for k in ["R2","RMSE","MAE","n"]}
    # pooled: small sources (non-Bhat) = engineering scenario
    small_idx=(groups!="BHAT2023"); bhat_idx=(groups=="BHAT2023")
    pooled={}
    for nm in mnames:
        pooled[nm]={"small_sources":{},"bhat_heldout":{}}
        for ti,t in enumerate(TGT):
            pooled[nm]["small_sources"][t]=metrics_on(loso_oof[nm],Y,small_idx,ti)
            pooled[nm]["bhat_heldout"][t]=metrics_on(loso_oof[nm],Y,bhat_idx,ti)
    results["loso_per_source"]=per_source
    results["loso_pooled"]=pooled

    # ---- C) raw vs phys 2x2 factorial (random CV for stable cells) ----
    def cell_random(kind, model):
        ag=[[] for _ in range(3)]
        for seed in SEEDS:
            oof,_=run_folds(df,kind,folds_random(n,seed),Y)
            for ti in range(3):
                mm=metrics_on(oof[model],Y,np.ones(n,bool),ti)
                if mm: ag[ti].append(mm["R2"])
        return [float(np.mean(a)) for a in ag]
    cells={
      "raw_ET":cell_random("raw","ET"),"raw_Fusion":cell_random("raw","Fusion"),
      "phys_ET":cell_random("phys","ET"),"phys_Fusion":cell_random("phys","Fusion")}
    def avg(c): return float(np.mean(c))
    fe=(avg(cells["phys_ET"])-avg(cells["raw_ET"]))
    fu=(avg(cells["raw_Fusion"])-avg(cells["raw_ET"]))
    inter=((avg(cells["phys_Fusion"])-avg(cells["phys_ET"]))-(avg(cells["raw_Fusion"])-avg(cells["raw_ET"])))
    results["factorial_random"]={"cells":cells,"phys_feature_gain_on_ET":fe,
        "fusion_gain_on_raw":fu,"interaction_phys_x_fusion":inter}

    # ---- D) source drift ICC + oracle (pooled LOSO Fusion OOF) ----
    drift={}
    o=loso_oof["Fusion"]
    for ti,t in enumerate(TGT):
        avail=~np.isnan(Y[:,ti]); mask=avail&~np.isnan(o[:,ti])
        e=o[:,ti]-Y[:,ti]
        s=pd.Series(e[mask]); g=pd.Series(groups[mask])
        means=s.groupby(g).mean(); cnt=s.groupby(g).size()
        multi=means[cnt>=2]; grand=s.mean()
        between=float(np.sum(cnt.values*(means.values-grand)**2)/len(s))
        within=float(sum(np.sum((s[g==gs]-means[gs])**2) for gs in means.index)/len(s))
        icc=between/(between+within) if (between+within)>0 else None
        calib=g.map(means).values
        drift[t]={"n":int(mask.sum()),"n_sources":int(len(means)),
          "std_source_mean_residual":float(multi.std(ddof=1)) if len(multi)>1 else None,
          "mean_abs_source_mean_residual":float(multi.abs().mean()),
          "within_source_residual_std":float(np.sqrt(within)),
          "ICC_between_source_share":float(icc) if icc is not None else None,
          "R2_raw":float(r2_score(Y[mask,ti],o[mask,ti])),
          "R2_oracle_sourceMeanCalibration_notDeployable":float(r2_score(Y[mask,ti],o[mask,ti]-calib))}
    results["source_drift_loso"]=drift

    # ---- E) paired Wilcoxon (random seed42, per-sample sq error) ----
    sig={}
    def paired(a,b,ti):
        avail=~np.isnan(Y[:,ti]); ea=(a[:,ti]-Y[:,ti])**2; eb=(b[:,ti]-Y[:,ti])**2
        m=avail&~np.isnan(ea)&~np.isnan(eb); d=ea[m]-eb[m]
        if len(d)<10 or np.allclose(d,0): return {"n":int(m.sum()),"p":None}
        try: _,p=wilcoxon(ea[m],eb[m])
        except Exception: p=None
        return {"n":int(m.sum()),"wilcoxon_p":(float(p) if p is not None else None),
                "median_sqErr_diff":float(np.median(d))}
    for ti,t in enumerate(TGT):
        sig[t]={"Fusion_vs_ET":paired(oof_seed42["Fusion"],oof_seed42["ET"],ti),
                "Fusion_vs_Linear":paired(oof_seed42["Fusion"],oof_seed42["Linear"],ti)}
    results["significance_random_seed42"]=sig

    # ---- F) feature importance (ET, mean over random folds seed42) ----
    Xd,cols=features(df,"phys")
    imp=np.zeros(len(cols))
    Xall=np.nan_to_num(Xd[cols].values,nan=0.0)
    for tr,te in folds_random(n,42):
        for ti in range(3):
            avail=~np.isnan(Y[:,ti]); trm=tr[avail[tr]]
            sx=StandardScaler().fit(Xall[trm])
            e2=ExtraTreesRegressor(n_estimators=300,max_features=0.5,random_state=42,n_jobs=-1)
            e2.fit(sx.transform(Xall[trm]), Y[trm,ti])
            imp+=e2.feature_importances_
    imp=imp/(5*3)
    order=np.argsort(imp)[::-1]
    results["feature_importance_top15"]={cols[i]:float(imp[i]) for i in order[:15]}

    json.dump(report,open(os.path.join(RES,"v4_report_meta.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    json.dump(results,open(os.path.join(RES,"recompute_v4_results.json"),"w",encoding="utf-8"),
              ensure_ascii=False,indent=2,default=str)

    # OOF csvs
    def dump(pred,name):
        pd.DataFrame({TGT[i]+"_pred":pred[:,i] for i in range(3)}).assign(
            **{TGT[i]:Y[:,i] for i in range(3)},
            source=groups,temper=df.temper.values).to_csv(
            os.path.join(RES,name),index=False,encoding="utf-8-sig")
    dump(oof_seed42["Fusion"],"oof_random_v4.csv")
    dump(loso_oof["Fusion"],"oof_loso_v4.csv")

    # console
    print("\n"+"="*70); print("Random 5-fold (phys), R2 mean +/- std over 5 seeds")
    for nm in ["Dummy","Linear","Ridge","Tree","RF","ET","XGB","LGB","GBRT","SVR","Fusion"]:
        r=random_summary[nm]
        print(f"{nm:<8}"+"  ".join(f"{TGTNAME[t]}={r[t]['R2_mean']:.3f}+-{r[t]['R2_std']:.3f}" for t in TGT if t in r))
    print("\nLOSO pooled small sources (Bhat-trained -> predict independent studies), Fusion:")
    for t in TGT: print(" ",t, pooled["Fusion"]["small_sources"][t])
    print("LOSO Bhat held-out (22 records -> predict 215), Fusion:")
    for t in TGT: print(" ",t, pooled["Fusion"]["bhat_heldout"][t])
    print("\nPer-source Fusion:")
    for src,v in per_source.items():
        print(" ",src, {t:(round(v['Fusion'][t]['R2'],3) if t in v['Fusion'] else None) for t in TGT})
    print("\nFactorial cells (avg R2):",{k:round(avg(c),3) for k,c in cells.items()})
    print("phys gain ET=%.4f fusion gain raw=%.4f interaction=%.4f"%(fe,fu,inter))
    print("\nSource drift:",json.dumps(drift,ensure_ascii=False,indent=1))
    print("\nSignificance:",json.dumps(sig,ensure_ascii=False))
    print("\nTop features:",json.dumps(results["feature_importance_top15"],ensure_ascii=False))
    print("\nSaved recompute_v4_results.json / oof_random_v4.csv / oof_loso_v4.csv")

if __name__=="__main__":
    main()
