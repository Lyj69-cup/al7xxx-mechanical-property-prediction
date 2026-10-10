# -*- coding: utf-8 -*-
"""v4 family-level grouped permutation importance (ET, random KFold seed42)."""
import os, json
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_v4 import features, TGT, folds_random

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = str(ROOT/"data"); RES = str(ROOT/"results")
Path(RES).mkdir(parents=True, exist_ok=True)
df = pd.read_csv(os.path.join(DATA, 'al7xxx_dataset_v4.csv'))
n = len(df); Y = df[TGT].values.astype(float)
Xd, cols = features(df, 'phys')
Xall = np.nan_to_num(Xd[cols].values.astype(float), nan=0.0)
idx = {c: j for j, c in enumerate(cols)}

FAMS = {
 'Temper family': ['tf_O','tf_T4nat','tf_T6peak','tf_T7over','tf_unknown'],
 'Major solutes (Zn/Mg/Cu)': ['Zn','Mg','Cu','Zn_Mg_ratio','Zn_Mg_Cu_total','Mg_Cu_ratio',
    'Zn_Mg_product','Zn_aging_interaction','Mg_solution_interaction','alloy_agingtemp'],
 'Processing (numeric)': ['solution_temp_C','solution_time_h','aging_temp_C','aging_time_h',
    'pre_deformation_pct','solution_temp_time','has_deformation','aging_time_log','aging_kinetic_P','has_special_process'],
 'Microalloying (Zr/Mn/Cr/Ti)': ['Zr','Mn','Cr','Ti','Zr_Mn_Cr_total'],
 'Product form': ['ff_extrusion','ff_forging','ff_castnearnet','ff_unknown'],
 'Fe/Si impurities': ['Fe','Si'],
 'Other solutes': ['Ni','Sc','Sr','Y','Er','Li','Ag','V'],
}
fcols = {k: [idx[c] for c in v if c in idx] for k, v in FAMS.items()}

rng = np.random.RandomState(42)
base_mae = np.zeros(3); perm_mae = {k: np.zeros(3) for k in FAMS}
folds = folds_random(n, 42)
for tr, te in folds:
    for ti in range(3):
        avail = ~np.isnan(Y[:, ti]); trm = tr[avail[tr]]; tem = te[avail[te]]
        sx = StandardScaler().fit(Xall[trm])
        Xtr = sx.transform(Xall[trm]); Xte = sx.transform(Xall[tem])
        et = ExtraTreesRegressor(n_estimators=300, max_features=0.5, random_state=42, n_jobs=-1)
        et.fit(Xtr, Y[trm, ti])
        base_mae[ti] += mean_absolute_error(Y[tem, ti], et.predict(Xte))
        for k, cc in fcols.items():
            Xp = Xte.copy()
            if len(cc) == 1:
                Xp[:, cc[0]] = rng.permutation(Xp[:, cc[0]])
            else:
                perm = rng.permutation(len(Xp))
                Xp[:, cc] = Xp[perm][:, cc]
            perm_mae[k][ti] += mean_absolute_error(Y[tem, ti], et.predict(Xp))
nf = len(folds)
out = {}
for k in FAMS:
    d = [(perm_mae[k][ti]-base_mae[ti])/nf for ti in range(3)]
    out[k] = {'dMAE_YS': round(d[0],3), 'dMAE_UTS': round(d[1],3),
              'dMAE_EL': round(d[2],3), 'dMAE_mean': round(float(np.mean(d)),3)}
out = dict(sorted(out.items(), key=lambda kv: -kv[1]['dMAE_mean']))
json.dump(out, open(os.path.join(RES,'family_permutation_v4.json'),'w',encoding='utf-8'),
          ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=1))
