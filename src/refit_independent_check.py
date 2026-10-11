from pathlib import Path
import sys,json,time,hashlib
import numpy as np,pandas as pd
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'src'))
import review_v4 as v,advance_v5 as a
from sklearn.preprocessing import StandardScaler
d=pd.read_csv(root/'results/v5/dataset_v5.csv').fillna({'aging_stages':''})
arch=pd.read_csv(root/'results/v5/predictions_series_proxy.csv')
start=time.time()
# One full matched series repeat with original tree counts and every fixed learner.
actual=a.group_run(d,'phys',42)[0]
p=pd.DataFrame(actual)
key=['kind','protocol','seed','fold','target','model','record_id']
p.fold=p.fold.astype(str);arch.fold=arch.fold.astype(str)
joined=p.merge(arch[arch.kind.eq('phys') & arch.seed.eq(42)],on=key,suffixes=('_new','_arch'),validate='one_to_one')
assert len(joined)==len(p)==5522
diff=joined.y_pred_new-joined.y_pred_arch
by=joined.groupby('model').apply(lambda z:float(np.max(abs(z.y_pred_new-z.y_pred_arch))),include_groups=False).to_dict()
p.to_csv(root/'verification/refit_series_seed42.csv',index=False)
# Rerun the most severe nested transfer using exactly the archived candidate search.
nr,choices,logs=a.nested_one(d,'loso',42)
npred=pd.DataFrame(nr); old=pd.read_csv(root/'results/v5/predictions_nested.csv')
npred.fold=npred.fold.astype(str);old.fold=old.fold.astype(str)
nj=npred.merge(old[old.protocol=='loso'],on=key,suffixes=('_new','_arch'),validate='one_to_one')
npred.to_csv(root/'verification/refit_nested_loso.csv',index=False)
metrics=[]
for t,g in nj.groupby('target'):
 metrics.append({'target':t,'max_prediction_difference':float(abs(g.y_pred_new-g.y_pred_arch).max()),'recomputed':v.metric(g.y_true_new.to_numpy(),g.y_pred_new.to_numpy())})
report={'series_seed42_prediction_rows':len(p),'maximum_prediction_differences_by_model':by,'nested_loso_refit':metrics,'seconds':time.time()-start,'scope':'Full fixed augmented series repeat and nested collection transfer refitted; other archived repeats checked by metric recovery. Compiled-platform differences are reported, not concealed.'}
(root/'verification/fresh_fit_comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
