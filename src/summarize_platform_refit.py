from pathlib import Path
import pandas as pd,json
from sklearn.metrics import r2_score,mean_absolute_error
root=Path(__file__).resolve().parents[1]
n=pd.read_csv(root/'verification/refit_series_seed42.csv');o=pd.read_csv(root/'results/v5/predictions_series_proxy.csv')
rows=[]
for (t,m),g in n.groupby(['target','model']):
 old=o[(o.kind=='phys')&(o.seed==42)&(o.target==t)&(o.model==m)]
 j=g[['record_id','y_pred']].merge(old[['record_id','y_pred']],on='record_id',suffixes=('_new','_archive'))
 rows.append({'target':t,'model':m,'archive_R2':r2_score(old.y_true,old.y_pred),'fresh_Windows_R2':r2_score(g.y_true,g.y_pred),'archive_MAE':mean_absolute_error(old.y_true,old.y_pred),'fresh_Windows_MAE':mean_absolute_error(g.y_true,g.y_pred),'max_prediction_difference':abs(j.y_pred_new-j.y_pred_archive).max()})
d=pd.DataFrame(rows);d.to_csv(root/'verification/platform_metrics_seed42.csv',index=False)
print(d[d.model.isin(['ET','Fusion','XGB'])].to_string(index=False))
