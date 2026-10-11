"""Rebuild the verified public-row crosswalk. Does not authenticate experimental identities."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def main():
 archive=pd.read_csv(ROOT/'data/al7xxx_dataset_v4.csv')
 raw=pd.read_csv(ROOT/'data/bhat_al_data.csv')
 raw_index=raw.columns[0]
 assert raw[raw_index].is_unique
 source=raw.set_index(raw_index)
 bhat=archive[archive.source_id.eq('BHAT2023')].copy()
 reviewed=pd.read_csv(ROOT/'results/dataset_reviewed.csv')
 retained=set(reviewed.record_id)
 excluded=pd.read_csv(ROOT/'results/duplicate_exclusions.csv').set_index('excluded_id')
 targets={'YS_MPa':'Yield Strength (MPa)','UTS_MPa':'Tensile Strength (MPa)','EL_pct':'Elongation (%)'}
 records=[]
 for _,r in bhat.iterrows():
  idx=int(r.localization.rsplit(' ',1)[1]);s=source.loc[idx]
  for col in ['Zn','Mg','Cu','Mn','Cr','Zr','Fe','Si','Ti','Ni','Sc','Er','Li','Ag','V']:
   assert np.isclose(r[col],round(float(s[col])*100,4),atol=1e-10,rtol=0), (r.record_id,col)
  for target,col in targets.items():
   assert (pd.isna(r[target]) and pd.isna(s[col])) or r[target]==s[col], (r.record_id,target)
  kept=r.record_id in retained
  representative=r.record_id if kept else excluded.loc[r.record_id,'retained_id']
  records.append({'archive_record_id':r.record_id,'public_dataset_row_id':idx,
   'retained_in_188_record_benchmark':kept,'representative_record_id':representative,
   'upstream_dataset_doi':'10.17632/b6br4yk6r3.1','public_processing_label':s.Processing,
   'Zn_wt_pct':r.Zn,'Mg_wt_pct':r.Mg,'Cu_wt_pct':r.Cu,
   **{t:r[t] for t in targets},
   'numerical_row_mapping_status':'verified against archived upstream CSV',
   'original_study_specimen_series_status':'not available in audited public repository or upstream CSV'})
 out=pd.DataFrame(records)
 assert len(out)==215 and out.retained_in_188_record_benchmark.sum()==166
 out.to_csv(ROOT/'supplementary/Bhat_public_row_crosswalk.csv',index=False,encoding='utf-8-sig')
 result={'status':'PASS','public_dataset_rows':len(raw),'selected_archive_rows':215,
  'retained_secondary_rows':166,'excluded_exact_repeat_rows':49,
  'verified':'Archived public row identities, original available compositions, target labels and repeat representatives',
  'not_verified':'Original study, specimen or real experimental series identities; record-specific extraction type or independence',
  'composition_conversion':'Fractions in the upstream CSV multiplied by 100 and rounded to four decimals as in upstream build_v4.py'}
 (ROOT/'verification/bhat_public_row_check.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
