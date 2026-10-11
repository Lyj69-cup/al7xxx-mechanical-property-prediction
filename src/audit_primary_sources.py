"""Independent extraction of published table values for comparison to transcription."""
from pathlib import Path
import json,re,hashlib,unicodedata
import numpy as np,pandas as pd,pdfplumber
import xml.etree.ElementTree as ET
import argparse
p=argparse.ArgumentParser(description='Independently compare the 78 new primary summaries against their original PDF/XML tables. Source files are named in verification/primary_source_manifest.json.')
p.add_argument('--source-dir',type=Path,required=True)
p.add_argument('--liu-pdf',type=Path)
args=p.parse_args()
ROOT=Path(__file__).resolve().parents[1];PKG=ROOT;S=args.source_dir.resolve()
liu_file=args.liu_pdf.resolve() if args.liu_pdf else S/'imr_7055_Sc.pdf'
manifest=json.loads((PKG/'verification/primary_source_manifest.json').read_text(encoding='utf8'))
for entry in manifest['files']:
 source=liu_file if entry['filename']=='imr_7055_Sc.pdf' else S/entry['filename']
 actual=hashlib.sha256(source.read_bytes()).hexdigest()
 if actual!=entry['sha256_of_checked_copy']:
  raise ValueError(f"Source version differs for {entry['filename']}: {actual}. Inspect the original article before accepting a changed source copy.")

raw=pd.read_csv(PKG/'data/primary_traceable_raw.csv')
evidence=[]
def record(sid,cond,vals,locator):
 evidence.append({'source_id':sid,'original_condition':cond,'YS_MPa':vals[0],'UTS_MPa':vals[1],'EL_pct':vals[2],'independently_extracted_from':locator})
with pdfplumber.open(liu_file) as z:
 text=z.pages[5].extract_text()
 matches=re.findall(r'AA7055(?:–0\.25Sc)?\s+(\d+)\s+(\d+)\s+(\d+)',text)
 assert len(matches)==6,matches
 states=['AA7055/SS','AA7055–0.25Sc/SS','AA7055/120 C 12 h','AA7055–0.25Sc/120 C 12 h','AA7055/120 C 24 h','AA7055–0.25Sc/120 C 24 h']
 for c,m in zip(states,matches):record('LIU2020_SC',c,[float(v) for v in m],'Table 2 p.730')
r=ET.parse(S/'orlowska2022.xml')
tab=next(t for t in r.findall('.//table-wrap') if t.findtext('label')=='Table 3')
for row in tab.findall('.//tbody/tr'):
 cells=[''.join(c.itertext()).strip() for c in row]
 if len(cells)!=4:continue
 vals=[]
 for c in cells[1:]:
  m=re.match(r'(\d+(?:\.\d+)?)\s*±\s*(\d+(?:\.\d+)?)',c);assert m,c
  vals.append(float(m[1]))
 record('ORLOWSKA2022_HE',cells[0],vals,'Published XML Table 3')
with pdfplumber.open(S/'tai2022.pdf') as z:
 text=z.pages[4].extract_text()
 lines=[l for l in text.splitlines() if re.match(r'^\d+ h ',l)]
 assert len(lines)==5
 for l in lines:
  nums=[float(v) for v in re.findall(r'\d+(?:\.\d+)?',l)]
  assert len(nums)==13,(l,nums)
  for j,prior in enumerate([0,7]):
   b=1+6*j;record('TAI2022_NA',f'NA-{prior}d/{int(nums[0])} h artificial aging',[nums[b],nums[b+2],nums[b+4]],'Table 2 PDF p.5')
 m=re.search(r'As[^\n]*received material: YS\s*(\d+)\s*±\s*\d+\s*MPa; UTS\s*(\d+)\s*±\s*\d+\s*MPa; Elongation\s*(\d+(?:\.\d+)?)',text)
 assert m,text[-700:];record('TAI2022_NA','As-received before solution treatment',[float(v) for v in m.groups()],'Table 2 footnote PDF p.5')
with pdfplumber.open(S/'chen2017.pdf') as z:
 text=unicodedata.normalize('NFKC',z.pages[1].extract_text())
 for l in text.splitlines():
  nums=re.findall(r'\d+(?:\.\d+)?',l)
  if len(nums)==9 and len(nums[0])==1 and 1<=int(nums[0])<=9 and float(nums[1])>=450:
   record('CHEN2017_7A04',f'L9 scheme {nums[0]}',[float(nums[6]),float(nums[5]),float(nums[7])],'Table 3 p.99')
 assert sum(x['source_id']=='CHEN2017_7A04' for x in evidence)==9
 text=unicodedata.normalize('NFKC',z.pages[2].extract_text())
 assert all(v in text for v in ['658.43','593.07','10.25'])
 record('CHEN2017_7A04','A1B1C3D2 optimized confirmation',[593.07,658.43,10.25],'Optimized confirmation p.100, visually checked')
with pdfplumber.open(S/'space2024.pdf') as z:
 t=z.pages[1].extract_tables()[0]
 for row in t[1:]:
  thick=3 if '≤' in row[0] else 4
  record('BI2024_7A09',f'O state, thickness {thick} mm',[float(row[1]),float(row[2]),float(row[4])],'Table 2 PDF p.2')
 t=z.pages[2].extract_tables()[0]
 for row in t[2:]:
  temp=int(row[1])
  for thick,b in [(3,5),(4,8)]:record('BI2024_7A09',f'Solution {temp} C; thickness {thick} mm; age 120/160 C',[np.nan,float(row[b]),float(row[b+2])],'Table 3 PDF p.3')
 t=z.pages[3].extract_tables()[0]
 for row in t[3:]:
  for thick,b,c in [(3,6,3),(4,9,4)]:
   if not row[b]:continue
   a1,a2=row[c].split('+')
   record('BI2024_7A09',f'Solution 475 C; thickness {thick} mm; age {a1}/{a2} C',[np.nan,float(row[b]),float(row[b+2])],'Table 4 PDF p.4')
check=pd.DataFrame(evidence);assert len(check)==78,len(check)
new=raw[~raw.source_id.isin(['LI2020','XU2015','XU2016','XU2018','PMC9571267'])]
for _,r in new.iterrows():
 e=check[(check.source_id==r.source_id)&(check.original_condition==r.original_condition)]
 if r.source_id=='BI2024_7A09':e=e[e.independently_extracted_from.str.startswith(r.source_locator.split(';')[0])]
 assert len(e)==1,r.record_id
 assert np.allclose(r[['YS_MPa','UTS_MPa','EL_pct']].to_numpy(float),e.iloc[0][['YS_MPa','UTS_MPa','EL_pct']].to_numpy(float),rtol=0,atol=1e-10,equal_nan=True),r.record_id
check.to_csv(PKG/'verification/independent_primary_table_extraction.csv',index=False)
report={'status':'PASS','new_raw_records_checked':len(new),'new_reported_target_values_checked':int(new[['YS_MPa','UTS_MPa','EL_pct']].count().sum()),'retained_new_records':76,'old_primary_records':22,'old_primary_evidence':'Existing checked primary ledgers and published-source review; not reclassified as new external observations','source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [S/'tai2022.pdf',S/'chen2017.pdf',S/'space2024.pdf',S/'orlowska2022.xml',liu_file]},'qualifications':['Tai Table 2 NA-7d/2 h EL=17.1%, text=17.3%; table preserved','Tai specimen count is at least three, not exactly three','Chen three repetitions refer to impact tests; no tensile repeat count invented','Two repeated Bi Table 3/4 condition summaries removed conservatively','Absent chemistry/processing remains unknown, except nominal absence of intentional Sc in Liu base alloy','No original individual specimen IDs fabricated']}
(PKG/'verification/primary_table_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8');print(json.dumps(report,ensure_ascii=False))
