# -*- coding: utf-8 -*-
"""Build v4 fully-traceable 7xxx dataset. Zero synthesis: every row carries a real source (DOI/URL)."""
import pandas as pd
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

COMP = ['Zn','Mg','Cu','Mn','Cr','Zr','Fe','Si','Ti','Ni','Sc','Sr','Y','Er','Li','Ag','V']
OUT_COLS = ['record_id','source_id','data_origin','localization','product_form','temper',
            'solution_temp_C','solution_time_h','aging_temp_C','aging_time_h',
            'pre_deformation_pct','special_process'] + COMP + ['YS_MPa','UTS_MPa','EL_pct',
            'citation','doi','url']

SOURCES = {
 'BHAT2023': dict(cit='Bhat N, Barnard A, Birbilis N. Aluminium alloy dataset for supervised learning. Mendeley Data, V1, 2023.',
   doi='10.17632/b6br4yk6r3.1', url='https://data.mendeley.com/datasets/b6br4yk6r3/1'),
 'XU2015': dict(cit='Tan C, Xu X-J, Jiang W, Sun L-S, Zhang X-L, Fan Y-Z, Zhao J-J. Effect of pre-recovery treatment on microstructure and properties of ultra-high-strength aluminium alloy extrusion treated by solid solution-T652 treatment. The Chinese Journal of Nonferrous Metals, 2015, 25(11): 3019-3025.',
   doi='', url='http://www.ysxbcn.com/down/2015/11_cn/08-p3019-33029.pdf'),
 'XU2016': dict(cit='Zhang X-L, Xu X-J, Ling Z-Y, Jiang W, Tan C, Zhao J-J. Effect of pre-deformation on microstructure and properties of ultra-high-strength aluminium alloy extrusion. The Chinese Journal of Nonferrous Metals, 2016, 26(3): 507-513.',
   doi='', url='http://www.ysxbcn.com/down/2016/03_cn/04-p0507-33268.pdf'),
 'XU2018': dict(cit='Zhang C, Xu X-J, Zhang J, Huang P, Du D-H, Jia W-J. Effect of heating rate for solid solution on microstructure and properties of Al-10.78Zn-2.78Mg-2.59Cu aluminium alloy. Materials for Mechanical Engineering, 2018, 42(8): 24-27.',
   doi='10.11973/jxgccl201808005', url='https://jxgccl.mat-test.com/cn/article/pdf/preview/10.11973/jxgccl201808005.pdf'),
 'LI2020': dict(cit='Li J, Zhang Y, Cao X, Zeng Q, Zhuang Y, Qian X, Chen H. Accelerated discovery of high-strength aluminum alloys by machine learning. Communications Materials, 2020, 1: 73.',
   doi='10.1038/s43246-020-00074-2', url='https://www.nature.com/articles/s43246-020-00074-2'),
 'PMC9571267': dict(cit='Zhang W, et al. Effects of deformation at high, medium, and cryogenic temperatures on the microstructures and mechanical properties of Al-Zn-Mg-Cu alloys. 2022 (PMC9571267).',
   doi='', url='https://pmc.ncbi.nlm.nih.gov/articles/PMC9571267/'),
}

rows = []
def add(rid, sid, origin, loc, form, temper, sol_t, sol_h, age_t, age_h, pre, special,
        comp, ys, uts, el):
    r = dict(record_id=rid, source_id=sid, data_origin=origin, localization=loc,
             product_form=form, temper=temper, solution_temp_C=sol_t, solution_time_h=sol_h,
             aging_temp_C=age_t, aging_time_h=age_h, pre_deformation_pct=pre,
             special_process=special)
    for c in COMP: r[c] = round(float(comp.get(c, 0.0)), 4)
    r['YS_MPa'] = ys; r['UTS_MPa'] = uts; r['EL_pct'] = el
    s = SOURCES[sid]; r['citation'] = s['cit']; r['doi'] = s['doi']; r['url'] = s['url']
    rows.append(r)

# ---------- 1) Bhat 2023 : 215 7xxx rows ----------
_bp = DATA / 'bhat_al_data.csv'
b = pd.read_csv(_bp).rename(columns={list(pd.read_csv(_bp, nrows=0).columns)[0]: 'idx'})
temper_map = {
 'Solutionised  + Artificially peak aged':'peak-aged (T6)',
 'Solutionised + Artificially over aged':'over-aged (T7)',
 'No Processing':'as-fabricated (O)',
 'Solutionised + Naturally aged':'naturally-aged (T4)',
 'Artificial aged':'artificially-aged'}
bsev = b[(b.Zn*100 >= 3.0) & ((b.Zn*100) > (b.Cu*100))].copy()
for i, (_, x) in enumerate(bsev.iterrows()):
    comp = {c: round(float(x[c])*100.0, 4) for c in COMP if c in b.columns}
    add(f'BH-{i+1:03d}', 'BHAT2023', 'measured (published open dataset)',
        f'Bhat dataset row index {int(x["idx"])}', 'not specified',
        temper_map.get(x.Processing, x.Processing), '', '', '', '', '',
        f'processing class: {x.Processing}', comp,
        (None if pd.isna(x['Yield Strength (MPa)']) else float(x['Yield Strength (MPa)'])),
        (None if pd.isna(x['Tensile Strength (MPa)']) else float(x['Tensile Strength (MPa)'])),
        (None if pd.isna(x['Elongation (%)']) else float(x['Elongation (%)'])))

# ---------- 2) Xu group : 6 rows ----------
add('XU15-01','XU2015','measured','Table 5','extrusion','T652',470,2,121,24,2,
    'multi-stage solution (450/2+460/2+470/2); T652 2% pre-compression',
    {'Zn':12.45,'Mg':3.56,'Cu':1.12,'Zr':0.21,'Sr':0.0553}, None,720.5,5.0)
add('XU15-02','XU2015','measured','Table 5','extrusion','pre-recovery+T652',470,2,121,24,2,
    'pre-recovery (250/24+300/6+350/6+400/6); multi-stage solution; T652 2% pre-compression',
    {'Zn':12.45,'Mg':3.56,'Cu':1.12,'Zr':0.21,'Sr':0.0553}, None,699.1,4.2)
add('XU16-01','XU2016','measured','Table 5','extrusion','T6',470,2,121,24,0,
    'multi-stage solution (450/2+470/2)',
    {'Zn':12.45,'Mg':3.56,'Cu':1.12,'Zr':0.21,'Sr':0.0553}, 667.8,693.6,4.0)
add('XU16-02','XU2016','measured','Table 5','extrusion','pre-deformation+T6',470,2,121,24,2.5,
    'multi-stage solution; 2-3% pre-stretch',
    {'Zn':12.45,'Mg':3.56,'Cu':1.12,'Zr':0.21,'Sr':0.0553}, 682.3,738.6,5.0)
add('XU18-01','XU2018','measured','Table 4','extrusion','T6',470,2,121,24,0,
    '400/2 hold then slow heating at 3.6 C/h to 470; solution 470/2',
    {'Zn':10.78,'Mg':2.78,'Cu':2.59,'Zr':0.22,'Sr':0.047}, 661.9,708.7,8.1)
add('XU18-02','XU2018','measured','Table 4','extrusion','T6',470,2,121,24,0,
    '400/2 hold then fast heating at 180 C/h to 470; solution 470/2',
    {'Zn':10.78,'Mg':2.78,'Cu':2.59,'Zr':0.22,'Sr':0.047}, 649.1,694.1,7.4)

# ---------- 3) Li 2020 : 12 cast rows (UTS only) ----------
li = [('1-1',7.67,1.65,2.38,0.40,0.04,419),('1-2',7.85,1.68,2.38,0.38,0.05,432),
      ('1-3',7.57,1.68,2.38,0.38,0.07,474),('2-1',6.89,1.70,2.38,0.35,0.08,443),
      ('2-2',7.36,1.69,2.38,0.36,0.08,468),('2-3',6.20,1.71,2.37,0.35,0.08,477),
      ('3-1',6.08,1.72,2.35,0.38,0.08,439),('3-2',7.13,1.67,2.35,0.41,0.08,446),
      ('3-3',7.43,1.78,2.36,0.41,0.09,517),('4-1',8.20,1.82,2.37,0.40,0.10,528),
      ('4-2',7.84,1.86,2.37,0.40,0.10,549),('4-3',7.47,1.93,2.37,0.39,0.09,562)]
for j,(nm,zn,mg,cu,ti,y,uts) in enumerate(li):
    add(f'LI-{j+1:03d}','LI2020','measured','Table 1','cast ingot','iterative screening',
        '','','','','', 'cast screening slab (heat treatment not itemised in table)',
        {'Zn':zn,'Mg':mg,'Cu':cu,'Y':ti,'Ti':y}, None, float(uts), None)

# ---------- 4) PMC9571267 : 4 rows ----------
pmc_comp = {'Zn':7.5,'Mg':1.5,'Cu':1.4,'Zr':0.15}
add('PMC-01','PMC9571267','measured','Sect 3.3 (Fig.10)','forging','as-forged (AR)',
    '','','','','', 'as-received hot forging', pmc_comp, 161,245,20.2)
add('PMC-02','PMC9571267','measured','Sect 3.3 (Fig.10)','forging','two-step aged',
    475,4,120,4,'', '3-pass multiaxial forging at 450 C; solution 475/4; two-step aging 120/4+165/4',
    pmc_comp, 514,539,7.4)
add('PMC-03','PMC9571267','measured','Sect 3.3 (Fig.10)','forging','two-step aged',
    475,4,120,4,'', '3-pass multiaxial forging at 250 C; solution 475/4; two-step aging 120/4+165/4',
    pmc_comp, 495,533,10.2)
add('PMC-04','PMC9571267','measured','Sect 3.3 (Fig.10)','forging','two-step aged',
    475,4,120,4,'', '3-pass multiaxial forging at -180 C; solution 475/4; two-step aging 120/4+165/4',
    pmc_comp, 506,535,11.1)

df = pd.DataFrame(rows)[OUT_COLS]
df.to_csv(DATA / 'al7xxx_dataset_v4.csv', index=False, encoding='utf-8-sig')
print('v4 total rows:', len(df))
print('by source:', df.source_id.value_counts().to_dict())
for t in ['YS_MPa','UTS_MPa','EL_pct']:
    print(t, 'non-null:', df[t].notna().sum())
print('temper:', df.temper.value_counts().to_dict())
