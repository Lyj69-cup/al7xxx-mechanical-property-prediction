# Multi-Model Ensemble Learning for Mechanical Property Prediction of 7xxx-Series Aluminum Alloys

A fully traceable, leakage-free machine-learning framework that maps the **composition and heat-treatment** of 7xxx-series (Al–Zn–Mg–Cu) aluminum alloys to three mechanical properties: **yield strength (YS)**, **ultimate tensile strength (UTS)**, and **elongation (EL)**. Every record in the dataset can be traced, row by row, to a published open dataset or a peer-reviewed paper (DOI/URL). No interpolated, imputed, or synthetic records are used.

## Why this repository

Most published ML property models (i) mix records from the same source across training and test folds (data leakage that inflates accuracy), and (ii) rest on datasets that cannot be independently verified. This repository addresses both: it provides a clean, 100% traceable dataset and a strict evaluation protocol that reports both the same-distribution (random fold) performance and the harder, cross-study (leave-one-source-out) transfer performance — including the direction in which transfer fails.

## Dataset

- **237 records from 6 independent sources**, each carrying a citation, DOI (where available), and URL.
- Inputs: 17 composition elements + 5 numeric heat-treatment parameters + 9 one-hot temper/product-form indicators + 13 physics-derived features = **44 model inputs**.
- Targets: YS (MPa), UTS (MPa), EL (%).

| Source | Records | Reference | DOI / URL | Licence |
|---|---|---|---|---|
| BHAT2023 | 215 | Bhat, N.; Barnard, A.; Birbilis, N. *Aluminium alloy dataset for supervised learning.* Mendeley Data, V1, 2023 | [10.17632/b6br4yk6r3.1](https://data.mendeley.com/datasets/b6br4yk6r3/1) | CC BY 4.0 |
| XU2015 | 2 | Tan, C.; Xu, X.-J.; et al. *Chin. J. Nonferrous Met.* 2015, 25(11), 3019–3025 | [PDF](http://www.ysxbcn.com/down/2015/11_cn/08-p3019-33029.pdf) | © authors |
| XU2016 | 2 | Zhang, X.-L.; Xu, X.-J.; et al. *Chin. J. Nonferrous Met.* 2016, 26(3), 507–515 | [PDF](http://www.ysxbcn.com/down/2016/03_cn/04-p0507-33268.pdf) | © authors |
| XU2018 | 2 | Zhang, C.; Xu, X.-J.; et al. *Mater. Mech. Eng.* 2018, 42(8), 24–28 | [10.11973/jxgccl201808005](https://jxgccl.mat-test.com/cn/article/pdf/preview/10.11973/jxgccl201808005.pdf) | © authors |
| LI2020 | 12 | Li, J.; et al. *Communications Materials* 2020, 1, 73 | [10.1038/s43246-020-00074-2](https://www.nature.com/articles/s43246-020-00074-2) | CC BY 4.0 |
| PMC9571267 | 4 | Zhang, W.; et al. *Materials* 2022, 15, 6955 | [PMC9571267](https://pmc.ncbi.nlm.nih.gov/articles/PMC9571267/) | CC BY 4.0 |

The 215 Bhat records are the 7xxx subset (objective screen Zn ≥ 3 wt.% and Zn > Cu) of 1154 aluminium records; the six Xu-group records document pre-recovery, multi-stage solution, T652 and pre-deformation processing of ultra-high-Zn alloys; the 12 Li-2020 records are iteratively designed cast alloys (UTS only); the four PMC records cover high/medium/cryogenic-temperature forging.

## Method

- **Base models:** Dummy, Linear, Ridge, Decision Tree (baselines); Random Forest, ExtraTrees, XGBoost, LightGBM, Gradient Boosting, SVR.
- **Fusion:** leakage-free **equal-weight average** of the six non-linear base models (RF, ET, XGB, LGB, GBRT, SVR). No data-dependent weights or stacking meta-learner.
- **Validation:**
  - Random 5-fold CV, averaged over five seeds — same-distribution benchmark.
  - Leave-one-source-out (LOSO), run in **both directions**: (engineering) train on the large Bhat library and predict each independent study; (reverse) hold Bhat out and extrapolate from the 22 detailed-process records.
- All scaling is fit **inside each fold**; hyperparameters are fixed in advance and never tuned on the test fold, so there is no target leakage.

## Results (honest reporting)

Random 5-fold CV (R², mean over five seeds):

| Model | YS | UTS | EL |
|---|---|---|---|
| ExtraTrees (best single) | **0.917** | **0.898** | 0.620 |
| XGBoost | 0.900 | 0.873 | **0.669** |
| Fusion | 0.894 | 0.874 | 0.639 |

LOSO cross-study transfer (pooled Fusion):

| Direction | YS | UTS | EL |
|---|---|---|---|
| Engineering: Bhat → independent studies | 0.954 (n=8) | 0.803 (n=22) | 0.619 (n=10) |
| Reverse: 22 records → Bhat (extrapolation) | −0.867 | 0.264 | −1.518 |

Two deliberately reported, non-flattering findings: the physics-derived features add only a small gain over raw features on ExtraTrees (**+0.0054 R²**), and the equal-weight average does **not** beat the best single model (ET) under random CV (**−0.0045 R²**). Elongation is consistently the hardest target, and R² is unreliable for the very small per-source groups (n = 2), where MAE/RMSE should be read instead.

## Project structure

```
├── data/
│   ├── al7xxx_dataset_v4.csv        # 237-record traceable dataset (main file)
│   ├── bhat_al_data.csv             # raw Bhat open dataset (1154 records)
│   ├── xu_group_extracted.csv       # Xu-group extracted records
│   ├── li2020_extracted.csv         # Li 2020 extracted records
│   ├── pmc9571267_extracted.csv     # PMC open-access records
│   └── lab_data_template.csv        # template for your own alloys
├── src/
│   ├── build_v4.py                  # rebuild the dataset from raw sources
│   ├── recompute_v4.py              # full CV + LOSO analysis (main engine)
│   ├── family_perm_v4.py            # feature-family permutation importance
│   ├── make_v4_figs.py              # render all figures
│   ├── gen_supplementary_v4.py      # build Supplementary Table S1
│   └── predict_v4.py                # predict new alloys (batch / interactive)
├── results/
│   ├── Fig1_parity_random.png       # predicted vs measured (3 panels)
│   ├── Fig2_random_vs_LOSO.png      # random-fold vs cross-study R²
│   ├── Fig3_per_source_MAE.png      # per-source transfer error
│   ├── Fig4_feature_importance.png  # ExtraTrees feature importance
│   ├── recompute_v4_results.json    # all metrics
│   ├── family_permutation_v4.json   # feature-family permutation results
│   ├── oof_random_v4.csv            # per-sample out-of-fold predictions
│   └── oof_loso_v4.csv
├── supplementary/
│   └── Supplementary_Table_S1_Data_Sources_v4.xlsx
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt

# 1. Reproduce the full analysis (random CV + bidirectional LOSO)
python src/recompute_v4.py

# 2. Render the four figures from the out-of-fold predictions
python src/make_v4_figs.py

# 3. Feature-family permutation importance
python src/family_perm_v4.py

# 4. Predict your own alloys from a CSV
python src/predict_v4.py --input my_alloys.csv --output predictions.csv
#    or enter one alloy interactively
python src/predict_v4.py

# 5. (Optional) Rebuild the dataset and the supplementary table
python src/build_v4.py
python src/gen_supplementary_v4.py
```

The input CSV for prediction uses measured composition in wt.% and heat-treatment parameters; see `data/lab_data_template.csv`. Missing columns are filled automatically with safe defaults.

## Citation

If you use this dataset or code, please cite the accompanying paper:

```
Liu, Y.; Liu, C.; Zhu, L.; Xu, X. Multi-Model Ensemble Learning for Mechanical
Property Prediction of 7xxx-Series Aluminum Alloys. Journal of Materials
Engineering and Performance (manuscript JMEP-26-09-51934), 2026.
```

## Acknowledgements and data rights

The Bhat, Li-2020 and PMC datasets are redistributed under **CC BY 4.0**; please cite the original works. The Chinese-language Xu-group papers remain the copyright of their respective publishers and are linked (not reproduced) here, with only the tabulated numerical records transcribed for analysis.

## License

Code released under the [MIT License](LICENSE).
