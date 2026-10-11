# Traceable Data and Reproducible Code for Mechanical-Property Prediction of 7xxx Aluminum Alloys

This repository contains the **data, code, numerical results, source/provenance ledgers and reproduction instructions** that support a machine-learning study of yield strength (YS), ultimate tensile strength (UTS) and elongation (EL) for 7xxx (Al–Zn–Mg–Cu) aluminum alloys.

It is published to let editors, reviewers and readers **verify that the data are real, traceable and reproducible**. It does **not** contain the manuscript text, the response to reviewers, the cover letter, or the final publication-ready figures; those are supplied through the journal. The scientific figures can be regenerated from the result tables with the plotting scripts in `src/` (they write to a `figures/` folder when run).

## Repository layout

| Path | Contents |
|------|----------|
| `data/` | Archived and analysis-ready cohorts: the 237-record v4 archive, the 188-record conservative de-duplicated benchmark, the 98-record primary-checked cohort and the 135-record journal-compiled cohort. |
| `src/` | Feature engineering, leakage-safe group/nested resampling, ten fixed regressors, the equal-weight six-model fusion, grouped/permutation importance, paired tests, residual strata, series proxies, external challenges, figure generation and integrity checks. |
| `results/` | Unrounded predictions, per-fold metrics, nested/paired/stratified outputs, series-proxy results and the two independent external challenges (`v5/`, `v6/`, `traceable/`). |
| `supplementary/` | The source/data/results workbook `Supplementary_Table_S1_Final.xlsx` (40 data, provenance and metric worksheets), plus per-row primary-source evidence and Bhat crosswalks. |
| `verification/` | Source-recheck, provenance and repository audits, frozen numerical ledgers, platform refits, environment files and integrity manifests. |
| `requirements.txt` | Pinned Python dependencies. |
| `SHA256SUMS.txt` | SHA-256 manifest of every distributed file; verify with `src/verify_delivery.py`. |

## Dataset cohorts

| Cohort | Records | YS/UTS/EL labels | Evidence level |
|---|---:|---:|---|
| v4 archived dataset | 237 | — | Filtered archive with per-row source labels and conservative exclusions. |
| Reviewed benchmark | 188 | 158 / 183 / 161 | Conservative numeric de-duplication of the archive; the historical reference analysis. |
| Primary-checked cohort | 98 | 44 / 98 / 86 | Values independently re-extracted from ten primary journal articles (eight study groups after combining one research line); no Bhat records. |
| Journal-compiled cohort | 135 | 123 / 134 / 131 | Published compilation (Yuan et al., 2024, *J. Mater. Inf.* 4:23, DOI 10.20517/jmi.2024.34, CC BY 4.0), Zn ≥ 3 wt.% and Zn > Cu, with original-study citations. |

Two later external challenges (not used for fitting) are included: Yin et al., 2025 (DOI 10.3390/ma18020391) and Wang et al., 2026 (DOI 10.3390/met16080906).

## Method

- Model inputs: the archival and primary cohorts use 32 raw inputs (18 chemistry fields, five numeric processing fields, nine categorical indicators), expanded to 45 inputs by 13 deterministic descriptors; a matched reported-state sensitivity adds eight documented fields (53 inputs). The journal compilation uses 22 composition inputs, or 27 after five deterministic composition descriptors.
- Process context vs. inputs: ordered solution/aging stages, multi-step and RRA sequences and quench conditions are preserved in the provenance ledgers; they are not all independent numeric columns. Flat temperature/time fields encode the final explicitly stated hold, and a dimensionless Kelvin-based exposure proxy (log-sum-exp over known positive aging holds) summarizes aging exposure. Quench conditions are not a separate set of numeric model inputs and are not claimed to be fully modeled.
- Resampling: chemistry-group and primary-source-group separation, nested outer/inner splits, and leave-one-primary-source-out evaluation; standardization and any data-dependent step are fitted inside each training fold.
- Models: a training-mean baseline, ordinary linear regression, ridge regression, a single decision tree, random forest, ExtraTrees, XGBoost, LightGBM, gradient boosting and radial-basis support vector regression. The fixed six-model average combines RF, ET, XGB, LGB, GBRT and SVR with equal weights. The historical selection of the fixed settings is undocumented; a separate nested analysis evaluates representation and hyperparameter selection using outer-training data only. No adaptive weights or stacking are fitted on the test fold.
- Interpretation: normalized squared-error-reduction importance with grouped permutation, plus paired model comparisons, residual strata and applicability-domain summaries.

## Reproduce

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt

# Check that every distributed file is present and byte-identical (read-only)
python src/verify_delivery.py

# Validate the committed numerical ledgers, or recompute everything from scratch
python src/reproduce_final.py
python src/reproduce_final.py --all
```

The runner regenerates the scientific figures from the result CSVs into `figures/`. Reported metrics use unrounded prediction archives. ET and nested results reproduce to numerical precision; XGBoost can show platform-dependent prediction differences (up to 23.05 MPa for YS; see Supplementary Table S2), so bit-identical cross-platform refitting is not promised.

## Evidence and provenance

- `supplementary/Supplementary_Table_S1_Final.xlsx` records source summaries, per-record provenance, the field dictionary, evidence ledgers, model matrices, metrics, external challenges and the primary-source and journal-compilation ledgers.
- `supplementary/primary_traceable_evidence.csv`, `primary_input_aliases.csv`, `primary_traceable_exclusions.csv` and `Bhat_public_row_crosswalk.csv` give per-row source locators, alias groups, repeat exclusions and the public-row crosswalk.
- `verification/source_recheck.json`, `local_provenance_audit.json` and `author_repository_audit.json` document independent source checks, the unresolved limitation that the secondary Bhat corpus lacks per-specimen original identities, and the repository audit. This limitation is reported rather than papered over.
- All held-out predictions, fold assignments, group masks, fixed-model averages and platform-comparison ledgers are provided so every reported metric can be recomputed.

## Source licensing and privacy

- The 98-record cohort is an independent transcription for verification from the cited primary articles; the article full texts are not redistributed.
- The 135-record compilation comes from a CC BY 4.0 release; preserve its original citation.
- Bhat et al. (Mendeley Data, DOI 10.17632/b6br4yk6r3.1; *J. Mater. Sci.* 2024, DOI 10.1007/s10853-023-09317-2) provides 215 public rows; only the published row numbers and values are reproduced, with no fabricated specimen or experimental-series identities.
- No private laboratory data, personal information or unpublished manuscript content is included.

## Versioned snapshots

- Baseline development snapshot: tag `v0.9.0`, commit `d742856f90dce2435b2fe4e2499bd2d87dacb6c2`.
- Cited baseline data/code release: tag `v1.0.0`, commit `31419e74f449230983e062a60491d1c379604544` (see GitHub Releases for the fixed archive and SHA-256 manifest).
- README-corrected release: tag `v1.0.1` (see GitHub Releases); corrects the README model list, input description, licensing and recomputation text. The v1.0.0 tag and commit are not moved.
- Analysis code is released under the MIT License; third-party data retain their source licenses.
