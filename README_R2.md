# Data and code snapshot for the second JMEP revision

Version v1.1.0 supports the second revision of *Machine Learning Prediction of Mechanical Properties in Al–Zn–Mg–Cu Alloys: Validation and Representation Limits*. It preserves the original v1.0.0 snapshot at commit c8d0da322c3d10ea2775ee26b31f4ca4db532798 byte for byte and adds the explicitly identified `revision_r2/` analyses. `README.md` and `SHA256SUMS.txt` describe that unchanged baseline. This file describes the additions, and `R2_SHA256SUMS.txt` covers the complete v1.1.0 delivery except itself.

Original public release https://github.com/Lyj69-cup/al7xxx-mechanical-property-prediction/releases/tag/v1.0.0

This snapshot contains source data, code and frozen scientific outputs. It does not contain original journal full texts, the manuscript, reviewer reports, response letters, private working logs or access credentials. A DOI is not asserted by this file. Cite the real version-specific archive DOI once the corresponding record has been published and independently checked.


Numeric reference IDs in the frozen baseline files and in source-derived R2 ledgers or scripts retain the original v1.0.0 manuscript numbering. The complete 30-reference mapping in `revision_r2/evidence/manuscript_reference_crosswalk.csv` identifies the final revised-manuscript numbers. For example, Tai 26 maps to 15 and Orlowska 27 maps to 16. Source IDs, DOIs, property values and analysis inputs are unchanged.

## Revision additions

- `revision_r2/src/` contains portable analysis, independent result audit, phase-evidence reconstruction, package verification and an output-safe reproduction wrapper.
- `revision_r2/results/` contains 3,534 held-out predictions, the metadata-only frozen plan and group assignments, fold memberships, pooled/per-group metrics, missingness and exact input-collision diagnostics.
- `revision_r2/evidence/` contains eight condition-level phase/property observations forming four paired diagnostics, source locators and uncertainty qualifications, discrete aging-schedule context, scientific notes and a mixed-license notice.
- `revision_r2/verification/` contains independent numerical checks, source-evidence checks, execution versions, baseline preservation and packaging/portability reports.

The focused fixed-comparator analysis uses the 98-record primary cohort, a 76-record reported designation-family subset, a 94-record reported aging-route subset, and a 56-record cohort after excluding the dominant 42-record article. Six population/protocol pairs include same-population publication/research-line holdout references. The comparators are training mean and ExtraTrees with 32 raw or 45 augmented inputs. ET has 500 trees, `max_features=0.5`, state 42 and one thread. Target-specific predictor imputation, input scaling and target scaling are fitted only to each training partition. Missing property labels are not imputed. Group definitions and input hashes were frozen before the new fits, after review; this was not a preregistered study. All adverse results are retained.

Designation families and broad aging-route architectures ask different transfer questions from publication holdout and do not additionally enforce publication separation. Unresolved designations and histories are excluded from both sides of the corresponding comparison. The 7055-based group includes explicitly Sc-modified conditions without certifying a registered designation. Uneven and sometimes single-label groups, source/processing confounding and one deterministic set of holdouts limit interpretation. See the scientific notes for exact denominators, same-population comparisons and uncertainty.

The 45-input primary matrix has 11 exact input-collision groups covering 24 summaries. Their reported property spans reach 145 MPa YS, 100 MPa UTS and 11.5 elongation points. The conditional sample constant-prediction RMSE minima are 23.339 MPa, 18.720 MPa and 2.097 points. These post-outcome algebraic diagnostics are not trained predictors, population error ceilings or causal effects. The 53-input matrix distinguishes all 98 rows, which does not establish zero prediction error or transferable performance.

The phase cases are existing conditions from Tai et al. (10.3390/met12101766) and Orłowska et al. (10.3390/ma15134577). Their source-reported phase evidence is retained without remeasurement or independent phase identification. All four pairs share a 45-input vector but differ in the 53-input reported-history representation. No phase descriptor was added to a fitted model, no new property label or experiment was generated, and no causal strengthening contribution or predictive gain is claimed. Source TEM/property summaries are condition-level observations, not authenticated per-specimen joint measurements.

## Verify and reproduce

Use Python and the pinned dependencies in `requirements.txt`. The focused revision fits require numpy, pandas, scipy and scikit-learn. Optional rechecking against lawfully obtained source full texts also requires pypdf and lxml; those original files are not distributed.

Run from the package root.

```text
# Read-only standard-library integrity check of both manifests
python revision_r2/src/verify_revision_package.py --package-dir .

# Read-only independent audit of frozen revision predictions and metrics
python revision_r2/src/audit_review_sensitivity.py --package-dir . --results-dir revision_r2/results

# Audit frozen results and reconstruct phase ledgers into a NEW directory
python revision_r2/src/reproduce_revision.py --package-dir . --out ../JMEP_R2_audit --audit-only

# Refit focused revision ET comparisons and reconstruct phase ledgers
python revision_r2/src/reproduce_revision.py --package-dir . --out ../JMEP_R2_recomputed
```

The output directory must not exist and must be outside the package. The wrapper checks package integrity before and after execution. It never overwrites frozen results. Full reconstruction rechecks the metadata-only frozen plan before fitting. Execution time and report metadata can differ across runs; numeric results should be assessed with documented software/platform limits rather than a promise of identical machine-specific files.

To recheck the source passages as well, separately obtain the cited originals lawfully and run the following into another new output folder.

```text
python revision_r2/src/build_phase_evidence.py --package-dir . --out ../JMEP_R2_source_check --verify-sources-dir PATH_TO_LOCAL_ORIGINALS
```

That optional directory must contain `tai2022.pdf` and `orlowska2022.xml`. The distributed verification ledger records hashes for the particular originals checked during revision. The source publications, rather than this compilation, remain authoritative for phase identification and experimental details.

## Provenance and licensing

The complete baseline README and ledgers remain applicable. Bhat-derived archival entries lack original measurement, specimen and experimental-series identities and record-specific upstream extraction types. The 98-record primary cohort supplies article/table/text traceability; the 135-record compilation supplies citation-level provenance. The new analyses do not recover missing experimental identities. Historical fixed-setting selection is undocumented. Baseline XGBoost and the six-model average can show documented compiled-platform differences; bit-identical cross-platform refitting is not promised.

Analysis code is MIT-licensed under `LICENSE`. Third-party data retain their original licenses and attribution requirements. The Bhat Mendeley Data V1 release (10.17632/b6br4yk6r3.1) and Yuan et al. compilation (10.20517/jmi.2024.34) are identified as CC BY 4.0 sources. MIT is not applied to all third-party contents of this mixed package. Original journal full texts are not redistributed. Consult `revision_r2/evidence/mixed_license_notice.txt` and the record-level source ledgers.

## Version relationship

The original release v1.0.0 and its full commit identifier remain unchanged. Version v1.1.0 is an additive scientific revision snapshot. Use the actual published version DOI and this version identifier when citing the final revised-study package; the original release link alone does not identify the new revision outputs.
