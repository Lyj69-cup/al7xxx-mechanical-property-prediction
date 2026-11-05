# Focused R2 validation and representation evidence

Status on 5 November 2026: completed and independently audited, numerical checks PASS. This scientific note documents the frozen revision analyses. Source data and the archived analysis were not altered.

## Frozen scope and reproducibility

`revision_r2/src/primary_review_sensitivity.py` is self-contained and reads the archived matrices and source metadata. First run `--prepare`, then `--run`. With a source checkout elsewhere, pass `--source PATH --out OUTPUT_PATH`. The code uses numpy, pandas, scipy and scikit-learn from the study environment; no full-text downloads or proprietary model files are needed. `audit_review_sensitivity.py` independently checks all output predictions, group separation, means and metrics, and reproduces empirical collision SSE using a second grouping implementation.

All group assignments and six source-file SHA-256 values were written to `frozen_plan.json` and `metadata_group_assignments.csv` before these new fits. These are retrospective analyses introduced after review; they were not preregistered. No labels, model scores or test errors are used to create categories. One deterministic complete leave-group-out run is performed per protocol. Fixed ET uses 500 trees, max_features=0.5, random_state=42 and one thread, with 32 raw or 45 augmented inputs. No estimator/representation selection occurs. Target-specific predictor medians, all-training-missing constant-zero handling, predictor scaling and target scaling are fitted inside each training partition. Missing targets are never imputed. The baseline predicts the training mean for each target/fold. All unfavorable results remain.

212 ET fits and 106 training-mean calculations produced 3,534 held-out predictions. Full98 raw and augmented ET source-holdout R2/RMSE/MAE exactly match the archived primary-source reference (maximum metric difference 0). All six source-file hashes remain unchanged. Environment versions are numpy2.3.5, pandas2.2.3, scikit-learn1.8.0 and scipy1.17.0. `execution_metadata.json` and `audit_results.json` provide the details.

## Group definitions and estimands

| Population and protocol | Rows | YS/UTS/EL labels | Groups and scope |
|---|---:|---|---|
| Full primary cohort, source holdout | 98 | 44/98/86 | Eight article/research-line groups, with the three Xu articles linked. YS/EL have labels in seven groups. |
| Reported designation-family subset, family holdout | 76 | 36/76/76 | 7A09=42, 7A04=10, 7075=18 (both Tai and Orłowska held out together), 7055-based=6. Twenty-two custom/unresolved-designation records are excluded from both training and evaluation. |
| Same designation subset, source holdout | 76 | 36/76/76 | Five source groups on exactly the same eligible rows. |
| Reported aging-route subset, route holdout | 94 | 40/94/82 | One-stage artificial33, two-stage artificial43, RRA12, natural-only3, as-quenched1, annealed2. RRA has UTS only; YS/EL therefore have five evaluated route groups. |
| Same aging-route subset, source holdout | 94 | 40/94/82 | Eight source groups; YS/EL have labels in seven. |
| Largest study excluded, source holdout | 56 | 42/56/44 | All42 Bi7A09 records excluded from both training and evaluation, then seven remaining source groups rebuilt. YS/EL have labels in six. |

The 7055-based class includes the paper's explicitly Sc-modified conditions and does not certify that these variants satisfy a registered AA7055 designation. The source-title and condition evidence for every assigned family is recorded. No composition-only guess is assigned a designation.

The aging-route comparison separates broad **reported aging architectures**, not all aspects of the complete heat-treatment scheme. A blank aging-stage field is not taken to mean no aging; the explicit 121C/24h Xu holds are classified as one-stage artificial aging. The four excluded histories are Liu's two solutionized/aging-unspecified conditions and the as-received PMC/Tai conditions. Annealed prior schedules, natural-aging temperatures and any other unreported stages remain unknown. These distinctions are not inferred from strength or ductility.

Designation/route holdouts ask different transfer questions from publication holdouts. They do not additionally forbid an article from appearing on both sides if it spans multiple aging routes, and they do not eliminate all nearby compositions or shared protocols. The designation comparison has only four uneven groups, strongly confounded with publication, processing and coverage. Holding out a rare as-quenched route uses a single test label and has undefined per-group R2. Pooled OOF R2 remains defined. The same-population source comparisons prevent changes in evaluated rows being mistaken for protocol effects, but group difficulty, training counts and response distributions still differ.

## Complete pooled results

Each triplet follows YS/UTS/EL. Strength MAE is in MPa, elongation MAE in percentage points. No SD or confidence interval is attached to one deterministic pooled holdout vector. Per-group errors, n, signed bias and equal-group macro MAE/RMSE are retained in the CSV files. These values do not establish statistical significance or rank all possible learning methods.

| Population/protocol | Comparator | Pooled R2, YS/UTS/EL | MAE, YS/UTS/EL |
|---|---|---|---|
| primary98/source_loso | training_mean | -0.164/-0.285/-0.224 | 98.649/92.250/4.005 |
| primary98/source_loso | ET_raw32 | 0.006/-0.502/0.067 | 93.411/97.780/2.827 |
| primary98/source_loso | ET_augmented45 | 0.061/-0.301/-0.262 | 93.905/88.357/3.689 |
| designation76/designation_loso | training_mean | -0.156/-0.674/-0.117 | 91.314/105.210/3.279 |
| designation76/designation_loso | ET_raw32 | -0.204/-0.267/-0.180 | 108.907/82.768/3.009 |
| designation76/designation_loso | ET_augmented45 | -0.281/-0.428/-0.366 | 112.837/89.824/3.396 |
| designation76/source_loso | training_mean | -0.246/-0.665/-0.183 | 95.933/104.013/3.369 |
| designation76/source_loso | ET_raw32 | -0.294/-0.260/-0.260 | 107.431/82.418/3.143 |
| designation76/source_loso | ET_augmented45 | -0.330/-0.304/-0.414 | 109.355/85.573/3.462 |
| aging_route94/aging_route_loso | training_mean | -1.722/-0.610/-0.176 | 163.795/103.270/3.728 |
| aging_route94/aging_route_loso | ET_raw32 | -1.492/-0.372/0.124 | 153.364/100.031/3.214 |
| aging_route94/aging_route_loso | ET_augmented45 | -2.781/-0.411/-0.248 | 194.621/100.194/4.073 |
| aging_route94/source_loso | training_mean | -0.186/-0.394/-0.277 | 85.289/93.180/4.079 |
| aging_route94/source_loso | ET_raw32 | -0.278/-0.466/0.373 | 98.564/91.675/2.272 |
| aging_route94/source_loso | ET_augmented45 | -0.299/-0.388/0.214 | 100.170/87.246/2.830 |
| without_dominant56/source_loso | training_mean | -0.215/-0.256/-0.334 | 84.363/86.779/4.610 |
| without_dominant56/source_loso | ET_raw32 | 0.116/-0.650/0.141 | 69.190/91.091/3.534 |
| without_dominant56/source_loso | ET_augmented45 | 0.155/-0.501/-0.002 | 67.377/87.299/4.034 |

The results do not form a universal monotonic ordering of validation protocols. In the aging-route subset, augmented ET YS is especially poor (R2=-2.781; MAE194.621MPa); raw ET differs substantially. Removing the largest source does not establish transferable UTS performance (augmented ET R2=-0.501). Negative R2 must be read with MAE and the training-mean comparator; it is not proof that a single omitted physical variable causes the failure. These are fixed-comparator sensitivities, not fully nested estimates of an optimally selected model.

## Exact input-collision diagnostic

Equality is tested across the entire **unimputed** numerical vector, with an identical missingness pattern required. The implementation uses direct tuple equality rather than a finite-width feature hash; a second pandas groupby implementation independently recovers the same SSE. In the98-row primary cohort, the45-input matrix contains85 unique vectors and11 collision groups involving24 summaries. All24 have each target. The maximum within-vector reported YS/UTS/EL spans are145MPa,100MPa and11.5points.

For a collision group g and target y, the minimum squared error of any deterministic constant prediction on that exact vector is SSE_g=sum_i(y_i-mean_g(y))^2. Across all collision groups, the empirical SSE minima are13,073.1853MPa^2 (YS),8,410.2333MPa^2 (UTS) and105.4995points^2 (EL). Conditional on the24 collision summaries, the oracle minimum RMSE is23.339MPa,18.720MPa and2.097points. Across all44/98/86 labeled rows, allowing singleton vectors to be fitted exactly, these become17.237MPa,9.264MPa and1.108points. The conditional denominator and full-cohort denominator answer different questions and must not be mixed.

The53-input matrix separates all98 rows into unique unimputed vectors. Its empirical interpolation SSE bound is consequently zero, because each vector is a singleton. This is **not** evidence of zero predictive error, sufficiency for future conditions, measured microstructure, or a generalization benefit. The added variables are reported processing/geometry histories. Missing fields remain missing in the matrices and are explicitly tabulated in `input_missingness.csv`. The earlier matched53-feature transfer analysis remains necessary when discussing actual predictive performance.

These oracle calculations use the test labels for a post hoc algebraic diagnostic only. They are neither trained predictions nor population error ceilings. Reported condition means, measurement variability, rounding and correlated interventions may all contribute to within-vector differences; the diagnostic does not attribute their magnitudes to any single omitted mechanism. Source-reported SDs are not substituted for prediction uncertainty and are not used to infer independent measurement identities.

