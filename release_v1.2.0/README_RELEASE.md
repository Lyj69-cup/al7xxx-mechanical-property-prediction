# Data and code for validation and state representation in Al-Zn-Mg-Cu alloys

This distribution combines the original public numerical snapshot and the additional analyses supporting the manuscript "Validation and State Representation for Mechanical Property Prediction in Al–Zn–Mg–Cu Alloys". The proposed distribution version is 1.2.0. Its actual publication date, release URL and persistent identifier are supplied by the public repository record when published.

## Files and their roles

- `Original_Public_v1.1.0.zip`: the unchanged earlier public repository snapshot. It contains the original archival, primary and compilation analyses and the earlier revision diagnostics. Consult its README files for the original analysis commands and platform qualifications.
- `JMEP_R2_Complete_Local_Code_Candidate.zip`: the additional source-resolved nested analyses, state-field constraints, source-qualified phase records, reference-information evaluations, unrounded predictions and main-figure code/source data. The retained filename identifies the verified preparation snapshot; its numerical files are unchanged in this distribution.
- `Supplementary_Table_S1_Final.xlsx`: the source workbook accompanying the manuscript. The code archives contain additional condition ledgers and provenance qualifications. `reference_number_crosswalk.csv` in the additional archive maps historical source/reference labels to the revised manuscript numbering.
- `SHA256SUMS_PUBLIC.txt`: SHA-256 checksums for these two archives, the source workbook and this README.

The two numerical archives are complementary. Use the earlier snapshot for the original analyses and the additional snapshot for the new analyses. Original manuscript/reviewer correspondence and publisher full texts are not part of this upload set. Preparation-stage wording in historical audit files is retained as part of the audit history; the actual public release record defines the publication status and citation of this distribution.

## Reproduce the additional analyses

Extract `JMEP_R2_Complete_Local_Code_Candidate.zip` and open a terminal inside `JMEP_R2_Research_Code_Candidate/`. Use Python 3.12 or 3.13 in an isolated environment:

```text
python -m pip install -r requirements.txt
python run_reproduction.py
```

The default run verifies input checksums, refits the twelve frozen target/model pipelines, reconstructs feature matrices, and recomputes predictions, metrics, readout sensitivities and state-field subsets. It writes fresh results to `recomputed/`. Full nested-grid refitting is optional:

```text
python run_reproduction.py --nested
```

See the inner README for figure and reference-error diagnostic commands. Numerical tolerances apply; bitwise equality across platforms is not promised.

## Provenance and scope

The primary cohort consists of 98 original-paper condition summaries. Source records, target locators, missing fields and aggregation qualifications are retained in the ledgers. Bhat-derived archival entries do not have recovered original measurement, specimen or experimental-series identities. They remain separate from the original-paper primary cohort. No new physical experiments or recovered specimen identities are claimed.

The reference-information evaluation is retrospective and records prior project exposure, reference costs, all query controls and reading/rounding assumptions. It is not an untouched prospective validation. Condition-level literature summaries are not automatically independent specimen replicates.

## Licences and attribution

Original analysis code is licensed under the MIT License included in each applicable archive. Third-party numeric observations retain their original source licences and attribution requirements; the MIT licence does not apply to third-party publications or experimental data. See the included LICENSE, ATTRIBUTION.md and source ledgers. This distribution does not grant additional rights over third-party material.

## Citation

Use the creators, distribution title, version and version-specific DOI printed on the final Zenodo record. The version-specific DOI should identify this exact set of files. The manuscript is under revision; no article DOI or acceptance is asserted here.
