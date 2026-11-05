# Condition matched precipitation evidence for the second revision

The checks and ledgers in this directory are complete. Eight existing primary-cohort conditions from two publications form four pairs. Every pair has an identical un-imputed 45-input vector and distinct 53-input vectors. No published property value was changed, and no phase descriptor was added to a fitted model. These are retrospective diagnostic illustrations within the existing primary cohort, not additional independent validation studies.

## Verified findings

| Pair | Omitted reported history that separates the 53-input vectors | Reported YS in A and B, MPa | Reported UTS in A and B, MPa | Reported EL in A and B, percent |
|---|---|---|---|---|
| Tai NA-0d and NA-7d, both 120 °C for 24 h | Prior natural aging 0 and 7 days | 545 ± 8 and 539 ± 6 | 598 ± 9 and 597 ± 7 | 15.4 ± 0.5 and 15.3 ± 0.4 |
| Orłowska HE10 and HE8, naturally aged | Applied HE strains 1.41 and 1.85, both naturally aged 180 days | 582 ± 2 and 552 ± 33 | 674 ± 2 and 608 ± 53 | 11.0 ± 0.2 and 4.7 ± 2.8 |
| Orłowska HE10 HT1 and HE8 HT1, both 100 °C for 24 h | Applied HE strains 1.41 and 1.85 | 603 ± 11 and 588 ± 24 | 652 ± 8 and 625 ± 18 | 8.8 ± 0.5 and 7.5 ± 1.9 |
| Orłowska HE10 HT2 and HE8 HT2, both 120 °C for 24 h | Applied HE strains 1.41 and 1.85 | 608 ± 22 and 563 ± 16 | 639 ± 19 and 590 ± 11 | 9.3 ± 0.2 and 5.4 ± 1.1 |

The Tai tensile spreads are SDs, and the Methods specify at least three specimens for each artificial-aging condition. The Orłowska Methods specify three longitudinal tensile specimens per state but do not explicitly define the statistic represented by Table 3 tensile ± values. Retain these as reported spreads without converting them into SDs, SEs or CIs. No significance test between conditions is inferred from the spreads.

## Supplementary evidence text

Condition matching was performed using the original article designations, processing tables and existing primary-cohort record IDs. For each pair, equality was checked across all 45 exported un-imputed input columns, retaining missingness. The same comparison across the 53 columns identifies the reported history fields that distinguish the conditions. Equality at the representation level is distinct from equal material state. Reported tensile spreads are retained; they are not prediction intervals. The minimum pair-constant RMSE in the accompanying CSV is one-half the absolute difference between two reported means. It is a post-outcome empirical diagnostic, not an independently fitted predictor, an experimental significance test or an estimated population error floor.

Tai et al. [26] provide a phase-state comparison for NA-0d and NA-7d following artificial aging at 120 °C for 24 h. Table 2 on p.5 supplies the tensile summaries. Section 3.4 and Fig.4c,d on pp.7–8 report edge-on η′ lengths of 3.46 ± 0.90 and 4.96 ± 1.62 nm; η₂ is identified in the latter condition using the source's phase analysis in Section 3.3/Fig.3. The quoted size passage does not explicitly define the size-spread statistic or particle count, so neither is inferred. The strength means are close despite distinct reported phase observations. This is evidence that near-equal tensile response cannot verify a model's recovery of phase state; it is not evidence for a large phase-caused strength difference. Table 2 also first attains its maximum reported NA-7d YS at 16 h, while NA-0d first attains its maximum at 24 h. These are discrete tested maxima, not continuous optimal aging times or new kinetic estimates. The source explicitly could not identify the natural-aging clusters as GPI zones; no confirmed-GP label is assigned here.

Orłowska et al. [27] provide paired deformation/aging states. Methods Section 2.1 identifies single-step HE10 with strain 1.41 and two-step HE8 with strain 1.85. Table 2 provides the 180-day natural-aging, 100 °C/24 h and 120 °C/24 h designations, and Table 3 supplies tensile summaries. Section 3.1.2, Figs.7–9 and Fig.10 report phase observations and context. The source describes greater precipitate growth and boundary precipitation in the higher-strain series and PFZs at 120 °C for HE8; it also reports differences in grain and dislocation structure. No numerical precipitate size, fraction, density or PFZ width is created for these conditions. The source's PFZ width up to 30 nm belongs to its RRA conditions, which are not the HT2 conditions or the natural-aging pair, and is not transferred to them. The source's mechanistic interpretations remain source interpretations; these observations do not identify an independent contribution from precipitation rather than correlated grain, dislocation and processing changes.

For engineering data collection, these cases identify specific omissions to avoid when comparing aging schedules or deformation routes. Record the prior natural-aging duration and applied deformation history, and link phase measurements to their actual processing state and measurement basis. More deformation and more prior natural aging do not uniformly increase the reported tensile means across these conditions. The existing 53-input analysis evaluates reported processing information, not a new precipitate-feature model. A validated improvement from phase descriptors would require a sufficiently complete matched dataset and a separately designed validation study.

## Condition evidence overview

| Source conditions | Source phase evidence | Mechanical context | Representation diagnosis |
|---|---|---|---|
| Tai NA-0d/24 h and NA-7d/24 h [26] | η′ length 3.46 ± 0.90 versus 4.96 ± 1.62 nm; η₂ reported for NA-7d. TEM Fig.4c,d and Section 3.4, p.7; phase identification Section 3.3/Fig.3. | UTS 598 ± 9 versus 597 ± 7 MPa; different source-reported phase states coexist with similar strength means. | Same 45-input vector; separated by 0 versus 7 prior natural-aging days in 53 inputs. |
| Orłowska HE10 and HE8 [27] | TEM Figs.8a/9a label η/η′; phase sizes and counts are qualitative for these conditions. | UTS 674 ± 2 versus 608 ± 53 MPa after 180-day natural aging; grain/dislocation context differs. | Same 45-input vector; separated by applied HE strain 1.41 versus 1.85 in 53 inputs. |
| Orłowska HE10 HT1 and HE8 HT1 [27] | TEM Figs.8b/9b; source describes stronger precipitate growth in higher-strain series at equal aging temperature. | UTS 652 ± 8 versus 625 ± 18 MPa, both 100 °C/24 h. | Same 45-input vector; separated by applied HE strain in 53 inputs. |
| Orłowska HE10 HT2 and HE8 HT2 [27] | TEM Figs.7b,c, 8c/9c; source describes stronger growth/boundary precipitation in higher-strain series and PFZs at 120 °C for HE8. | UTS 639 ± 19 versus 590 ± 11 MPa; YS 608 ± 22 versus 563 ± 16 MPa; both 120 °C/24 h. | Same 45-input vector; separated by applied HE strain in 53 inputs. |

Table note. Conditions are existing primary records, not new experimental or independent validation units. Phase identities and trends are retained as reported by the original authors, without remeasurement. Tai tensile ± values are SDs; the definition of its η′ length ± values is not explicitly stated in the checked passage. Orłowska tensile ± values are retained as reported spreads. Co-varying grain/dislocation states preclude a causal strengthening decomposition. No quantitative RRA PFZ width is assigned to these conditions. The 53-input vector contains reported processing context, not the phase observations summarized here.

## Specific source checks and reconciliation

- Tai original PDF p.5 Table 2 and p.7 Section 3.4 were checked directly. Mechanical and η′ values match the ledger. Section 3.4 observes NA-7d clusters but explicitly does not confirm GPI identity.
- The Tai Fig.4 prose and caption do not use the same zone-axis label for the final-aging images. This ancillary direction label is not encoded or used to make a new phase identification. Retain the authors' reported phase/size conclusions without silently resolving that source inconsistency.
- Orłowska Table 3 values were parsed from the primary XML and checked field by field against all seven existing records for the article, including CG T6. All means and reported spreads match.
- The Orłowska results prose mentions HE10 elongation of 17%, while Table 3 gives 11.0 ± 0.2%. The existing data and this ledger preserve Table 3, which is the declared numerical source. This is a source discrepancy, not a value to replace using the prose.
- Orłowska source RRA1/RRA2 have qualitative phase/PFZ context but no tensile rows in Table 3. They are not added as property records and are not used to enlarge the diagnostic sample.
- The phase ledger represents condition-level TEM and tensile summaries; the publications do not establish per-specimen joint phase/property identities for this compilation. The particle counts and experimental repeat numbers must not be combined into a larger regression sample.

## Reproduction

Use the bundled Python runtime with the distributed data folder specified explicitly.

```text
python build_phase_evidence.py --package-dir PATH_TO_RELEASE --out NEW_OUTPUT_FOLDER
```

Optional close-reading checks against locally obtained lawful originals can be reproduced with `--verify-sources-dir PATH_TO_SOURCES`, which expects `tai2022.pdf` and `orlowska2022.xml` and requires pypdf/lxml. Those originals are not redistributed in the analysis package. The supplied `phase_evidence_verification.json` records the source-file SHA-256 values and all successful checks used for this revision.

## Files supplied

- `phase_evidence_ledger.csv` provides eight record-level observations and all source/uncertainty qualifiers.
- `phase_pair_diagnostics.csv` provides four pair comparisons and 45/53 input equality checks.
- `aging_schedule_context.json` preserves the tested-time basis for the Tai schedule observation.
- `phase_evidence_verification.json` records the source checks and input hashes.
- `build_phase_evidence.py` reconstructs the ledgers without fitting any model or generating new labels.

