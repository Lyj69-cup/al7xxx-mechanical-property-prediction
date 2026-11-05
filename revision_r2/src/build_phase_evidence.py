"""Build condition-matched precipitation diagnostics from existing primary records.

No model is fitted, target is changed, phase quantity is imputed, or image digitized.
Optional local-original checks require lxml and pypdf. Original articles are not
distributed by this script. Run with --help for portable input paths.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


META = {"record_id", "source_id", "YS_MPa", "UTS_MPa", "EL_pct"}
TARGETS = ["YS_MPa", "UTS_MPa", "EL_pct"]
PAIR_IDS = {
    "Tai_NA_120C_24h": ["TAI2022_NA-015", "TAI2022_NA-016"],
    "Orlowska_HE_natural_180d": ["ORLOWSKA2022_HE-019", "ORLOWSKA2022_HE-022"],
    "Orlowska_HE_100C_24h": ["ORLOWSKA2022_HE-020", "ORLOWSKA2022_HE-023"],
    "Orlowska_HE_120C_24h": ["ORLOWSKA2022_HE-021", "ORLOWSKA2022_HE-024"],
}


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def as_index(rows):
    ans = {r["record_id"]: r for r in rows}
    assert len(ans) == len(rows), "Record IDs must be unique"
    return ans


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def equal_numeric(a, b):
    if a == b == "":
        return True
    if not a or not b:
        return False
    return float(a) == float(b)


def vec(row, columns):
    return [None if row[c] == "" else float(row[c]) for c in columns]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def phase_fields(record_id):
    if record_id.startswith("TAI"):
        prior = 0 if record_id.endswith("015") else 7
        return {
            "phase_observation": (
                "Edge-on eta-prime precipitate length reported as 3.46 +/- 0.90 nm."
                if prior == 0 else
                "Edge-on eta-prime precipitate length reported as 4.96 +/- 1.62 nm; eta-2 also identified."
            ),
            "phase_observation_locator": "PDF p.7, Section 3.4 and Fig.4c,d; phase identification Section 3.3/Fig.3, pp.5-6",
            "phase_measurement_method": "Source TEM; source HR-TEM/FFT for phase identification",
            "eta_prime_length_nm": 3.46 if prior == 0 else 4.96,
            "eta_prime_length_reported_spread_nm": 0.90 if prior == 0 else 1.62,
            "eta_prime_length_spread_definition": "Not explicitly defined for this size summary in the checked passage; do not assume SD or CI",
            "phase_particle_count": "",
            "phase_particle_count_qualification": "Not reported for the quoted size summaries",
            "source_interpretation": (
                "Authors interpret prior natural aging as accelerating subsequent eta-prime precipitation and shortening the time to the tested strength plateau."
            ),
            "property_spread_definition": "Standard deviation, per source Methods p.3",
            "tensile_repeat_count_qualification": "At least three specimens per artificial-aging condition, per source Methods p.3; no exact n assumed",
            "phase_scope_limit": "Do not label the unaged NA-7d clusters as confirmed GP zones; the source explicitly could not establish that identity."
        }
    code = {"019": "HE10", "022": "HE8", "020": "HE10 HT1", "023": "HE8 HT1", "021": "HE10 HT2", "024": "HE8 HT2"}[record_id[-3:]]
    fig = {"HE10": "Fig.8a", "HE8": "Fig.9a", "HE10 HT1": "Fig.8b", "HE8 HT1": "Fig.9b", "HE10 HT2": "Figs.7b and 8c", "HE8 HT2": "Figs.7c and 9c"}[code]
    obs = {
        "HE10": "Source TEM labels eta/eta-prime precipitates; naturally aged HE10 has fewer/smaller eta precipitates than the CG T6 reference.",
        "HE8": "Source TEM labels eta/eta-prime precipitates in the naturally aged HE8 condition; no condition-level numerical size/count distribution tabulated.",
        "HE10 HT1": "Source TEM labels eta/eta-prime precipitates; artificial aging increases precipitate size/number relative to natural aging, described qualitatively.",
        "HE8 HT1": "Source TEM labels eta/eta-prime precipitates; source comparison describes more pronounced precipitate growth in the higher-strain series at equal aging temperature.",
        "HE10 HT2": "Source TEM labels eta/eta-prime precipitates, with grain refinement and dislocation-state changes relative to CG T6; no numerical precipitate size distribution tabulated.",
        "HE8 HT2": "Source TEM labels eta/eta-prime precipitates; greater precipitate growth and grain-boundary precipitation are described for the higher-strain series; PFZs are observed at 120 C."
    }[code]
    return {
        "phase_observation": obs,
        "phase_observation_locator": "Section 3.1.2, " + fig + "; supporting phase context Fig.10 for CG T6/HE10/HE10 HT2/HE8/HE8 HT2",
        "phase_measurement_method": "Source TEM with diffraction; source SEM/EDS Fig.6 and XRD Fig.10 provide supporting context",
        "eta_prime_length_nm": "",
        "eta_prime_length_reported_spread_nm": "",
        "eta_prime_length_spread_definition": "No numerical condition-level eta-prime length used",
        "phase_particle_count": "",
        "phase_particle_count_qualification": "No numerical condition-level particle count used",
        "source_interpretation": "Authors discuss competition among grain-boundary/dislocation structure and precipitation, including boundary versus interior precipitation. This is their interpretation, not a coefficient identified by the present analysis.",
        "property_spread_definition": "Reported +/- values in Table 3; the checked Methods do not explicitly define the statistic for these tensile spreads",
        "tensile_repeat_count_qualification": "Three longitudinal tensile specimens per state, per source Section 2.1",
        "phase_scope_limit": "The up-to-30 nm PFZ width belongs to RRA conditions in the source and is not assigned to these HE natural/HT1/HT2 records."
    }


def verify_original_sources(sources, source_rows):
    """Check the transcribed numerical/locator claims against local originals."""
    from lxml import etree
    from pypdf import PdfReader

    tai_path = sources / "tai2022.pdf"
    orl_path = sources / "orlowska2022.xml"
    tai = PdfReader(tai_path)
    tai_text = [" ".join((p.extract_text() or "").split()) for p in tai.pages]
    checks = {}
    checks["tai_24h_tensile_row"] = "24 h 545 ± 8 598 ± 9 15.4 ± 0.5 539 ± 6 597 ± 7 15.3 ± 0.4" in tai_text[4]
    checks["tai_phase_length_no_NA"] = "3.46 ± 0.90 nm" in tai_text[6]
    checks["tai_phase_length_7d_NA"] = "4.96 ± 1.62 nm" in tai_text[6]
    checks["tai_eta2_at_24h"] = bool(re.search(r"equilibrium.*?phase was also found in the NA", tai_text[6]))
    checks["tai_tensile_repeat_lower_bound"] = "at least three samples" in tai_text[2]
    checks["tai_tensile_spread_is_SD"] = "standard devi" in tai_text[2]
    checks["tai_GP_not_confirmed"] = "it could not be concluded that they were GPI zones" in tai_text[6]

    root = etree.parse(str(orl_path))
    table = root.xpath('//table-wrap[label="Table 3"]/table')[0]
    table_rows = {}
    for tr in table.xpath('.//tbody/tr'):
        cells = [" ".join("".join(td.itertext()).split()) for td in tr.xpath('./td')]
        if cells:
            table_rows[cells[0]] = cells[1:]
    for rid in [k for k in source_rows if k.startswith("ORLOWSKA2022")]:
        row = source_rows[rid]
        if row["original_condition"] not in table_rows:
            continue
        vals = table_rows[row["original_condition"]]
        for target, v in zip(TARGETS, vals):
            numbers = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", v)]
            assert len(numbers) == 2, (rid, target, v)
            spread_col = target.split("_")[0] + "_spread"
            checks[f"orlowska_{rid}_{target}_mean_spread"] = numbers == [float(row[target]), float(row[spread_col])]
    all_text = " ".join("".join(root.getroot().itertext()).split())
    for label, token in {
        "orlowska_three_tensile_specimens": "For each state, three samples were tested",
        "orlowska_more_precipitate_growth": "more pronounced precipitate growth at the same aging temperature",
        "orlowska_PFZ_120C": "PFZs are already observed at an aging temperature of 120 °C",
        "orlowska_30nm_is_RRA_context": "What can be noticed is that RRA heat treatment led to over-aging",
    }.items():
        checks[label] = token in all_text
    assert all(checks.values()), {k: v for k, v in checks.items() if not v}
    return {
        "status": "PASS",
        "checks": checks,
        "original_source_files": [
            {"path": str(tai_path.resolve()), "sha256": digest(tai_path), "doi": "10.3390/met12101766"},
            {"path": str(orl_path.resolve()), "sha256": digest(orl_path), "doi": "10.3390/ma15134577"},
        ],
        "method": "Literal numeric/locator checks supplement close reading; source images were not remeasured or independently reidentified."
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-dir", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--verify-sources-dir", type=Path, help="Optional folder containing tai2022.pdf and orlowska2022.xml")
    args = ap.parse_args()
    out = args.out.resolve()
    package = args.package_dir.resolve()
    assert out != package and package not in out.parents, "Output must be outside the immutable package"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Output directory is not empty; existing results are never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    paths = [args.package_dir / "data" / f for f in ["primary_traceable.csv", "primary_state_enriched.csv", "primary_model_matrix_45.csv", "primary_model_matrix_53.csv"]]
    original = as_index(read_rows(paths[0]))
    enriched = as_index(read_rows(paths[1]))
    matrix45 = as_index(read_rows(paths[2]))
    matrix53 = as_index(read_rows(paths[3]))
    c45 = [k for k in next(iter(matrix45.values())) if k not in META]
    c53 = [k for k in next(iter(matrix53.values())) if k not in META]
    assert len(c45) == 45 and len(c53) == 53
    ledgers, pairs = [], []
    for pair_id, rids in PAIR_IDS.items():
        left, right = rids
        same45 = vec(matrix45[left], c45) == vec(matrix45[right], c45)
        same53 = vec(matrix53[left], c53) == vec(matrix53[right], c53)
        assert same45 and not same53, pair_id
        different = [c for c in c53 if not equal_numeric(matrix53[left][c], matrix53[right][c])]
        pair = {
            "pair_id": pair_id, "record_A": left, "record_B": right,
            "same_unimputed_45_vector": same45, "same_unimputed_53_vector": same53,
            "differing_53_fields": "; ".join(different),
        }
        for target in TARGETS:
            a, b = float(original[left][target]), float(original[right][target])
            pair[target + "_A_minus_B"] = a-b
            pair[target + "_minimum_pair_constant_RMSE"] = abs(a-b)/2
        pair["diagnostic_qualification"] = "Post-outcome empirical pair diagnostic; not a fitted predictor, causal effect, independent test, population error bound, or statistical-significance result"
        pairs.append(pair)
        for rid in rids:
            row, state = original[rid], enriched[rid]
            ledger = {
                "pair_id": pair_id, "record_id": rid, "source_id": row["source_id"],
                "manuscript_reference": 26 if rid.startswith("TAI") else 27,
                "doi": "10.3390/met12101766" if rid.startswith("TAI") else "10.3390/ma15134577",
                "original_condition": row["original_condition"],
                "solution_temp_C_encoded": row["solution_temp_C"],
                "solution_time_h_encoded": row["solution_time_h"],
                "aging_temp_C": row["aging_temp_C"], "aging_time_h": row["aging_time_h"],
                "prior_natural_aging_days": state["prior_natural_aging_days"],
                "reported_applied_HE_strain": state["reported_HE_strain"],
                "YS_MPa": row["YS_MPa"], "YS_reported_spread_MPa": row["YS_spread"],
                "UTS_MPa": row["UTS_MPa"], "UTS_reported_spread_MPa": row["UTS_spread"],
                "EL_pct": row["EL_pct"], "EL_reported_spread_percentage_points": row["EL_spread"],
                "property_locator": "Table 2, PDF p.5; Methods pp.2-3" if rid.startswith("TAI") else "Table 3; processing Table 2 and Methods Section 2.1",
                **phase_fields(rid),
                "same_45_vector_as_paired_condition": same45,
                "distinct_53_vector_from_paired_condition": not same53,
                "separating_reported_fields": "; ".join(different),
                "evidence_type": "Previously published condition-level observations; not a new experiment, new independent validation cohort, or per-specimen phase-property linkage",
                "source_qualification": "Source phase identifications are retained as reported, without independent remeasurement, diffraction reindexing, or quantitative strengthening decomposition",
            }
            for target in TARGETS:
                assert equal_numeric(row[target], matrix45[rid][target])
                assert equal_numeric(row[target], matrix53[rid][target])
            ledgers.append(ledger)

    timing = []
    for prior in [0,7]:
        selected = [r for r in original.values() if r["source_id"] == "TAI2022_NA" and r["original_condition"].startswith(f"NA-{prior}d/")]
        m = max(float(r["YS_MPa"]) for r in selected)
        peak = sorted(float(r["aging_time_h"]) for r in selected if float(r["YS_MPa"]) == m)
        timing.append({
            "prior_natural_aging_days": prior,
            "artificial_aging_temperature_C": 120,
            "tested_artificial_aging_times_h": sorted(float(r["aging_time_h"]) for r in selected),
            "maximum_reported_YS_MPa_among_tested_times": m,
            "tested_times_attaining_maximum_h": peak,
            "first_tested_time_attaining_maximum_h": min(peak),
            "qualification": "Discrete tested condition means, not a continuous optimum, new kinetic fit, or significance test; source Table 2 and Conclusions p.8",
        })
    write_csv(out / "phase_evidence_ledger.csv", ledgers)
    write_csv(out / "phase_pair_diagnostics.csv", pairs)
    write_json(out / "aging_schedule_context.json", timing)
    metadata = {
        "records": len(ledgers), "pairs": len(pairs), "unique_existing_primary_sources": 2,
        "phase_features_added_to_model": 0, "new_property_labels_generated": 0,
        "new_experiments": 0, "new_models_fitted": 0,
        "all_pairs_collide_in_45": all(p["same_unimputed_45_vector"] for p in pairs),
        "all_pairs_separate_in_53": not any(p["same_unimputed_53_vector"] for p in pairs),
        "phase_particle_counts_inferred": False,
        "property_values_unchanged": True,
        "input_files": [{"path": str(p.relative_to(args.package_dir)), "sha256": digest(p)} for p in paths],
        "original_source_checks": "NOT_REQUESTED",
    }
    if args.verify_sources_dir:
        metadata["original_source_checks"] = verify_original_sources(args.verify_sources_dir, original)
    write_json(out / "phase_evidence_verification.json", metadata)
    print(json.dumps({k: metadata[k] for k in ["records", "pairs", "all_pairs_collide_in_45", "all_pairs_separate_in_53", "property_values_unchanged"]}))


if __name__ == "__main__":
    main()
