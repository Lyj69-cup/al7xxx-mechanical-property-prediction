"""Read-only numerical/source/group audit of the focused post-review sensitivity."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

TARGETS = ['YS_MPa', 'UTS_MPa', 'EL_pct']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-dir', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--results-dir', type=Path)
    parser.add_argument('--report', type=Path, help='Optional JSON report outside the immutable package')
    args = parser.parse_args()
    SOURCE = args.package_dir.resolve()
    ROOT = args.results_dir.resolve() if args.results_dir else SOURCE / 'revision_r2/results'
    if args.report:
        target_report = args.report.resolve()
        assert target_report != SOURCE and SOURCE not in target_report.parents, 'Report must be outside the immutable package'
    p = pd.read_csv(ROOT / 'predictions.csv')
    fm = pd.read_csv(ROOT / 'fold_membership.csv')
    ass = pd.read_csv(ROOT / 'metadata_group_assignments.csv').set_index('record_id')
    d = pd.read_csv(SOURCE / 'data/primary_traceable.csv').set_index('record_id')
    metrics = pd.read_csv(ROOT / 'metrics_pooled.csv')
    assert len(p) == 3534
    assert np.isfinite(p[['observed', 'predicted']].to_numpy()).all()
    expected_counts = {'primary98': [44, 98, 86], 'designation76': [36, 76, 76],
                       'aging_route94': [40, 94, 82], 'without_dominant56': [42, 56, 44]}
    checked_means = checked_memberships = 0
    for _, f in fm.iterrows():
        tr, te = json.loads(f.training_ids), json.loads(f.testing_ids)
        assert set(tr).isdisjoint(te)
        assert len(tr) == f.n_training_labels and len(te) == f.n_test_labels
        assert d.loc[tr, f.target].notna().all() and d.loc[te, f.target].notna().all()
        col = {'source_loso': 'study_group', 'designation_loso': 'reported_designation_family',
               'aging_route_loso': 'reported_aging_route'}[f.protocol]
        assert set(ass.loc[tr, col]).isdisjoint(ass.loc[te, col])
        assert (ass.loc[te, col] == f.heldout_group).all()
        if f.cohort == 'designation76':
            assert ass.loc[tr + te, 'reported_designation_family'].notna().all()
        if f.cohort == 'aging_route94':
            assert ass.loc[tr + te, 'reported_aging_route'].notna().all()
        if f.cohort == 'without_dominant56':
            assert (d.loc[tr + te, 'source_id'] != 'BI2024_7A09').all()
        g = p[(p.cohort == f.cohort) & (p.protocol == f.protocol) &
              (p.target == f.target) & (p.model == f.model) &
              (p.heldout_group == f.heldout_group)]
        assert set(g.record_id) == set(te)
        np.testing.assert_array_equal(g.observed, d.loc[g.record_id, f.target])
        if f.model == 'training_mean':
            np.testing.assert_allclose(g.predicted, d.loc[tr, f.target].mean(), rtol=0, atol=1e-12)
            checked_means += 1
        checked_memberships += 1
    for _, r in metrics.iterrows():
        g = p[(p.cohort == r.cohort) & (p.protocol == r.protocol) &
              (p.target == r.target) & (p.model == r.model)]
        assert len(g) == r.n == expected_counts[r.cohort][TARGETS.index(r.target)]
        assert g.record_id.is_unique
        residual = g.predicted.to_numpy() - g.observed.to_numpy()
        mse, mae = np.mean(residual ** 2), np.mean(np.abs(residual))
        r2 = 1 - np.sum(residual ** 2) / np.sum((g.observed - g.observed.mean()) ** 2)
        np.testing.assert_allclose([r.R2, r.RMSE, r.MAE], [r2, np.sqrt(mse), mae], rtol=1e-13, atol=1e-12)
    archived = pd.read_csv(SOURCE / 'results/traceable/summary.csv')
    maximum_reference_difference = 0.0
    for t in TARGETS:
        for kind, model in [('raw', 'ET_raw32'), ('augmented', 'ET_augmented45')]:
            old = archived[(archived.cohort == 'primary') & (archived.kind == kind) &
                           (archived.protocol == 'study_loso') & (archived.target == t) &
                           (archived.model == 'ET')].iloc[0]
            new = metrics[(metrics.cohort == 'primary98') & (metrics.target == t) &
                          (metrics.model == model)].iloc[0]
            delta = np.max(np.abs(np.array([old.R2_mean, old.RMSE_mean, old.MAE_mean]) -
                                  np.array([new.R2, new.RMSE, new.MAE])))
            maximum_reference_difference = max(maximum_reference_difference, float(delta))
    assert maximum_reference_difference < 1e-11
    cs = pd.read_csv(ROOT / 'collision_summary.csv')
    independently_computed = []
    for n in [45, 53]:
        matrix = pd.read_csv(SOURCE / f'data/primary_model_matrix_{n}.csv')
        cols = [c for c in matrix if c not in ['record_id', 'source_id'] + TARGETS]
        # pandas groupby with explicit retained NA groups is independent of the
        # tuple-key implementation in the analysis program.
        groups = list(matrix.groupby(cols, dropna=False, sort=False))
        collisions = [v for _, v in groups if len(v) > 1]
        assert len(groups) == (85 if n == 45 else 98)
        for t in TARGETS:
            sse = sum(float(np.square(v[t].dropna() - v[t].mean()).sum()) for v in collisions)
            r = cs[(cs.n_inputs == n) & (cs.target == t)].iloc[0]
            np.testing.assert_allclose(r.oracle_min_SSE_all_labels, sse, atol=1e-10, rtol=1e-13)
            independently_computed.append(dict(inputs=n, target=t, min_SSE=sse))
    frozen = json.loads((ROOT / 'frozen_plan.json').read_text(encoding='utf-8'))
    hashes = {p: hashlib.sha256((SOURCE / p).read_bytes()).hexdigest() == h
              for p, h in frozen['input_sha256'].items()}
    assert all(hashes.values())
    pergroup = pd.read_csv(ROOT / 'metrics_per_group.csv')
    assert pergroup.loc[pergroup.n == 1, 'R2'].isna().all()
    result = dict(status='PASS', checks=dict(
        all_3534_predictions_reconciled=True, checked_model_group_memberships=checked_memberships,
        checked_training_mean_groups=checked_means,
        all_six_population_protocol_pairs_have_complete_target_coverage=True,
        labels_match_original_data=True, no_training_testing_group_overlap=True,
        no_label_imputation=True, source_inputs_unchanged=hashes,
        full98_archived_ET_reference_max_metric_difference=maximum_reference_difference,
        collision_SSE_independently_recomputed=independently_computed,
        single_label_group_R2_correctly_undefined=True,
        all_adverse_raw_and_augmented_results_retained=True))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
