"""Focused retrospective R2 sensitivities on the frozen primary-source cohort.

Run with the study's pinned scientific dependencies. This self-contained program
reads frozen feature CSVs, never imports or alters the archived analysis, and
writes only --out. No source or outcome-dependent grouping is fitted.

First --prepare freezes metadata-only assignments. Then --run requires the same
assignments/hashes, runs fixed ET and training-mean comparisons, and writes exact
sample input-collision diagnostics. These are post-review sensitivities, not a
preregistered study or new independent experiments.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time
import warnings

DEFAULT_SOURCE = Path(__file__).resolve().parents[2]
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

TARGETS = ['YS_MPa', 'UTS_MPa', 'EL_pct']
KEYS = ['record_id', 'source_id']
DESIGNATIONS = {
    'BI2024_7A09': '7A09',
    'CHEN2017_7A04': '7A04',
    'LIU2020_SC': '7055_based',
    'TAI2022_NA': '7075',
    'ORLOWSKA2022_HE': '7075',
}
DESIGNATION_EVIDENCE = {
    'BI2024_7A09': 'Ref29 article title and source ledger explicitly identify 7A09.',
    'CHEN2017_7A04': 'Ref28 article title and source ledger explicitly identify 7A04.',
    'LIU2020_SC': 'Ref25 article title/conditions identify 7055 and 7055 with 0.25Sc. This is a 7055-based family, not certification of a registered designation for the Sc-modified alloy.',
    'TAI2022_NA': 'Ref26 article title and original conditions explicitly identify AA7075.',
    'ORLOWSKA2022_HE': 'Ref27 article title and Methods explicitly identify AA7075.',
}
PARAMS = dict(n_estimators=500, max_features=0.5, random_state=42, n_jobs=1)
INPUT_FILES = [
    'data/primary_traceable.csv', 'data/primary_model_matrix_45.csv',
    'data/primary_model_matrix_53.csv', 'data/primary_state_enriched.csv',
    'supplementary/primary_traceable_evidence.csv',
    'src/primary_state_sensitivity.py',
]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonwrite(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def route_of(row):
    """Use only reported route/temper/hold metadata; never property values."""
    temper = str(row.temper).lower().strip()
    raw = row.aging_stages
    holds = json.loads(raw) if isinstance(raw, str) and raw.strip() else []
    if holds:
        if not all(np.isfinite(t) and np.isfinite(h) and h > 0 for t, h in holds):
            raise ValueError(f'Invalid explicitly listed hold: {row.record_id}')
        if 'rra' in temper:
            return 'RRA', 'Explicit RRA designation and ordered three-hold schedule.'
        if len(holds) == 1:
            return 'one_stage_artificial', 'One explicitly reported positive artificial-aging hold.'
        if len(holds) == 2:
            return 'two_stage_artificial', 'Two explicitly reported positive artificial-aging holds.'
        return None, 'Other multistage route not merged into an assumed category.'
    if temper == 'naturally aged':
        return 'natural_only', 'Explicit natural-aging state with no positive artificial-aging hold.'
    if temper == 'as-quenched':
        return 'as_quenched', 'Explicit as-quenched condition; zero artificial-aging duration.'
    if temper == 'o':
        return 'annealed', 'Explicit O state; prior thermal schedule remains unknown.'
    # Existing Xu source evidence supplies one 121 C/24 h hold even though the
    # aging_stages cell is blank. Do not interpret blank as no aging.
    if row.source_id in ['XU2015', 'XU2016', 'XU2018']:
        if float(row.aging_temp_C) == 121.0 and float(row.aging_time_h) == 24.0:
            return 'one_stage_artificial', 'Explicit T6/T652 121 C/24 h hold in primary conditions/numerical fields; no extra aging stages inferred.'
        raise ValueError('Xu reported hold unexpectedly changed')
    return None, 'As-received history or post-solution aging unspecified; no physical absence inferred from missing data.'


def load(source):
    d = pd.read_csv(source / 'data/primary_traceable.csv').fillna({'aging_stages': '', 'special_process': ''})
    a = pd.read_csv(source / 'data/primary_model_matrix_45.csv')
    b = pd.read_csv(source / 'data/primary_model_matrix_53.csv')
    assert len(d) == len(a) == len(b) == 98
    assert d.record_id.is_unique and (d.record_id == a.record_id).all() and (d.record_id == b.record_id).all()
    c45 = [c for c in a if c not in KEYS + TARGETS]
    c53 = [c for c in b if c not in KEYS + TARGETS]
    assert len(c45) == 45 and len(c53) == 53 and c53[:45] == c45
    np.testing.assert_equal(a[c45].to_numpy(), b[c45].to_numpy())
    for t in TARGETS:
        np.testing.assert_equal(d[t].to_numpy(), a[t].to_numpy())
        np.testing.assert_equal(d[t].to_numpy(), b[t].to_numpy())
    return d, a[c45], b[c53]


def assignments(d):
    rows = []
    for _, r in d.iterrows():
        route, evidence = route_of(r)
        rows.append(dict(record_id=r.record_id, source_id=r.source_id,
                         study_group=r.study_group,
                         reported_designation_family=DESIGNATIONS.get(r.source_id),
                         designation_evidence=DESIGNATION_EVIDENCE.get(r.source_id, 'No authenticated registered alloy designation assigned; custom/unresolved alloy excluded from designation subset.'),
                         reported_aging_route=route, route_evidence=evidence,
                         source_locator=r.source_locator, source_url=r.source_url))
    return pd.DataFrame(rows)


def plan(source, d, ass):
    return dict(
        analysis_date='2026-11-05', status='metadata_rules_frozen_before_new_fits',
        retrospective=True, preregistered=False,
        input_sha256={str(p): digest(source / p) for p in INPUT_FILES},
        assignment_sha256=hashlib.sha256(ass.to_csv(index=False, lineterminator='\n').encode()).hexdigest(),
        primary_records=98, primary_publication_line_groups=int(d.study_group.nunique()),
        designation_counts={str(k): int(v) for k, v in ass.reported_designation_family.value_counts().items()},
        designation_excluded=int(ass.reported_designation_family.isna().sum()),
        aging_route_counts={str(k): int(v) for k, v in ass.reported_aging_route.value_counts().items()},
        aging_route_excluded=int(ass.reported_aging_route.isna().sum()),
        rules=dict(designation='Authentic reported title/condition family only; 7055-based includes explicitly Sc-modified variants. All 22 unassigned rows excluded from BOTH training and evaluation.',
                   route='Broad explicitly reported aging-route architecture; unknown/as-received histories excluded from BOTH training and evaluation. Not exact schedule equivalence, not certification of every earlier hold.',
                   source='Existing study_group; all three Xu articles linked as one research line.',
                   exclusion='BI2024_7A09 largest source (42 records) excluded from BOTH training and evaluation before rebuilding seven remaining source holdouts.'),
        analyses=[
            ['primary98', 'source_loso', 'study_group'],
            ['designation76', 'designation_loso', 'reported_designation_family'],
            ['designation76', 'source_loso', 'study_group'],
            ['aging_route94', 'aging_route_loso', 'reported_aging_route'],
            ['aging_route94', 'source_loso', 'study_group'],
            ['without_dominant56', 'source_loso', 'study_group'],
        ],
        models=['ET_raw32', 'ET_augmented45', 'training_mean'],
        ET_parameters=PARAMS, model_selection=False,
        preprocessing='Target-specific training-only median imputation, all-training-missing fields constant0; input and target training-only standardization. No missing-label imputation.',
        protocol_scope='Source-group holdouts keep study groups disjoint. Route/designation holdouts are distinct estimands and do not additionally enforce source disjointness. Matched source-holdout references use identical restricted row populations.',
        collision_scope='Exact equality of all numeric un-imputed columns including the missing-value pattern. Group mean oracle uses observed labels only for post hoc empirical SSE diagnostics. It is not a fitted predictor, population ceiling, or estimate of a causal omitted-variable effect.',
        uncertainty='One deterministic complete set of leave-group-out predictions per protocol. No repeated-seed SD, significance test or independent-study confidence interval is inferred. Per-group MAE and pooled/macro metrics are reported; small/constant groups have undefined R2.'
    )


def metrics(y, p):
    return dict(n=int(len(y)), R2=float(r2_score(y, p)) if len(y) > 1 and np.ptp(y) > 0 else None,
                RMSE=float(np.sqrt(mean_squared_error(y, p))),
                MAE=float(mean_absolute_error(y, p)), bias_pred_minus_observed=float(np.mean(p-y)))


def collision_tables(d, x45, x53, out):
    summaries, groups, members = [], [], []
    for name, x in [('augmented45', x45), ('reported_state53', x53)]:
        # Tuple equality, not a finite-width row hash; no accidental hash aliases.
        keys = [tuple(None if pd.isna(v) else float(v) for v in row) for row in x.to_numpy()]
        index = {}
        for i, key in enumerate(keys):
            index.setdefault(key, []).append(i)
        colliders = [ids for ids in index.values() if len(ids) > 1]
        for j, ids in enumerate(colliders, 1):
            group_id = f'{name}_{j:02d}'
            for i in ids:
                members.append(dict(representation=name, collision_group=group_id, record_id=d.at[i, 'record_id'], source_id=d.at[i, 'source_id']))
            for t in TARGETS:
                vals = d.loc[ids, t].dropna().to_numpy()
                row = dict(representation=name, collision_group=group_id, target=t,
                           n_records=len(ids), n_labels=len(vals),
                           record_ids=json.dumps(d.loc[ids, 'record_id'].tolist()),
                           sources=json.dumps(sorted(d.loc[ids, 'source_id'].unique().tolist())),
                           label_span=float(np.ptp(vals)) if len(vals) else None,
                           oracle_group_mean=float(vals.mean()) if len(vals) else None,
                           oracle_group_min_SSE=float(np.square(vals-vals.mean()).sum()) if len(vals) else 0.0,
                           oracle_group_min_RMSE=float(np.sqrt(np.square(vals-vals.mean()).mean())) if len(vals) else None)
                groups.append(row)
        for t in TARGETS:
            n_total = int(d[t].notna().sum())
            tg = [g for g in groups if g['representation'] == name and g['target'] == t]
            sse = sum(g['oracle_group_min_SSE'] for g in tg)
            n_collision_labels = sum(g['n_labels'] for g in tg)
            summaries.append(dict(representation=name, target=t, n_inputs=x.shape[1],
                                  unique_unimputed_vectors=len(index), n_collision_groups=len(colliders),
                                  n_records_in_collision_groups=sum(map(len, colliders)), n_total_labels=n_total,
                                  n_labels_in_collision_groups=n_collision_labels,
                                  n_collision_groups_with_2plus_labels=sum(g['n_labels'] >= 2 for g in tg),
                                  n_collision_groups_with_nonzero_label_span=sum((g['label_span'] or 0) > 0 for g in tg),
                                  oracle_min_SSE_all_labels=sse,
                                  oracle_min_RMSE_all_labels=math.sqrt(sse/n_total),
                                  oracle_min_RMSE_collision_labels=math.sqrt(sse/n_collision_labels) if n_collision_labels else None,
                                  maximum_collision_label_span=max((g['label_span'] or 0 for g in tg), default=0.0)))
    pd.DataFrame(summaries).to_csv(out / 'collision_summary.csv', index=False)
    pd.DataFrame(groups).to_csv(out / 'collision_groups.csv', index=False)
    pd.DataFrame(members).to_csv(out / 'collision_membership.csv', index=False)
    pd.DataFrame([dict(representation=name, field=c, missing=int(x[c].isna().sum()), n=len(x))
                  for name, x in [('augmented45', x45), ('reported_state53', x53)] for c in x]).to_csv(out / 'input_missingness.csv', index=False)


def fit_one(x, y_train, train, test):
    a, b = x.loc[train].to_numpy(float), x.loc[test].to_numpy(float)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(a, axis=0)
    empty = ~np.isfinite(med)
    med[empty] = 0.0
    a, b = np.where(np.isfinite(a), a, med), np.where(np.isfinite(b), b, med)
    sx = StandardScaler().fit(a)
    sy = StandardScaler().fit(y_train.reshape(-1, 1))
    model = ExtraTreesRegressor(**PARAMS)
    model.fit(sx.transform(a), sy.transform(y_train.reshape(-1, 1)).ravel())
    p = sy.inverse_transform(model.predict(sx.transform(b)).reshape(-1, 1)).ravel()
    assert np.isfinite(p).all()
    return p, x.columns[empty].tolist()


def run(d, x45, out, frozen_plan):
    start = time.monotonic()
    a = pd.read_csv(out / 'metadata_group_assignments.csv')
    joined = d.merge(a[['record_id', 'reported_designation_family', 'reported_aging_route']], on='record_id', validate='one_to_one')
    assert (joined.record_id == d.record_id).all()
    populations = {
        'primary98': d.index.to_numpy(),
        'designation76': d.index[joined.reported_designation_family.notna()].to_numpy(),
        'aging_route94': d.index[joined.reported_aging_route.notna()].to_numpy(),
        'without_dominant56': d.index[d.source_id != 'BI2024_7A09'].to_numpy(),
    }
    assert {k: len(v) for k, v in populations.items()} == dict(primary98=98, designation76=76, aging_route94=94, without_dominant56=56)
    predictions, folds, skipped = [], [], []
    for cohort, protocol, column in frozen_plan['analyses']:
        ids = populations[cohort]
        for label in sorted(joined.loc[ids, column].unique()):
            te = ids[joined.loc[ids, column].to_numpy() == label]
            tr = ids[joined.loc[ids, column].to_numpy() != label]
            assert len(set(tr) & set(te)) == 0
            assert set(joined.loc[tr, column]).isdisjoint(joined.loc[te, column])
            for target in TARGETS:
                train = tr[d.loc[tr, target].notna()]
                test = te[d.loc[te, target].notna()]
                meta = dict(cohort=cohort, protocol=protocol, heldout_group=label, target=target,
                            n_training_rows=len(tr), n_test_rows=len(te),
                            n_training_labels=len(train), n_test_labels=len(test),
                            training_ids=json.dumps(d.loc[train, 'record_id'].tolist()),
                            testing_ids=json.dumps(d.loc[test, 'record_id'].tolist()),
                            training_source_groups=json.dumps(sorted(d.loc[train, 'study_group'].unique().tolist())),
                            testing_source_groups=json.dumps(sorted(d.loc[test, 'study_group'].unique().tolist())))
                if not len(test):
                    skipped.append(dict(**meta, reason='No observed target label in held-out group; labels not imputed.'))
                    continue
                assert len(train) > 1
                yt = d.loc[train, target].to_numpy()
                pred_sets = {'training_mean': (np.repeat(yt.mean(), len(test)), [])}
                for model, x in [('ET_raw32', x45.iloc[:, :32]), ('ET_augmented45', x45)]:
                    pred_sets[model] = fit_one(x, yt, train, test)
                for model, (p, missing) in pred_sets.items():
                    folds.append(dict(**meta, model=model, entirely_missing_training_fields=json.dumps(missing)))
                    for i, estimate in zip(test, p):
                        predictions.append(dict(cohort=cohort, protocol=protocol, heldout_group=label,
                                                target=target, model=model, record_id=d.at[i, 'record_id'],
                                                source_id=d.at[i, 'source_id'], study_group=d.at[i, 'study_group'],
                                                observed=float(d.at[i, target]), predicted=float(estimate)))
            print(cohort, protocol, label, f'{time.monotonic()-start:.1f}s', flush=True)
        pd.DataFrame(predictions).to_csv(out / 'predictions.csv', index=False)
        pd.DataFrame(folds).to_csv(out / 'fold_membership.csv', index=False)
        pd.DataFrame(skipped).to_csv(out / 'skipped_target_groups.csv', index=False)
    p = pd.DataFrame(predictions)
    summary, pergroup = [], []
    by = ['cohort', 'protocol', 'target', 'model']
    for key, g in p.groupby(by, sort=False):
        assert g.record_id.is_unique
        values = metrics(g.observed.to_numpy(), g.predicted.to_numpy())
        local = []
        for group, gg in g.groupby('heldout_group', sort=False):
            mg = metrics(gg.observed.to_numpy(), gg.predicted.to_numpy())
            local.append(mg)
            pergroup.append(dict(zip(by, key), heldout_group=group, **mg))
        summary.append(dict(zip(by, key), **values,
                            evaluated_groups=len(local),
                            macro_group_MAE=float(np.mean([k['MAE'] for k in local])),
                            macro_group_RMSE=float(np.mean([k['RMSE'] for k in local])),
                            minimum_group_n=min(k['n'] for k in local),
                            groups_with_defined_R2=sum(k['R2'] is not None for k in local)))
    pd.DataFrame(summary).to_csv(out / 'metrics_pooled.csv', index=False)
    pd.DataFrame(pergroup).to_csv(out / 'metrics_per_group.csv', index=False)
    return dict(seconds=time.monotonic()-start, predictions=len(p), fitted_target_group_representations=len(folds),
                skipped_target_groups=len(skipped),
                dependencies={n: importlib.metadata.version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'scipy']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', '--package-dir', dest='source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare == args.run:
        parser.error('Use exactly one of --prepare and --run')
    source, out = args.source.resolve(), args.out.resolve()
    assert source != out and source not in out.parents, 'Outputs must not modify the immutable source tree.'
    out.mkdir(parents=True, exist_ok=True)
    d, x45, x53 = load(source)
    a = assignments(d)
    current = plan(source, d, a)
    if args.prepare:
        if (out / 'frozen_plan.json').exists():
            raise FileExistsError('Frozen plan already exists; inspect instead of overwriting.')
        jsonwrite(out / 'frozen_plan.json', current)
        a.to_csv(out / 'metadata_group_assignments.csv', index=False, lineterminator='\n')
        print(json.dumps(current, indent=2, ensure_ascii=False), flush=True)
        return
    frozen = json.loads((out / 'frozen_plan.json').read_text(encoding='utf-8'))
    assert frozen == current, 'Data, metadata assignment, or frozen rule has changed.'
    assert digest(out / 'metadata_group_assignments.csv') == frozen['assignment_sha256']
    collision_tables(d, x45, x53, out)
    info = run(d, x45, out, frozen)
    info['input_sha256_recheck'] = {p: digest(source / p) == h for p, h in frozen['input_sha256'].items()}
    assert all(info['input_sha256_recheck'].values())
    info['status'] = 'COMPLETE'
    jsonwrite(out / 'execution_metadata.json', info)
    print(json.dumps(info, indent=2), flush=True)


if __name__ == '__main__':
    main()
