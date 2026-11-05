"""Audit or recompute focused R2 analyses without writing inside the release."""
from pathlib import Path
import argparse
import json
import subprocess
import sys


def run(script, *args):
    subprocess.run([sys.executable, str(script), *map(str, args)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-dir', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--out', type=Path, required=True, help='A new output directory outside the release')
    p.add_argument('--audit-only', action='store_true', help='Audit frozen predictions and rebuild phase ledgers, without fitting learners')
    args = p.parse_args()
    package, out = args.package_dir.resolve(), args.out.resolve()
    assert out != package and package not in out.parents, 'Choose an output directory outside the immutable release'
    if out.exists():
        raise FileExistsError('Output already exists; results are never overwritten')
    out.mkdir(parents=True)
    src = package / 'revision_r2/src'
    run(src / 'verify_revision_package.py', '--package-dir', package)
    if args.audit_only:
        run(src / 'audit_review_sensitivity.py', '--package-dir', package,
            '--results-dir', package / 'revision_r2/results', '--report', out / 'audit_results.json')
    else:
        analysis = out / 'analysis'
        run(src / 'primary_review_sensitivity.py', '--package-dir', package, '--out', analysis, '--prepare')
        old = json.loads((package / 'revision_r2/results/frozen_plan.json').read_text(encoding='utf-8'))
        new = json.loads((analysis / 'frozen_plan.json').read_text(encoding='utf-8'))
        assert new == old, 'Rebuilt metadata-only plan differs from the frozen revision plan'
        run(src / 'primary_review_sensitivity.py', '--package-dir', package, '--out', analysis, '--run')
        run(src / 'audit_review_sensitivity.py', '--package-dir', package,
            '--results-dir', analysis, '--report', analysis / 'audit_results.json')
    run(src / 'build_phase_evidence.py', '--package-dir', package, '--out', out / 'phase')
    run(src / 'verify_revision_package.py', '--package-dir', package)
    print('R2 outputs written outside the immutable release:', out)
    print('Source phase observations are retained as reported; no full-text source is redistributed.')


if __name__ == '__main__':
    main()
