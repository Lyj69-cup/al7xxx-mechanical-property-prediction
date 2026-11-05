"""Read-only verification of the v1.0.0 baseline and complete additive R2 snapshot."""
from pathlib import Path
import argparse
import hashlib
import json

BASE_MANIFEST_SHA256 = '000d056aebdf334f6583a8526d516beb81a37045d90df3800a01ac2805c9636d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(root, name):
    lines = (root / name).read_text(encoding='utf-8').splitlines()
    names, errors = [], []
    for line in lines:
        digest, relative = line.split('  ', 1)
        path = (root / relative).resolve()
        if root not in path.parents:
            errors.append([relative, 'outside package'])
        elif not path.is_file():
            errors.append([relative, 'missing'])
        elif sha(path) != digest:
            errors.append([relative, 'hash mismatch'])
        names.append(relative)
    if len(names) != len(set(names)):
        errors.append([name, 'duplicate manifest entries'])
    return names, errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-dir', type=Path, default=Path(__file__).resolve().parents[2])
    args = p.parse_args()
    root = args.package_dir.resolve()
    assert sha(root / 'SHA256SUMS.txt') == BASE_MANIFEST_SHA256, 'Original baseline manifest changed'
    old, old_errors = check(root, 'SHA256SUMS.txt')
    all_names, all_errors = check(root, 'R2_SHA256SUMS.txt')
    assert len(old) == 132
    actual = {x.relative_to(root).as_posix() for x in root.rglob('*') if x.is_file()
              and not any(k in x.relative_to(root).parts for k in ['.git', '__pycache__', '.pytest_cache'])}
    expected = set(all_names) | {'R2_SHA256SUMS.txt'}
    added_unmanifested = sorted(actual - expected)
    missing = sorted(expected - actual)
    result = {
        'status': 'PASS' if not old_errors and not all_errors and not added_unmanifested and not missing else 'FAIL',
        'baseline_files_checked': len(old), 'baseline_manifest_unchanged': True,
        'r2_manifest_files_checked': len(all_names), 'total_files_including_R2_manifest': len(actual),
        'baseline_errors': old_errors, 'r2_errors': all_errors,
        'unmanifested_files': added_unmanifested, 'missing_files': missing,
        'read_only': True, 'version': 'v1.1.0',
    }
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
