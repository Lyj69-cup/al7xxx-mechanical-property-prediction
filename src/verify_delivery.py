"""Read-only SHA256 verification of the delivered file manifest; no ML dependencies."""
from pathlib import Path
import hashlib, sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    bad = []
    rows = (ROOT / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
    for row in rows:
        digest, name = row.split('  ', 1)
        path = (ROOT / name).resolve()
        if ROOT not in path.parents:
            bad.append((name, 'path outside package'))
        elif not path.is_file():
            bad.append((name, 'missing'))
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            bad.append((name, 'hash mismatch'))
    for name, problem in bad:
        print(problem + ': ' + name)
    print(f'{len(rows)} files checked; {len(bad)} problems')
    if bad:
        sys.exit(1)

if __name__ == '__main__':
    main()
