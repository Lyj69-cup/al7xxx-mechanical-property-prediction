"""Validate the delivery or recompute in a separate directory without overwriting it."""
import argparse,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def run(script,*args,cwd=ROOT):
 subprocess.run([sys.executable,str(cwd/'src'/script),*map(str,args)],cwd=cwd,check=True)
def copy_snapshot(dest):
 shutil.copytree(ROOT,dest,ignore=shutil.ignore_patterns('.git','.venv','venv','__pycache__','.pytest_cache','figures'))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--all',action='store_true');parser.add_argument('--traceable-only',action='store_true');parser.add_argument('--out');parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
 if not args.all and not args.traceable_only:
  with tempfile.TemporaryDirectory(prefix='jmep-numerical-audit-') as temporary:
   snapshot=Path(temporary)/'snapshot'
   copy_snapshot(snapshot)
   run('validate_final.py',cwd=snapshot);run('validate_traceable.py',cwd=snapshot);run('validate_primary_state.py',cwd=snapshot)
  print('Frozen numerical ledgers validated in a temporary copy; distributed files unchanged.')
  return
 if not args.out:parser.error('Recomputation requires --out for a new, empty directory')
 dest=Path(args.out).resolve()
 if dest==ROOT or ROOT in dest.parents:parser.error('Choose a directory outside the delivered package')
 if dest.exists():parser.error('Output directory must not exist; existing work is never overwritten')
 copy_snapshot(dest)
 if args.traceable_only:
  run('traceable_validation.py',cwd=dest);run('primary_state_sensitivity.py',cwd=dest);run('validate_traceable.py',cwd=dest);run('validate_primary_state.py',cwd=dest);run('figures_traceable.py',cwd=dest)
  print('Bhat-free analysis saved in',dest);return
 for folder in ['results/reviewed_v4']:
  existing=dest/folder
  if existing.exists():raise ValueError('Unexpected baseline cache; use a clean delivered package')
 run('review_parallel_v4.py','--workers',args.workers,cwd=dest)
 # Make the fresh baseline available at the paths used by the V5 additions.
 fresh=dest/'results/reviewed_v4'
 for f in fresh.iterdir():
  if f.is_file() and not f.name.startswith('cache_'):shutil.copy2(f,dest/'results'/f.name)
 run('advance_v5.py','--workers',args.workers,cwd=dest)
 run('auxiliary_v5.py',cwd=dest)
 run('external_wang2026.py',cwd=dest)
 run('validate_final.py',cwd=dest)
 run('figures_v5.py',cwd=dest);run('figures_v6.py',cwd=dest)
 run('traceable_validation.py',cwd=dest);run('primary_state_sensitivity.py',cwd=dest);run('validate_traceable.py',cwd=dest);run('validate_primary_state.py',cwd=dest);run('figures_traceable.py',cwd=dest)
 print('Recomputed analysis saved in',dest)
 print('Compare recomputed CSVs with archived reports; platform differences may change XGBoost and the fixed average.')
if __name__=='__main__':main()
