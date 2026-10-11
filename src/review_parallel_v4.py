"""Parallel execution of independent audited evaluations (same results as review_v4.py)."""
import argparse, hashlib, importlib.metadata, json, os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
# Keep every learner single-threaded; parallelism is between evaluations.
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[key]='1'
import review_v4 as r

def evaluate(args):
    d,kind,protocol,seed,repeats=args
    path=r.OUT/f'cache_{kind}_{protocol}_{seed}.json'
    if path.exists():return f'{kind}/{protocol}/{seed}: cached'
    z=r.run(d,kind,protocol,seed,repeats)
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(z,allow_nan=False));temporary.replace(path)
    return f'{kind}/{protocol}/{seed}: complete'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--seeds',default='42,1,2,3,4');ap.add_argument('--permutation-repeats',type=int,default=20)
    args=ap.parse_args();seeds=[int(x) for x in args.seeds.split(',')]
    if args.workers<1 or len(set(seeds))!=len(seeds) or args.permutation_repeats<2:raise ValueError('Invalid run settings')
    d=r.prepare()
    settings={'seeds':seeds,'permutation_repeats':args.permutation_repeats,
      'script_sha256':hashlib.sha256(Path(r.__file__).read_bytes()).hexdigest(),
      'dataset_sha256':hashlib.sha256((r.OUT/'dataset_reviewed.csv').read_bytes()).hexdigest(),
      'versions':{x:importlib.metadata.version(x) for x in ['numpy','pandas','scikit-learn','scipy','xgboost','lightgbm']}}
    sp=r.OUT/'analysis_settings.json'
    if sp.exists() and json.loads(sp.read_text())!=settings:raise ValueError('Cached settings differ; archive reviewed_v4 before recomputing')
    if not sp.exists() and list(r.OUT.glob('cache_*.json')):raise ValueError('Unversioned caches exist')
    sp.write_text(json.dumps(settings,indent=2))
    tasks=[]
    for kind in ['phys','raw']:
        for protocol in ['input_grouped','loso','random']:
            for seed in ([42] if protocol=='loso' else seeds):
                tasks.append((d,kind,protocol,seed,args.permutation_repeats if kind=='phys' and protocol=='input_grouped' and seed==42 else 0))
    print('Evaluations:',len(tasks),'workers:',args.workers,flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending=[pool.submit(evaluate,t) for t in tasks]
        for future in as_completed(pending):print(future.result(),flush=True)
    sys.argv=[sys.argv[0],'--seeds',args.seeds,'--permutation-repeats',str(args.permutation_repeats)]
    r.main()

if __name__=='__main__':main()
