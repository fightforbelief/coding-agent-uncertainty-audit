"""Compute remaining frozen shard caches from the other end, at bounded CPU.

Calls the identical frozen worker into a separate output directory. Cache copies
are atomic; existing completed shard outputs must be byte-identical.
"""
import argparse
import concurrent.futures
import json
import os
import shutil
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis_v2')]
import v3_checkpoint_features as frozen


def run(data,meta,manifest,main,cache,workers):
    items=json.loads(Path(manifest).read_text(encoding='utf8'))['files']
    main,cache=Path(main),Path(cache);cache.mkdir(parents=True,exist_ok=True)
    code=frozen.sha(Path(frozen.__file__))
    # Reverse order complements the running ascending-order primary extractor.
    todo=[]
    for item in reversed(items):
        name=item['path'].replace('/','__')+'.parquet'
        if not (main/Path(name).with_suffix('.json')).exists():todo.append(item)
    completed=0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(frozen.worker,item,str(data),str(meta),str(cache),code):item for item in todo}
        for future in concurrent.futures.as_completed(futures):
            record=future.result();item=futures[future]
            name=item['path'].replace('/','__')+'.parquet'
            for source in (cache/name,(cache/name).with_suffix('.json')):
                target=main/source.name
                if target.exists():
                    if frozen.sha(target)!=frozen.sha(source):raise ValueError('Concurrent frozen caches disagree: '+source.name)
                else:
                    staged=target.with_suffix(target.suffix+'.cache_helper')
                    shutil.copy2(source,staged);os.replace(staged,target)
            completed+=1
            print(json.dumps(dict(stage='auxiliary_frozen_cache_complete',completed=completed,shard=item['path'])),flush=True)
    (cache/'AUXILIARY_CACHE_MANIFEST.json').write_text(json.dumps(dict(status='COMPLETE',frozen_worker_sha256=code,worker_count=workers,completed_shards=completed,
             main_cache=str(main.resolve()),helper_sha256=frozen.sha(Path(__file__))),indent=2),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','meta','manifest','main','cache'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.data,a.meta,a.manifest,a.main,a.cache,a.workers)
