"""Share byte-verified completed caches between identical frozen workers."""
import argparse
import json
import os
import shutil
import time
from pathlib import Path
from v3_checkpoint_features import sha


def transfer(source,destination):
    count=0
    for manifest in source.glob('data__*.json'):
        try:record=json.loads(manifest.read_text(encoding='utf8'))
        except (ValueError,OSError):continue
        parquet=manifest.with_suffix('.parquet');target=destination/parquet.name
        target_manifest=destination/manifest.name
        if target_manifest.exists():continue
        if not parquet.exists() or sha(parquet)!=record['output_sha256']:continue
        if target.exists():
            if sha(target)!=record['output_sha256']:raise ValueError('Frozen caches disagree: '+parquet.name)
        else:
            staged=target.with_suffix('.cache_sync');shutil.copy2(parquet,staged);os.replace(staged,target)
        staged=target_manifest.with_suffix('.cache_sync_json');shutil.copy2(manifest,staged);os.replace(staged,target_manifest)
        count+=1
    return count


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--main',type=Path,required=True);p.add_argument('--cache',type=Path,required=True)
    a=p.parse_args();deadline=time.time()+4*3600
    while time.time()<deadline:
        n=transfer(a.main,a.cache)+transfer(a.cache,a.main)
        if n:print(json.dumps(dict(stage='verified_cache_sync',new_records=n,main_completed=len(list(a.main.glob('data__*.json'))))),flush=True)
        if (a.main/'FEATURE_RUN_MANIFEST.json').exists() and (a.cache/'AUXILIARY_CACHE_MANIFEST.json').exists():break
        time.sleep(10)
