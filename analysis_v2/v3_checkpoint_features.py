"""Multi-checkpoint features using frozen parser, structure and detector.

Only parse the prefix needed for K<=30, retaining full short trajectories.
No outcomes or reviewer labels enter extraction. Shards are resumable and hashed.
"""
from __future__ import annotations
import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "analysis_v2")]
import pandas as pd
import pyarrow.parquet as pq
from p8audit.trajectory_parser import coerce_trajectory, parse_trajectory
from p8audit.prefix_features import STRUCTURAL_FIELDS, _features_at_checkpoint
from p8v3.testcmd import detect_test_command, parse_test_output
from v3_4_lib import build_calls

CHECKPOINTS = (5, 10, 15, 20, 30)
_rounds = None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def prefix_parsed(raw, total):
    records, _ = coerce_trajectory(raw)
    if records is None:raise ValueError('Trajectory could not be coerced')
    limit=min(len(records),80)
    while True:
        parsed=parse_trajectory(records[:limit])
        if parsed.n_complete_cycles>=30 or limit==len(records):break
        limit=min(len(records),limit+80)
    if total>=30:
        end=parsed.prefix_message_count(30)
        if end is None:raise ValueError('Metadata says >=30, prefix has no boundary')
        records=records[:end]
        parsed=parse_trajectory(records)
    else:
        records=records[:limit]
    if parsed.status not in {'ok','no_cycles'} or parsed.n_complete_cycles != min(total,30):
        raise ValueError('Frozen prefix and historical round count disagree')
    return parsed,records


def features(raw, total, scaffold):
    parsed, records=prefix_parsed(raw,total)
    calls=build_calls(scaffold,parsed,records)
    detected=[]
    for order,call in enumerate(calls):
        hit=detect_test_command(call.get('command'))
        if hit is not None and hit.confidence=='high':
            result=parse_test_output(call.get('output'))
            detected.append((int(call['round']),order,hit.runner,result))
    row={}
    for k in CHECKPOINTS:
        structural=_features_at_checkpoint(parsed,k,config=scaffold)
        row.update({f'k{k}_{field}':structural[f'k{k}_{field}'] for field in STRUCTURAL_FIELDS})
        selected=[x for x in detected if x[0]<=k]
        failed_runners=set();then_pass=False
        for _,_,runner,result in selected:
            if result.test_passed_summary and not result.test_failed and runner in failed_runners:then_pass=True
            if result.test_failed:failed_runners.add(runner)
        row.update({f'k{k}_reached':int(total>=k),f'k{k}_ran_tests':int(bool(selected)),f'k{k}_n_test_runs':len(selected),
                    f'k{k}_test_fail':int(any(x[3].test_failed for x in selected)),f'k{k}_fail_then_pass':int(then_pass),
                    f'k{k}_first_test_round':selected[0][0] if selected else None,
                    f'k{k}_missing_commands':sum(c.get('tool') in {'bash','execute_bash'} and c.get('command') is None and int(c['round'])<=k for c in calls)})
    for name in ('ran_tests','n_test_runs','test_fail','fail_then_pass'):
        values=[row[f'k{k}_{name}'] for k in CHECKPOINTS]
        if values!=sorted(values):raise ValueError('Non-nested cumulative feature')
    return row


def worker(item,data_root,meta_path,output,code_sha):
    global _rounds
    if _rounds is None:
        m=pd.read_parquet(meta_path,columns=['trajectory_id','n_rounds_total'])
        _rounds=dict(zip(m.trajectory_id,m.n_rounds_total))
    relative=item['path'];path=Path(data_root)/relative
    if path.stat().st_size!=int(item['bytes']) or sha(path)!=item['sha256']:
        raise ValueError('Pinned shard mismatch: '+relative)
    target=Path(output)/(relative.replace('/','__')+'.parquet')
    record_path=target.with_suffix('.json')
    if target.exists() and record_path.exists():
        record=json.loads(record_path.read_text(encoding='utf8'))
        if record['code_sha256']==code_sha and record['source_sha256']==item['sha256'] and record['output_sha256']==sha(target):return record
        raise ValueError('Existing shard features have different provenance')
    scaffold='openhands' if 'openhands' in relative else 'sweagent'
    rows=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8,columns=['trajectory_id','trajectory']):
        for source in batch.to_pylist():
            tid=str(source['trajectory_id'])
            row=features(source['trajectory'],int(_rounds[tid]),scaffold)
            row['trajectory_id']=tid
            rows.append(row)
    frame=pd.DataFrame(rows)
    temp=target.with_suffix('.partial')
    frame.to_parquet(temp,index=False);os.replace(temp,target)
    record={'shard':relative,'rows':len(frame),'source_sha256':item['sha256'],'output_sha256':sha(target),'code_sha256':code_sha}
    record_path.write_text(json.dumps(record,indent=2),encoding='utf8')
    return record


def run(data_root,meta_path,manifest_path,output,workers):
    manifest=json.loads(Path(manifest_path).read_text(encoding='utf8'))
    if manifest['revision']!='c144d9d160dcd9f5ebcd9520d2d54111dda8b681' or len(manifest['files'])!=84:raise ValueError('Wrong pinned manifest')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    code_sha=sha(Path(__file__))
    start=time.time();records=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        futures=[executor.submit(worker,item,str(data_root),str(meta_path),str(output),code_sha) for item in manifest['files']]
        for future in concurrent.futures.as_completed(futures):
            record=future.result();records.append(record)
            print(json.dumps({'completed_shards':len(records),'total_shards':84,'elapsed_seconds':round(time.time()-start),'shard':record['shard'],'rows':record['rows']}),flush=True)
    frames=[pd.read_parquet(output/(item['path'].replace('/','__')+'.parquet')) for item in manifest['files']]
    frame=pd.concat(frames,ignore_index=True)
    meta=pd.read_parquet(meta_path)
    if len(frame)!=207489 or frame.trajectory_id.duplicated().any() or set(frame.trajectory_id)!=set(meta.trajectory_id):raise ValueError('Feature census identity mismatch')
    frame=meta.merge(frame,on='trajectory_id',validate='one_to_one')
    frame.to_parquet(output/'features_v3.parquet',index=False)
    record={'status':'COMPLETE','rows':len(frame),'checkpoints':CHECKPOINTS,'elapsed_seconds':time.time()-start,'code_sha256':code_sha,
            'features_sha256':sha(output/'features_v3.parquet'),'metadata_sha256':sha(meta_path),'manifest_sha256':sha(manifest_path),'resolved_used_in_extraction':False}
    (output/'FEATURE_RUN_MANIFEST.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--meta',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();run(a.data_root,a.meta,a.manifest,a.output,a.workers)
