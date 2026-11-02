"""Validate completed results and build a hash-verified local delivery archive."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import zipfile
import numpy as np
import pandas as pd
import fitz

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'analysis_v2/original_plan'
RUN=PLAN/'v3_full_analysis_20261103'
MEAS=PLAN/'owner_authorized_ai_adjudication_20261103'
PDF=ROOT/'output/pdf/P8_v3_exploratory_audit_20261103.pdf'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def build():
    assert read(RUN/'WORKFLOW_STATUS.json')['stage']=='COMPLETE'
    for name,digest in read(ROOT/'analysis_v2/V3_IMPLEMENTATION_FREEZE.json')['implementation_sources'].items():
        assert sha(ROOT/'analysis_v2'/name)==digest,name
    fm=read(PLAN/'v3_features_20261103/FEATURE_RUN_MANIFEST.json')
    mm=read(RUN/'checkpoints/MODEL_RUN_MANIFEST.json')
    wm=read(RUN/'within_issue/WITHIN_ISSUE_MANIFEST.json')
    assert fm['status']==mm['status']==wm['status']=='COMPLETE'
    assert fm['rows']==207489 and mm['cells']==12 and wm['analyses']==14
    assert mm['context']['all_207489_historical_k5_structural_rows_exactly_matched']
    features=PLAN/'v3_features_20261103/features_v3.parquet'
    assert sha(features)==fm['features_sha256']==mm['context']['features_sha256']==wm['context']['features_sha256']
    excluded=set((ROOT/'analysis_v2/audit/audit_ids_EXCLUDE.txt').read_text().splitlines())
    assert len(excluded)==1196
    predictions=list((RUN/'checkpoints').glob('*/predictions.parquet'))
    assert len(predictions)==12
    for path in predictions:
        meta=read(path.parent/'metrics.json');frame=pd.read_parquet(path)
        assert sha(path)==meta['prediction_sha256'] and not frame.trajectory_id.duplicated().any()
        assert not set(frame.trajectory_id)&excluded
        for model,m in meta['models'].items():
            assert np.isfinite(frame[model]).all() and frame[model].between(0,1).all()
            if model!='M0':assert m['bootstrap_valid']+len(m['bootstrap_skipped'])==1000
    lock=read(MEAS/'AI_FINAL_LABEL_LOCK.json')
    for name,digest in lock['files'].items():assert sha(MEAS/name)==digest,name
    cells=[read(p) for p in (RUN/'within_issue').glob('*_k*.json')]
    assert len(cells)==14
    for cell in cells:
        if cell['status']=='COMPLETE':
            assert cell['bootstrap_requested']==500
            assert cell['bootstrap_valid']+len(cell['bootstrap_failed'])==500
    with fitz.open(PDF) as doc:
        assert len(doc)==6
        assert all('SYNTHETIC' not in p.get_text() for p in doc)
        assert all('Within-issue' in doc[4].get_text() for _ in [0])
    qa=ROOT/'output/pdf/PDF_VISUAL_QA.json'
    assert read(qa)['pdf_sha256']==sha(PDF) and read(qa)['all_pages_visually_inspected'] is True
    destination=ROOT/'output/P8_v3_final_20261103';destination.mkdir(parents=True,exist_ok=True)
    primary=read(RUN/'checkpoints/PRIMARY_EXPLORATORY_RESULT.json')['primary']
    statuses={c['analysis_name']+'_k'+str(c['k']):dict(status=c['status'],valid=c.get('bootstrap_valid'),failed=len(c.get('bootstrap_failed',[]))) for c in cells}
    verification=dict(status='COMPLETE_OWNER_AUTHORIZED_DEVIATION_ROUTE',strict_original_unaided_human_gold_route_complete=False,
        new_results_exploratory=True,primary=primary,feature_rows=fm['rows'],audit_ids_excluded=len(excluded),prediction_cells=len(predictions),
        within_issue_statuses=statuses,tests_passed=333,tests_skipped=0,pdf_sha256=sha(PDF),pdf_pages=6,all_pages_visually_inspected=True)
    (destination/'FINAL_VERIFICATION.json').write_text(json.dumps(verification,indent=2),encoding='utf8')
    (destination/'START_HERE.md').write_text('''# P8 completed exploratory extension

This delivery completes the project-owner-authorized AI-assisted review and AI-adjudication route. It does not claim completion of the original unaided independent-human gold-standard route. The original historical result remains separate.

Read `output/pdf/P8_v3_exploratory_audit_20261103.pdf` first, then `analysis_v2/original_plan/v3_full_analysis_20261103/publication/SUMMARY.md` and `paper_v3.md`. Machine-readable CSV/JSON, LaTeX tables, saved predictions and bootstrap results accompany the report. `FINAL_VERIFICATION.json` lists unsupported or failed analyses explicitly.

Provenance: submitted AI-assisted returns were reported by the owner as individually human-reviewed by Yunjie Guo and Hanqing Yang; the owner accepted the timeline explanation, stated no mutual viewing, and selected disclosed AI candidate adjudication. No software verification of reviewer identities or exact review timestamps is claimed. AI sources and the 506 additional logical consistency changes are retained. No third independent human adjudicator is claimed.

All new results are exploratory because the hypothesis reuses earlier observations and adjudication deviated from the original human route. Associations are not causal effects. Measurement ambiguity bounds are not confidence intervals. Failed fits are not zero effects.

## Reproduction

Extract the archive into a new project directory. Create a Python 3.12 environment using `requirements_delivery.txt`. Source data are the pinned nvidia/Open-SWE-Traces revision c144d9d160dcd9f5ebcd9520d2d54111dda8b681; the source-shard manifest is included, while raw trajectory shards are not duplicated in this archive. The verified final feature matrix, metadata, audit exclusions and historical K5 equality-check matrix are included.

The frozen protocol is `analysis_v2/PROTOCOL_v3.md`; implementation hashes and dependencies are in `V3_IMPLEMENTATION_FREEZE.json` and `V3_PROTOCOL_BINDINGS.json`. `FILE_MANIFEST.json` hashes every packaged payload. Existing results should be inspected without rescoring. For an independent rerun, use a NEW output directory:

```
python analysis_v2/v3_finish_workflow.py --feature-dir analysis_v2/original_plan/v3_features_20261103 --measurement analysis_v2/original_plan/owner_authorized_ai_adjudication_20261103 --output independent_rerun
```

This will reproduce the saved exploratory plan, not provide an independent external replication. The historical protocol commits were made locally only; no remote upload or publication was performed.
''',encoding='utf8')
    req=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (destination/'requirements_delivery.txt').write_text(req,encoding='utf8')
    paths=set()
    def add(p):
        p=Path(p)
        if p.is_file():paths.add(p)
    def tree(p):
        for file in Path(p).rglob('*'):
            if '__pycache__' not in file.parts and file.suffix not in {'.pyc','.partial'}:add(file)
    for p in [PDF,qa,features,PLAN/'ORIGINAL_PLAN_STATUS.json',PLAN/'RUNNING_ANALYSIS_JOBS.json',PLAN/'FINAL_TEST_RECORD_20261103.json',PLAN/'v3_features_20261103/FEATURE_RUN_MANIFEST.json',ROOT/'README.md',ROOT/'pytest.ini',ROOT/'analysis_v2/meta.parquet',ROOT/'analysis_v2/audit/audit_ids_EXCLUDE.txt',ROOT/'artifacts/manifests/pinned_parquet_files.json',ROOT/'artifacts/l2_run_20261027_01/prefix_features.parquet']:
        assert p.exists(),p;add(p)
    for pattern in ['*.py','*.md','V3*.json']:
        for p in (ROOT/'analysis_v2').glob(pattern):add(p)
    for p in [ROOT/'src',ROOT/'scripts',ROOT/'tests',ROOT/'analysis_v2/tests',ROOT/'analysis_v2/p8v3',RUN,MEAS,PLAN/'accepted_first_round_measurement_20261103_v2',PLAN/'reproduction_20261103',PLAN/'frozen_eval_extras_20261103']:
        tree(p)
    intake=PLAN/'returned_reviews_20261103'
    for p in intake.iterdir():add(p)
    for p in destination.iterdir():
        if p.name not in {'FILE_MANIFEST.json'} and p.suffix!='.zip':add(p)
    manifest={str(p.relative_to(ROOT)).replace('\\','/'):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)}
    (destination/'FILE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    archive=ROOT/'output/P8_v3_final_20261103.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(paths):z.write(path,path.relative_to(ROOT).as_posix())
        for name in ['START_HERE.md','FINAL_VERIFICATION.json','requirements_delivery.txt','FILE_MANIFEST.json']:
            z.write(destination/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,entry in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==entry['sha256'],name
    (ROOT/'output/P8_v3_final_20261103.zip.sha256').write_text(sha(archive)+'  '+archive.name+'\n',encoding='ascii')
    status=read(PLAN/'ORIGINAL_PLAN_STATUS.json')
    status.update(status='COMPLETE: OWNER-AUTHORIZED AI DEVIATION ROUTE',original_plan_complete=False,owner_authorized_deviation_route_complete=True,
        new_v3_test_set_scored=True,final_verification=str((destination/'FINAL_VERIFICATION.json').relative_to(ROOT)),final_archive=str(archive.relative_to(ROOT)),tests_passed=333,tests_skipped=0)
    (PLAN/'ORIGINAL_PLAN_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf8')
    print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,files=len(paths),sha256=sha(archive),verification=verification),indent=2))


if __name__=='__main__':build()
