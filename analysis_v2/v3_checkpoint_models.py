"""Frozen V3 checkpoint logistic and nonlinear prediction evaluations.

Save each scoring cell once; resume reads its saved result. New results are
exploratory, with the owner-approved AI adjudication deviations disclosed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis_v2')]
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from p8audit.l2_models import select_c,fit_selected,paired_bootstrap
from p8audit.prefix_features import STRUCTURAL_FIELDS

KS=(5,10,15,20,30)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(frame,k):
    work=frame.copy()
    work[f'k{k}_log_test_runs']=np.log1p(work[f'k{k}_n_test_runs'])
    structural=[f'k{k}_{name}' for name in STRUCTURAL_FIELDS]
    variables=[f'k{k}_{name}' for name in ('ran_tests','log_test_runs','test_fail','fail_then_pass')]
    return work,{'M0':structural,'M1':structural+[f'k{k}_test_fail'],'M2':structural+variables}


def nonlinear(train,valid,fit,fields):
    best=None;best_score=-np.inf;scores=[]
    for leaves in (15,31):
        for rate in (.05,.1):
            params=dict(max_leaf_nodes=leaves,learning_rate=rate,max_iter=100,early_stopping=True,random_state=20261103)
            model=HistGradientBoostingClassifier(**params).fit(train[fields].to_numpy(),train.y.to_numpy())
            score=float(average_precision_score(valid.y,model.predict_proba(valid[fields].to_numpy())[:,1]))
            scores.append(dict(**params,validation_ap=score))
            if score>best_score+1e-15:best,best_score=params,score
    model=HistGradientBoostingClassifier(**best).fit(fit[fields].to_numpy(),fit.y.to_numpy())
    return model,dict(selected=best,validation_grid=scores)


def score_cell(frame,k,population,kind,output):
    folder=output/f'{kind}_{population}_k{k}'
    metrics_path=folder/'metrics.json'
    if metrics_path.exists():
        result=json.loads(metrics_path.read_text(encoding='utf8'))
        if sha(folder/'predictions.parquet')!=result['prediction_sha256']:raise ValueError('Saved predictions changed')
        return result
    if folder.exists():raise FileExistsError('Incomplete score cell requires explicit recovery; do not silently re-score '+str(folder))
    work,fields=prepare(frame,k)
    work=work[work[f'k{k}_reached'].eq(1)]
    if population=='fixed30':work=work[work.k30_reached.eq(1)]
    models=('M0','M1','M2') if kind=='logistic' else ('M0','M2')
    train=work[work.split.eq('train')];valid=work[work.split.eq('validation')];test=work[work.split.eq('test')]
    fit=work[work.split.isin(['train','validation'])]
    if any(part.y.nunique()!=2 for part in (train,valid,test)):raise ValueError('A scoring cell lacks both classes')
    folder.mkdir(parents=True)
    predictions=test[['trajectory_id','issue_id','repo','scaffold','model','y']].copy()
    results={};coefficients=[]
    for name in models:
        columns=fields[name]
        if not np.isfinite(work[columns].to_numpy(dtype=float)).all():raise ValueError('Missing/nonfinite predictors')
        if kind=='logistic':
            chosen,grid=select_c(train[columns].to_numpy(dtype=float),train.y.to_numpy(),valid[columns].to_numpy(dtype=float),valid.y.to_numpy())
            model=fit_selected(fit[columns].to_numpy(dtype=float),fit.y.to_numpy(),chosen)
            selection=dict(selected_c=chosen,validation_ap_grid=grid)
            if name=='M2':
                coefficients.extend(dict(feature=f,standardized_coefficient=float(c)) for f,c in zip(columns,model.named_steps['model'].coef_[0]))
        else:model,selection=nonlinear(train,valid,fit,columns)
        prob=model.predict_proba(test[columns].to_numpy(dtype=float))[:,1]
        predictions[name]=prob
        results[name]=dict(ap=float(average_precision_score(test.y,prob)),auroc=float(roc_auc_score(test.y,prob)),
                           brier=float(brier_score_loss(test.y,prob)),selection=selection,predictor_fields=columns)
        print(json.dumps(dict(stage='model_scored',kind=kind,population=population,k=k,model=name,test_rows=len(test))),flush=True)
    # Written before any resampling. Resampling never feeds model/parameter choice.
    predictions.to_parquet(folder/'predictions.parquet',index=False)
    bootstrap=[]
    for name in models:
        if name=='M0':continue
        result=paired_bootstrap(predictions.y.to_numpy(),predictions.M0.to_numpy(),predictions[name].to_numpy(),predictions.repo.astype(str).tolist(),seed=20261092,n_replicates=1000)
        low,high=result.percentile_interval()
        results[name]['delta_ap']=results[name]['ap']-results['M0']['ap']
        results[name]['delta_ap_ci95']=[low,high]
        results[name]['bootstrap_valid']=result.n_valid
        results[name]['bootstrap_skipped']=result.skipped_replicates
        for index,delta in enumerate(result.deltas):bootstrap.append(dict(model=name,valid_replicate_index=index,delta_ap=delta))
    pd.DataFrame(bootstrap).to_csv(folder/'paired_bootstrap.csv',index=False)
    if coefficients:pd.DataFrame(coefficients).to_csv(folder/'M2_coefficients.csv',index=False)
    metric=dict(kind=kind,population=population,k=k,n_train=len(train),n_validation=len(valid),n_test=len(test),test_positive_prevalence=float(test.y.mean()),
                models=results,prediction_sha256=sha(folder/'predictions.parquet'),test_prediction_vectors_generated_once=True)
    metrics_path.write_text(json.dumps(metric,indent=2,allow_nan=False),encoding='utf8')
    return metric


def summarize(results,output,primary_k):
    rows=[]
    for result in results:
        for name,m in result['models'].items():
            ci=m.get('delta_ap_ci95',[None,None])
            rows.append(dict(kind=result['kind'],population=result['population'],k=result['k'],model=name,n_train=result['n_train'],n_validation=result['n_validation'],
                             n_test=result['n_test'],random_ap_baseline=result['test_positive_prevalence'],ap=m['ap'],auroc=m['auroc'],brier=m['brier'],delta_ap=m.get('delta_ap'),
                             delta_ap_low=ci[0],delta_ap_high=ci[1],primary_exploratory_endpoint=result['kind']=='logistic' and result['population']=='main' and result['k']==primary_k and name=='M2'))
    table=pd.DataFrame(rows);table.to_csv(output/'checkpoint_metrics.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(8,7),layout='constrained')
    for population,style in [('main','-'),('fixed30','--')]:
        for name,color in [('M1','#b6634e'),('M2','#287c8e')]:
            sub=table[(table.kind=='logistic')&(table.population==population)&(table.model==name)].sort_values('k')
            axes[0].plot(sub.k,sub.delta_ap,style,marker='o',color=color,label=f'{name}-M0, {population}')
            axes[0].fill_between(sub.k,sub.delta_ap_low,sub.delta_ap_high,color=color,alpha=.10)
    axes[0].axhline(0,color='grey',linewidth=.8);axes[0].set(xlabel='Completed-round checkpoint',ylabel='Paired delta AP',title='Prespecified exploratory checkpoint comparison')
    axes[0].legend(fontsize=8,frameon=False)
    coverage=pd.read_csv(output/'coverage.csv')
    for split,part in coverage.groupby('split'):axes[1].plot(part.k,part.coverage,marker='o',label=split)
    axes[1].set(xlabel='Completed-round checkpoint',ylabel='Known-label non-audit coverage',ylim=(0,1.03));axes[1].legend(frameon=False)
    fig.savefig(output/'fig_checkpoints.png',dpi=180);plt.close(fig)
    primary=table[table.primary_exploratory_endpoint].iloc[0].to_dict()
    status='increment not established' if primary['delta_ap_low']<=0<=primary['delta_ap_high'] else 'interval excludes zero; exploratory result with measurement limitations'
    (output/'PRIMARY_EXPLORATORY_RESULT.json').write_text(json.dumps(dict(status=status,primary=primary,all_new_results_exploratory=True),indent=2),encoding='utf8')
    return primary


def run(features_path,bindings_path,exclusions_path,output):
    bindings=json.loads(Path(bindings_path).read_text(encoding='utf8'))
    if sha(exclusions_path)!=bindings['exclusion_set_sha256']:raise ValueError('Exclusion set changed')
    for path,digest in bindings['code_hashes'].items():
        if sha(ROOT/path)!=digest:raise ValueError('Frozen dependency changed: '+path)
    frame=pd.read_parquet(features_path)
    if len(frame)!=207489 or frame.trajectory_id.duplicated().any():raise ValueError('Wrong feature table')
    ids=set(Path(exclusions_path).read_text(encoding='utf8').splitlines())
    work=frame[frame.resolved.isin([0,1])&~frame.trajectory_id.isin(ids)].copy();work['y']=work.resolved.eq(0).astype(int)
    repos={split:set(part.repo) for split,part in work.groupby('split')}
    if any(repos[a]&repos[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):raise ValueError('Repository leakage')
    coverage=[]
    for split,part in work.groupby('split'):
        for k in KS:coverage.append(dict(split=split,k=k,n_total=len(part),n_reached=int(part[f'k{k}_reached'].sum()),coverage=float(part[f'k{k}_reached'].mean())))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    context=dict(code_sha256=sha(Path(__file__)),features_sha256=sha(features_path),bindings_sha256=sha(bindings_path),exclusion_ids_sha256=sha(exclusions_path))
    start=output/'MODEL_RUN_CONTEXT.json'
    if start.exists():
        if json.loads(start.read_text(encoding='utf8'))!=context:raise ValueError('Run provenance changed; do not re-score saved cells')
    else:start.write_text(json.dumps(context,indent=2),encoding='utf8')
    pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False)
    results=[]
    for population in ('main','fixed30'):
        for k in KS:results.append(score_cell(work,k,population,'logistic',output))
    for k in sorted({5,int(bindings['primary_k'])}):results.append(score_cell(work,k,'main','hist_gradient_boosting',output))
    primary=summarize(results,output,int(bindings['primary_k']))
    record=dict(status='COMPLETE',cells=len(results),primary=primary,new_v3_test_set_scored=True,all_new_results_exploratory=True,context=context)
    (output/'MODEL_RUN_MANIFEST.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(json.dumps(record),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--bindings',type=Path,required=True)
    p.add_argument('--exclusions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.features,a.bindings,a.exclusions,a.output)
