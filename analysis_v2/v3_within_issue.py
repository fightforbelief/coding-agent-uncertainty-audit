"""Exact conditional logistic likelihood, checked against statsmodels.

Vectorized elementary-symmetric-polynomial DP accelerates the prespecified
repository bootstrap. Repository multiplicities repeat whole likelihood terms,
not rows merged into an enlarged conditional group.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'analysis_v2')]
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from statsmodels.discrete.conditional_models import ConditionalLogit


class ExactConditional:
    def __init__(self,x,y,groups,repos):
        self.blocks=[];self.support=[];self.n_groups=0
        positions={}
        for i,g in enumerate(groups):positions.setdefault(g,[]).append(i)
        packed={}
        for indices in positions.values():
            xi,yi=x[indices],y[indices]
            if yi.min()==yi.max():continue
            unique=set(repos[indices].tolist())
            if len(unique)!=1:raise ValueError('A conditional group crosses repositories')
            m=int(yi.sum());n=len(yi)
            if m>n/2:xi,yi,m=-xi,1-yi,n-m
            packed.setdefault((n,m),[]).append((xi,yi@xi,int(repos[indices[0]])))
            self.n_groups+=1
        for (n,m),items in packed.items():
            xblock=np.stack([z[0] for z in items]);repo=np.array([z[2] for z in items])
            self.blocks.append((xblock,np.stack([z[1] for z in items]),repo,n,m))
            centered=xblock-xblock.mean(axis=1,keepdims=True)
            self.support.append((repo,np.einsum('gnp,gnq->gpq',centered,centered)))

    def value_gradient(self,beta,multiplicities=None):
        ll=0.;gradient=np.zeros(len(beta));mass=0.
        for x,observed,repo,n,m in self.blocks:
            weights=np.ones(len(repo)) if multiplicities is None else multiplicities[repo]
            if not weights.any():continue
            eta=x@beta;center=eta.max(axis=1);w=np.exp(np.clip(eta-center[:,None],-700,0))
            forward=np.zeros((len(x),n+1,m+1));forward[:,0,0]=1
            for i in range(1,n+1):
                forward[:,i,:]=forward[:,i-1,:]
                forward[:,i,1:]+=w[:,i-1,None]*forward[:,i-1,:-1]
            denom=forward[:,n,m]
            if np.any(denom<=0):return -np.inf,np.full(len(beta),np.nan),float(weights.sum())
            ll+=float(weights@(observed@beta-m*center-np.log(denom)))
            adjoint=np.zeros((len(x),m+1));adjoint[:,m]=1/denom
            expected=np.zeros_like(observed)
            for i in range(n,0,-1):
                inclusion=w[:,i-1]*np.sum(adjoint[:,1:]*forward[:,i-1,:-1],axis=1)
                expected+=inclusion[:,None]*x[:,i-1,:]
                previous=adjoint.copy();previous[:,:-1]+=adjoint[:,1:]*w[:,i-1,None];adjoint=previous
            gradient+=(weights[:,None]*(observed-expected)).sum(axis=0)
            mass+=float(weights.sum())
        return ll,gradient,mass


def optimize(x,y,repo,conditional=None,multiplicities=None,start=None):
    if conditional is None:
        weights=np.ones(len(y)) if multiplicities is None else multiplicities[repo]
        mass=weights.sum()
        if mass<=0 or np.linalg.matrix_rank(x.T@(weights[:,None]*x))<x.shape[1]:return None
        def fun(beta):
            eta=x@beta
            return float(weights@(np.logaddexp(0,eta)-y*eta)/mass),x.T@(weights*(expit(eta)-y))/mass
    else:
        support=np.zeros((x.shape[1],x.shape[1]))
        for group_repo,scatter in conditional.support:
            weights=np.ones(len(group_repo)) if multiplicities is None else multiplicities[group_repo]
            support+=np.einsum('g,gpq->pq',weights,scatter)
        if np.linalg.matrix_rank(support)<x.shape[1]:return None
        def fun(beta):
            ll,g,mass=conditional.value_gradient(beta,multiplicities)
            return -ll/max(mass,1),-g/max(mass,1)
    result=minimize(fun,np.zeros(x.shape[1]) if start is None else start,jac=True,method='BFGS',options={'maxiter':200,'gtol':1e-7})
    if not np.isfinite(result.x).all() or not np.isfinite(result.fun) or np.max(np.abs(fun(result.x)[1]))>1e-5:return None
    return result.x


def design(frame,k,configuration=True):
    x=pd.DataFrame(index=frame.index)
    x['ran_tests']=frame[f'k{k}_ran_tests'].astype(float)
    x['log_test_runs']=np.log1p(frame[f'k{k}_n_test_runs'])
    x['test_fail']=frame[f'k{k}_test_fail'].astype(float)
    x['fail_then_pass']=frame[f'k{k}_fail_then_pass'].astype(float)
    x['log_tool_calls']=np.log1p(frame[f'k{k}_prefix_tool_call_count'])
    x['distinct_tools']=frame[f'k{k}_prefix_distinct_tool_count']
    x['log_observation_chars']=np.log1p(frame[f'k{k}_prefix_tool_observation_chars'])
    if configuration:
        config=frame.scaffold.astype(str)+'|'+frame.model.astype(str)
        x=pd.concat([x,pd.get_dummies(config,prefix='config',drop_first=True,dtype=float)],axis=1)
    return x.astype(float)


def fit_analysis(frame,k,model_name,group_column,bootstrap=500):
    frame=frame.copy().reset_index(drop=True)
    all_repos=sorted(frame.repo.unique());repo_index={r:i for i,r in enumerate(all_repos)}
    repo=np.array([repo_index[r] for r in frame.repo])
    xdf=design(frame,k,configuration=model_name in {'A','B'})
    y=frame.resolved.eq(0).to_numpy(dtype=float)
    if group_column is not None:
        groups=frame[group_column].to_numpy()
        outcome=pd.Series(y).groupby(groups).transform('nunique').to_numpy()>1
        if group_column=='cell' and frame.loc[outcome,group_column].nunique()<500:
            return dict(status='NOT_RUN_INSUFFICIENT_SUPPORT',k=k,model=model_name,eligible_groups=int(frame.loc[outcome,group_column].nunique()))
        frame=frame.loc[outcome].reset_index(drop=True);xdf=xdf.loc[outcome].reset_index(drop=True)
        y=y[outcome];repo=repo[outcome];groups=groups[outcome]
        centered=xdf-xdf.groupby(groups).transform('mean')
        keep=(centered.pow(2).sum()>1e-10)
    else:
        groups=None;keep=xdf.var()>1e-12
    dropped=xdf.columns[~keep].tolist();xdf=xdf.loc[:,keep]
    fields=xdf.columns.tolist()
    if not fields:return dict(status='NOT_IDENTIFIED_NO_VARYING_PREDICTORS',k=k,model=model_name)
    scales=xdf.std(ddof=0).to_numpy();means=xdf.mean().to_numpy()
    x=(xdf.to_numpy()-means)/scales
    if group_column is None:
        x=np.column_stack([np.ones(len(x)),x]);fields=['intercept']+fields;scales=np.r_[1,scales]
        conditional=None
    else:conditional=ExactConditional(x,y,groups,repo)
    beta=optimize(x,y,repo,conditional)
    if beta is None:return dict(status='POINT_FIT_DID_NOT_CONVERGE',k=k,model=model_name,n_rows=len(y),dropped=dropped)
    verification=None
    if conditional is not None:
        reference=ConditionalLogit(y,x,groups=groups)
        ll,g,_=conditional.value_gradient(beta)
        ref_ll=reference.loglike(beta);ref_score=reference.score(beta)
        if not np.isfinite(ref_ll) or not np.isfinite(ref_score).all() or not np.isfinite(ll) or not np.isfinite(g).all():
            return dict(status='POINT_REFERENCE_NUMERIC_FAILURE',k=k,model=model_name,n_rows=len(y),dropped=dropped)
        verification=dict(loglike_abs_difference=float(abs(ref_ll-ll)),score_max_abs_difference=float(np.max(np.abs(ref_score-g))))
        if verification['loglike_abs_difference']>1e-6 or verification['score_max_abs_difference']>1e-5:raise ValueError('Exact likelihood differs from statsmodels')
    rng=np.random.default_rng(20261092);draws=[];failures=[]
    for i in range(bootstrap):
        multiplicities=np.bincount(rng.integers(0,len(all_repos),len(all_repos)),minlength=len(all_repos))
        fitted=optimize(x,y,repo,conditional,multiplicities,beta)
        if fitted is None:failures.append(i)
        else:draws.append(fitted/scales)
        if (i+1)%100==0:print(json.dumps(dict(stage='within_issue_bootstrap',k=k,model=model_name,completed=i+1,total=bootstrap,valid=len(draws))),flush=True)
    raw=beta/scales;values=np.asarray(draws)
    coefficients=[]
    for j,field in enumerate(fields):
        if field=='intercept':continue
        interval=np.percentile(values[:,j],[2.5,97.5]).tolist() if len(values) else [None,None]
        def safe_exp(value):return float(np.exp(value)) if value is not None and -700<value<700 else None
        coefficients.append(dict(feature=field,log_or=float(raw[j]),odds_ratio=safe_exp(raw[j]),log_or_ci95=interval,or_ci95=[safe_exp(z) for z in interval]))
    return dict(status='COMPLETE',k=k,model=model_name,n_rows=len(y),n_groups=int(len(set(groups))) if groups is not None else None,
                repositories_in_bootstrap_universe=len(all_repos),bootstrap_requested=bootstrap,bootstrap_valid=len(draws),bootstrap_failed=failures,
                dropped_no_within_group_variation=dropped,statsmodels_likelihood_verification=verification,coefficients=coefficients)


def fit_variant(part,k,name,group):
    result=fit_analysis(part,k,'B' if name.startswith('B_') else name,group)
    result['analysis_name']=name
    return result


def run(features,exclusions,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    data=pd.read_parquet(features);ids=set(Path(exclusions).read_text(encoding='utf8').splitlines())
    data=data[data.resolved.isin([0,1])&~data.trajectory_id.isin(ids)].copy()
    data['cell']=data.issue_id.astype(str)+'|'+data.scaffold.astype(str)+'|'+data.model.astype(str)
    context=dict(code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),features_sha256=hashlib.sha256(Path(features).read_bytes()).hexdigest(),
                 exclusion_sha256=hashlib.sha256(Path(exclusions).read_bytes()).hexdigest(),bootstrap_replicates=500)
    context_file=output/'WITHIN_ISSUE_CONTEXT.json'
    if context_file.exists():
        if json.loads(context_file.read_text(encoding='utf8'))!=context:raise ValueError('Run provenance changed')
    else:context_file.write_text(json.dumps(context,indent=2),encoding='utf8')
    runs=[]
    pending=[]
    support=[]
    for k in (10,15):
        current=data.copy()
        for group_name in ('issue_id','cell'):
            sizes=current.groupby(group_name).size();outcomes=current.groupby(group_name).resolved.nunique()
            support.append(dict(k=k,group=group_name,n_groups=len(sizes),n_repeated=int(sizes.ge(2).sum()),n_outcome_discordant=int(outcomes.ge(2).sum()),max_group_size=int(sizes.max())))
        variants=[('A',None,current),('B','issue_id',current),('C','cell',current)]
        for (scaffold,model),part in current.groupby(['scaffold','model']):variants.append((f'B_{scaffold}_{model}','issue_id',part))
        for name,group,part in variants:
            path=output/f'{name}_k{k}.json'
            if path.exists():result=json.loads(path.read_text(encoding='utf8'))
            else:
                pending.append((path,part,k,name,group))
                continue
            runs.append(result)
            print(json.dumps(dict(stage='within_issue_fit_complete',analysis=name,k=k,status=result['status'])),flush=True)
    # Each analysis has its own fixed RNG seed, so scheduling cannot change draws.
    # Workers never share estimators or output files; only the parent saves results.
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures={executor.submit(fit_variant,part,k,name,group):path for path,part,k,name,group in pending}
        for future in as_completed(futures):
            result=future.result();path=futures[future]
            path.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
            runs.append(result)
            print(json.dumps(dict(stage='within_issue_fit_complete',analysis=result['analysis_name'],k=result['k'],status=result['status'])),flush=True)
    runs.sort(key=lambda r:(r['k'],r['analysis_name']))
    pd.DataFrame(support).to_csv(output/'within_issue_support.csv',index=False)
    rows=[]
    for result in runs:
        for coefficient in result.get('coefficients',[]):
            rows.append(dict(analysis=result['analysis_name'],k=result['k'],feature=coefficient['feature'],odds_ratio=coefficient['odds_ratio'],
                             or_low=coefficient['or_ci95'][0],or_high=coefficient['or_ci95'][1],log_or=coefficient['log_or'],bootstrap_valid=result['bootstrap_valid']))
    pd.DataFrame(rows,columns=['analysis','k','feature','odds_ratio','or_low','or_high','log_or','bootstrap_valid']).to_csv(output/'within_issue_odds_ratios.csv',index=False)
    (output/'WITHIN_ISSUE_MANIFEST.json').write_text(json.dumps(dict(status='COMPLETE',analyses=len(runs),context=context,all_effects_observational=True),indent=2),encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--exclusions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.features,a.exclusions,a.output)
