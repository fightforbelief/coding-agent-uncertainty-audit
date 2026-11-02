import numpy as np
from statsmodels.discrete.conditional_models import ConditionalLogit
from v3_within_issue import ExactConditional
from v3_within_issue import fit_analysis
import pandas as pd


def test_exact_conditional_likelihood_and_gradient_match_statsmodels():
    rng=np.random.default_rng(5)
    groups=np.repeat(np.arange(4),[2,3,4,5])
    x=rng.normal(size=(len(groups),3))
    y=np.array([0,1, 0,1,1, 0,0,1,1, 0,1,1,1,1],dtype=float)
    repos=np.repeat(np.arange(4),[2,3,4,5])
    model=ExactConditional(x,y,groups,repos)
    reference=ConditionalLogit(y,x,groups=groups)
    beta=np.array([.3,-.4,.1])
    value,gradient,mass=model.value_gradient(beta)
    np.testing.assert_allclose(value,reference.loglike(beta),atol=1e-10)
    np.testing.assert_allclose(gradient,reference.score(beta),atol=1e-9)
    assert mass==4


def test_repository_multiplicity_repeats_groups_without_merging_them():
    x=np.array([[0.,1.],[1.,0.],[.3,.1],[.2,.8]])
    y=np.array([0.,1.,0.,1.]);groups=np.array(['a','a','b','b']);repos=np.array([0,0,1,1])
    model=ExactConditional(x,y,groups,repos)
    beta=np.array([.2,.5])
    value,gradient,_=model.value_gradient(beta,np.array([2,1]))
    copied_x=np.vstack([x[:2],x[:2],x[2:]])
    copied_y=np.r_[y[:2],y[:2],y[2:]]
    reference=ConditionalLogit(copied_y,copied_x,groups=np.array(['a_copy1','a_copy1','a_copy2','a_copy2','b','b']))
    np.testing.assert_allclose(value,reference.loglike(beta),atol=1e-10)
    np.testing.assert_allclose(gradient,reference.score(beta),atol=1e-9)


def test_zero_weight_group_does_not_contribute():
    x=np.array([[0.],[1.],[2.],[3.]])
    model=ExactConditional(x,np.array([0.,1.,0.,1.]),np.array([0,0,1,1]),np.array([0,0,1,1]))
    value,gradient,mass=model.value_gradient(np.array([.4]),np.array([1,0]))
    reference=ConditionalLogit(np.array([0.,1.]),x[:2],groups=np.array([0,0]))
    np.testing.assert_allclose(value,reference.loglike(np.array([.4])))
    assert mass==1


def synthetic_frame():
    rng=np.random.default_rng(13);n=400
    y=np.concatenate([rng.permutation([0,0,1,1]) for _ in range(100)])
    count=rng.integers(0,5,n)
    return pd.DataFrame(dict(issue_id=np.repeat([f'issue-{i}' for i in range(100)],4),
        cell=np.repeat([f'cell-{i}' for i in range(100)],4),repo=np.repeat([f'repo-{i//10}' for i in range(100)],4),
        scaffold=np.tile(['openhands','openhands','sweagent','sweagent'],100),model=np.tile(['m1','m2','m1','m2'],100),resolved=1-y,
        k10_ran_tests=count>0,k10_n_test_runs=count,k10_test_fail=(count>0)&(rng.random(n)<.2),k10_fail_then_pass=(count>1)&(rng.random(n)<.1),
        k10_prefix_tool_call_count=np.full(n,10),k10_prefix_distinct_tool_count=rng.integers(1,5,n),k10_prefix_tool_observation_chars=rng.integers(100,3000,n)))


def test_complete_synthetic_fit_matches_reference_and_bootstraps():
    result=fit_analysis(synthetic_frame(),10,'B','issue_id',bootstrap=3)
    assert result['status']=='COMPLETE'
    assert result['bootstrap_requested']==3 and result['bootstrap_valid']==3
    assert result['statsmodels_likelihood_verification']['loglike_abs_difference']<1e-6
    assert 'log_tool_calls' in result['dropped_no_within_group_variation']


def test_conditional_cell_model_does_not_force_insufficient_support():
    result=fit_analysis(synthetic_frame(),10,'C','cell',bootstrap=3)
    assert result['status']=='NOT_RUN_INSUFFICIENT_SUPPORT'
    assert result['eligible_groups']==100
