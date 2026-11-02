import numpy as np
from statsmodels.discrete.conditional_models import ConditionalLogit
from v3_within_issue import ExactConditional


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
