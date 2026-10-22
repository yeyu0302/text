import itertools
import numpy as np
import pandas as pd
from src.evaluation import exact_assignment, folds_for
from src.features import encode, FeatureBuilder


def test_exact_objectives_match_exhaustive_search():
    rng=np.random.default_rng(27)
    for _ in range(5):
        p=rng.dirichlet([1,1,1],size=6)
        for objective in ['accuracy','log']:
            score=p if objective=='accuracy' else np.log(p)
            pred=exact_assignment(p,[2,2,2],objective)
            achieved=score[np.arange(6),pred].sum()
            optimum=max(score[np.arange(6),a].sum() for a in itertools.product(range(3),repeat=6)
                        if np.array_equal(np.bincount(a,minlength=3),[2,2,2]))
            assert abs(achieved-optimum)<1e-8


def test_assignment_handles_ties_and_invalid_counts():
    import pytest
    p=np.ones((9,3))/3
    assert np.array_equal(np.bincount(exact_assignment(p,[3,3,3])),[3,3,3])
    with pytest.raises(ValueError):
        exact_assignment(p,[2,3,3])


def test_unknown_groups_use_only_fit_prior():
    from src.features import TE_SPECS
    cols=set(x for row in TE_SPECS for x in row[:2] if x is not None)
    fit=pd.DataFrame({c:['a','b','c','d','e','f'] for c in cols})
    apply=pd.DataFrame({c:['unknown'] for c in cols})
    p=encode(fit,np.array([0,0,0,1,1,2]),apply)
    np.testing.assert_allclose(p[['f141_m3_te_c0','f141_m3_te_c1','f141_m3_te_c2']].iloc[0],[.5,1/3,1/6])


def test_outer_validation_label_noninterference():
    # Holdout labels cannot affect features: API accepts only fit labels.
    import src.features as module
    data=pd.read_csv('train.csv').groupby('label',group_keys=False).head(20).reset_index(drop=True)
    y=data.pop('label').to_numpy()-1
    folds=folds_for(y)
    fit=np.flatnonzero(folds!=0);val=np.flatnonzero(folds==0)
    builder=FeatureBuilder('full')
    a,b=builder.fit_transform(data.iloc[fit],y[fit],data.iloc[val])
    mutated=y.copy();mutated[val]=(mutated[val]+1)%3
    c,d=FeatureBuilder('full').fit_transform(data.iloc[fit],mutated[fit],data.iloc[val])
    pd.testing.assert_frame_equal(a,c);pd.testing.assert_frame_equal(b,d)
    # Instrument actual inner encoder: no row's label enters its own TE stats.
    original=module.encode
    calls=[]
    def spy(fit_keys,labels,apply_keys,*args,**kwargs):
        calls.append((len(fit_keys),len(apply_keys),len(labels)))
        return original(fit_keys,labels,apply_keys,*args,**kwargs)
    module.encode=spy
    try:
        FeatureBuilder('full').fit_transform(data.iloc[fit],y[fit],data.iloc[val])
    finally:
        module.encode=original
    assert len(calls)==5
    assert all(n+m==len(fit) and n==l for n,m,l in calls[:4])


def test_holdout_covariates_cannot_change_training_features():
    data=pd.read_csv('train.csv').groupby('label',group_keys=False).head(20).reset_index(drop=True)
    y=data.pop('label').to_numpy()-1
    folds=folds_for(y);fit=np.flatnonzero(folds!=0);val=np.flatnonzero(folds==0)
    a,b=FeatureBuilder('full').fit_transform(data.iloc[fit],y[fit],data.iloc[val])
    changed=data.iloc[val].copy()
    changed['roofing_kind']='NEVER_SEEN_CATEGORY'
    changed['structure_age']=999
    changed['f_141']=999999
    c,d=FeatureBuilder('full').fit_transform(data.iloc[fit],y[fit],changed)
    pd.testing.assert_frame_equal(a,c)
    assert list(b.columns)==list(d.columns)


def test_metrics_are_multiclass_and_track_all_recalls():
    from src.evaluation import metrics
    y=np.array([0,1,2,0,1,2])
    p=np.eye(3)[y]*.8+.2/3
    m=metrics(y,p)
    assert m['balanced_accuracy']==m['accuracy']==m['macro_f1']==1
    assert all(m[f'recall_{k}']==1 for k in [1,2,3])


def test_legacy_adapters_accept_no_validation_labels():
    from src.train import legacy_nested,custom_predict
    data=pd.read_csv('train.csv').groupby('label',group_keys=False).head(30).reset_index(drop=True)
    y=data.pop('label').to_numpy()-1
    folds=folds_for(y);fit=np.flatnonzero(folds!=0);val=np.flatnonzero(folds==0)
    a,b=legacy_nested(data.iloc[fit],y[fit],data.iloc[val],42)
    for kind in ['hist','spline']:
        p=custom_predict(kind,a,y[fit],b,np.bincount(y[fit])/len(fit),42)
        assert p.shape==(len(val),3)
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1)


def test_adapter_fails_if_original_source_contract_changes():
    import pytest
    from src.train import source_replace
    with pytest.raises(RuntimeError):
        source_replace('unrelated code','critical audited pattern','replacement')
