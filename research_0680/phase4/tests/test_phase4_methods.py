import numpy as np
import pytest
from scipy.optimize import check_grad
from research_0680.src.common import read_data
from research_0680.phase4 import run,pairwise,robust,discovery

@pytest.fixture
def small():
    x,y,_,_=read_data();idx=np.random.default_rng(93).permutation(len(x))
    return x.iloc[idx[:360]],y[idx[:360]],x.iloc[idx[360:380]]

@pytest.mark.parametrize('backend',[run,pairwise,robust,discovery])
def test_query_batch_isolation(backend,small,monkeypatch,tmp_path):
    if backend is discovery:monkeypatch.setattr(discovery,'OUT',tmp_path)
    x,y,q=small;a=backend.predictions(x,y,q.iloc[:8]);b=backend.predictions(x,y,q)
    for name in a:
        assert np.isfinite(a[name]).all() and np.allclose(a[name].sum(1),1)
        assert np.max(abs(a[name]-b[name][:8]))<1e-10

def test_all_expanded_te_features_exclude_own_target(small):
    x,y,q=small;_,_,a,_=run.features(x,y,q)
    changed=y.copy();changed[0]=(changed[0]+1)%3
    _,_,b,_=run.features(x,changed,q)
    assert np.array_equal(a.iloc[0].to_numpy(),b.iloc[0].to_numpy(),equal_nan=True)

@pytest.mark.parametrize('kind',['brier','capped_ce'])
def test_robust_objective_gradient(kind):
    rng=np.random.default_rng(12);x=rng.normal(size=(30,6));y=rng.integers(0,3,30)
    anchor=rng.normal(size=(6,3));flat=anchor.ravel()+rng.normal(0,.2,18);weights=np.ones(30)/30
    def f(v):return robust.objective(v,x,y,weights,anchor,.1,kind)[0]
    def g(v):return robust.objective(v,x,y,weights,anchor,.1,kind)[1]
    assert check_grad(f,g,flat)<1e-6
