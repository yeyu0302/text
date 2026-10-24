import numpy as np
from research_0680.src.common import read_data
from research_0680.phase2.src.methods import PosteriorProxy,PosteriorFactory,ConditionalHierarchy,physical_predict,fit_offset
from research_0680.src.train_baseline import BASE
from research_0680.src.build_proxy_features import trusted,CLEAN

def data():
    x,y,_,_=read_data();idx=np.concatenate([np.flatnonzero(y==k)[:35] for k in range(3)])
    return x.iloc[idx].reset_index(drop=True),y[idx]

def test_posterior_keeps_silver_valid_main_and_is_finite_with_missing_sensors():
    x,y=data();obj=PosteriorProxy().fit(x);a,u=obj.transform(x)
    for source,name in CLEAN.items():
        good=trusted(x,source);np.testing.assert_array_equal(a.loc[good,name],x.loc[good,source])
    q=x.iloc[:2].copy()
    for cols in __import__('research_0680.src.build_proxy_features',fromlist=['GROUPS']).GROUPS.values():q[cols]=np.nan
    a,u=obj.transform(q)
    assert np.isfinite(u.filter(like='_posterior_').to_numpy()).all()

def test_posterior_te_excludes_own_damage_label():
    x,y=data();changed=y.copy();changed[7]=(changed[7]+1)%3
    a,_=PosteriorFactory(BASE).fit_transform(x,y,x.iloc[:2]);b,_=PosteriorFactory(BASE).fit_transform(x,changed,x.iloc[:2])
    np.testing.assert_array_equal(a.iloc[7],b.iloc[7])

def test_hierarchy_unknown_groups_fall_back_to_physical_model():
    x,y=data();fit=np.flatnonzero(np.arange(len(x))%5!=0);val=np.flatnonzero(np.arange(len(x))%5==0)
    q=x.iloc[val].copy();q[['f_90','f_109','f_141']]=999999
    p=ConditionalHierarchy(8).fit_predict(x.iloc[fit],y[fit],q)
    raw=physical_predict(x.iloc[fit],y[fit],q)
    from src.evaluation import prior_correct
    expected=prior_correct(raw,np.bincount(y[fit],minlength=3)/len(fit))
    np.testing.assert_allclose(p,expected,atol=1e-8)
    assert np.allclose(p.sum(1),1)

def test_map_offsets_shrink_for_small_sample():
    base=np.tile([.2,.3,.5],(5,1));y=np.array([2]*5);parent=np.zeros(3)
    weak=fit_offset(base,y,parent,2);strong=fit_offset(base,y,parent,32)
    assert np.linalg.norm(strong)<np.linalg.norm(weak)
