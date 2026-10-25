import hashlib,json,ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from research_0680.phase8.fixes import SandwichFeatures,TE_PREFIXES,combine_plans
from research_0680.phase8.source_audit import provided,HERE
from research_0680.phase8.fold_base import FoldBase
from research_0680.src.common import read_data

@pytest.fixture
def toy():
    base=pd.DataFrame(dict(clean_age=[10,80]*3,clean_storey=[1,4]*3,clean_area=[300]*6,clean_height=[8,28]*3,f_182=[0]*6,f_214=[0]*6))
    p3=np.array([.8,.1,.7,.2,.6,.3]);te=pd.DataFrame({f'{g}_bal_p{k}':v for g in TE_PREFIXES for k,v in [(1,.95-p3),(2,np.full(6,.05)),(3,p3)]})
    return base,te


def test_uploaded_sources_identical_and_immutable():
    a=(HERE/'evidence/optimize_pipeline_4.py').read_bytes();b=(HERE/'evidence/optimize_pipeline_5.py').read_bytes()
    assert a==b
    assert hashlib.sha256(a).hexdigest()=='1f1dc188479fb9f6034c17bfa026fdcded514eea561f9bc7b6157a75e4ec9d0b'


def test_provided_schema_fails_but_fixed_api_accepts_producer_names(toy):
    base,te=toy;m=provided()
    with pytest.raises(KeyError,match='clean_storeys'):m.compute_clean_preserving_blup_and_sandwich_features(base,te.to_numpy(),list(te))
    out=SandwichFeatures().fit(base,te).transform(base,te)
    assert out.shape==(6,5)
    assert np.count_nonzero(out[:,0])>0 and np.count_nonzero(out[:,1])>0


def test_original_geographic_filter_silently_zeroes_features(toy):
    base,te=toy;base=base.assign(clean_storeys=base.clean_storey);m=provided()
    out=m.compute_clean_preserving_blup_and_sandwich_features(base,te.to_numpy(),list(te))
    np.testing.assert_array_equal(out.iloc[:,:2],0)


def test_fixed_transform_is_query_batch_invariant(toy):
    base,te=toy;model=SandwichFeatures().fit(base,te)
    np.testing.assert_allclose(model.transform(base.iloc[:1],te.iloc[:1]),model.transform(base,te)[:1])
    v=base.iloc[:1].copy();v['clean_age']=50
    np.testing.assert_allclose(model.transform(v,te.iloc[:1]),model.transform(pd.concat([v,base.iloc[1:]],ignore_index=True),te)[:1])


def test_missing_geo_te_fails_explicitly(toy):
    base,te=toy
    with pytest.raises(ValueError,match='geographic TE'):SandwichFeatures().fit(base,te.drop(columns=te.columns[0]))


def test_planB_changes_oof_and_test_together():
    a=np.array([[.8,.1,.1],[.1,.8,.1]]);b=a[:,::-1].copy()
    oa,ta=combine_plans(a,b,a,b,0);ob,tb=combine_plans(a,b,a,b,1)
    np.testing.assert_array_equal(oa,a);np.testing.assert_array_equal(ob,b)
    np.testing.assert_array_equal(ta,a);np.testing.assert_array_equal(tb,b)


def test_missing_planB_oof_cannot_be_silently_accepted():
    p=np.ones((3,3))/3
    with pytest.raises(ValueError,match='Plan B requires'):combine_plans(p,None,p,p)


def test_original_planB_oof_ignores_weight_with_quota_stubbed():
    m=provided();m.enforce_exact_cardinality=lambda p,counts:p.argmax(1)+1
    a=np.array([[.7,.2,.1],[.2,.7,.1]]);b=a[:,::-1].copy()
    o0,t0,_=m.run_v11_dual_plan_inference(a,a,b,a,a,b,w_planB=0)
    o1,t1,_=m.run_v11_dual_plan_inference(a,a,b,a,a,b,w_planB=1)
    np.testing.assert_allclose(o0,o1)
    assert not np.allclose(t0,t1)


def test_original_solver_ignores_failure_flag():
    m=provided();m.minimize=lambda *a,**k:SimpleNamespace(x=np.zeros(6),success=False,message='deliberate test failure')
    p=m.fit_predict_bohning_softmax(np.ones((3,2)),np.array([1,2,3]),np.ones((1,2)))
    np.testing.assert_allclose(p,np.ones((1,3))/3)


def test_original_main_only_loads_and_prints():
    tree=ast.parse((HERE/'evidence/optimize_pipeline_4.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    calls={ast.unparse(n.func) for n in ast.walk(main) if isinstance(n,ast.Call)}
    assert calls=={'print','os.path.dirname','os.path.abspath','os.path.join','os.path.exists','open','json.load','pd.read_csv','len'}


def test_fold_base_context_does_not_depend_on_query_batch():
    x,_,q,_=read_data();factory=FoldBase().fit(x.iloc[:1000]);allq=factory.transform(q.iloc[:30]);single=factory.transform(q.iloc[:1])
    pd.testing.assert_frame_equal(single,allq.iloc[:1].reset_index(drop=True))
    solo=factory.transform(q.iloc[:1].assign(f_141=999999))
    assert solo.f_141_log_count.iloc[0]==0


def test_encoder_excludes_heldout_labels_for_frozen_indices():
    x,y,q,_=read_data();m=provided();x=x.iloc[:1000];base=m.clean_and_engineer_base_features(x,q.iloc[:0]);fit=np.arange(800);val=np.arange(800,1000);labels=y[:1000]+1
    a=np.eye(3)[labels-1]
    before=m.fit_apply_te_and_res(base,a,labels,np.zeros((1000,1)),fit,val,include_seismic_res=False)
    changed=labels.copy();changed[val]=changed[val]%3+1
    after=m.fit_apply_te_and_res(base,np.eye(3)[changed-1],changed,np.zeros((1000,1)),fit,val,include_seismic_res=False)
    pd.testing.assert_frame_equal(before,after)
