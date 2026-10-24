import hashlib,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research_0680.phase6.safe_decision import decode
from research_0680.phase6.wiener_audit import SOURCES,fit_linear,predict_linear
from research_0680.phase6.submission_audit import group_probs
from research_0680.phase6.shift_audit import solve_prior

HERE=Path(__file__).resolve().parents[1]

def test_recovery_bytes_match_user_best():
    a=(HERE/'evidence/submission_3.csv').read_bytes()
    assert a==(HERE/'downloads/submission_recovery_known_best_0679.csv').read_bytes()
    assert hashlib.sha256(a).hexdigest()=='61d3d73ee03aeb8bacf938fbdd2fa9673e38a7b7016d3aeaddf22c6413ebdc1f'

def test_report_provenance_conflict():
    assert (HERE/'evidence/report_4.md').read_bytes()==(HERE/'evidence/report_5.md').read_bytes()==(HERE/'evidence/report_for_submission3_user.md').read_bytes()

def test_146_is_net_change_not_number_of_changed_rows():
    a=pd.read_csv(HERE/'evidence/submission_5.csv').label.to_numpy()
    b=pd.read_csv(HERE/'evidence/submission_6.csv').label.to_numpy()
    assert np.sum((a==2)&(b!=2))==151
    assert np.sum((a!=2)&(b==2))==5
    assert np.sum(a!=b)==160

def test_lmmse_query_main_value_never_used():
    rng=np.random.default_rng(81); source='structure_age';n=60
    physical=rng.uniform(10,90,n)
    x=pd.DataFrame({c:physical+rng.normal(0,10,n) for c in SOURCES[source]})
    x[source]=physical
    train=x.iloc[:40].copy();query=x.iloc[40:].copy()
    par=fit_linear(train,source);first=predict_linear(par,query,source)
    query[source]=999
    np.testing.assert_array_equal(first,predict_linear(par,query,source))
    query[source]=np.nan
    np.testing.assert_array_equal(first,predict_linear(par,query,source))

def test_local_soft_targets_ignore_validation_labels_global_does_not():
    key=pd.Series(['a','a','a','b']);y=np.array([0,1,2,1]);train=np.array([0,1,3])
    local=group_probs(key.iloc[train],y[train]);global_before=group_probs(key,y)
    y[2]=0
    pd.testing.assert_frame_equal(local,group_probs(key.iloc[train],y[train]))
    assert not np.allclose(global_before.loc['a'],group_probs(key,y).loc['a'])

def test_identical_x_soft_frequency_is_ce_identity():
    labels=np.array([0,1,1,2,2,2]);p=np.array([.2,.3,.5])
    counts=np.bincount(labels,minlength=3)
    assert np.isclose(-np.log(p[labels]).sum(),-len(labels)*(counts/len(labels)*np.log(p)).sum())

def test_known_count_accuracy_matches_exhaustive_optimum():
    e=json.loads((HERE/'results/transport_objective_counterexample.json').read_text())
    p=np.array(e['probabilities']);pred=decode(p,known_counts=[1,1,1])-1
    utility=sum(p[i,k] for i,k in enumerate(pred))
    assert np.isclose(utility,max(sum(p[i,k] for i,k in enumerate(a)) for a in itertools.permutations(range(3))))
    assert utility>e['expected_correct_log']
    np.testing.assert_array_equal(decode(p),p.argmax(1)+1)

@pytest.mark.parametrize('invalid',[np.array([1,2,3]),np.array([[.2,.2,.2]]),np.array([[np.nan,0,1]]),np.array([[-.2,.5,.7]])])
def test_decoder_rejects_invalid_inputs(invalid):
    with pytest.raises(ValueError):decode(invalid)

def test_bbse_recovers_prior_under_declared_model():
    c=np.array([[.8,.1,.1],[.1,.7,.2],[.1,.2,.7]]);prior=np.array([.2,.5,.3])
    estimate,_=solve_prior(c,c@prior)
    np.testing.assert_allclose(estimate,prior)
