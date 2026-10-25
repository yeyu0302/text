from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from research_0680.phase7.audit import load,probs,fit_calibration,calibrated,interval,main
from src.features import FeatureBuilder


def test_uploaded_identity_and_actual_id_range():
    x,y,test,a,b=load()
    assert [x.id.min(),x.id.max()]==[1,15000]
    assert [test.id.min(),test.id.max()]==[1,5100]
    assert len(y)==15000
    np.testing.assert_allclose(a['bagged'].sum(1),1)
    np.testing.assert_allclose(b['bagged'].sum(1),1)

@pytest.mark.parametrize('row',[[.2,.2,.2],[-.1,.4,.7],[np.nan,.4,.6],[np.inf,0,0]])
def test_corrupt_probabilities_fail(row):
    frame=pd.DataFrame([row],columns=['p_p1','p_p2','p_p3'])
    with pytest.raises(ValueError):probs(frame,'p')

def test_csv_rounding_only_is_normalized():
    frame=pd.DataFrame([[.333333,.333333,.333333]],columns=['p_p1','p_p2','p_p3'])
    np.testing.assert_allclose(probs(frame,'p'),np.ones((1,3))/3)


def test_balanced_calibration_targets_uniform_for_uninformative_predictions():
    y=np.repeat(np.arange(3),[30,60,120]);p=np.tile(np.array([1,2,4])/7,(len(y),1))
    theta=fit_calibration(p,y)
    np.testing.assert_allclose(calibrated(theta,p),np.ones_like(p)/3,atol=1e-5)


def test_query_features_cannot_change_fit_mapping():
    x,_,test,_,_=load();x=x.iloc[:300].copy();q=test.iloc[:20].copy()
    builder=FeatureBuilder('proxy',native=True).fit_base(x)
    before=builder.base(x)
    builder.base(q)
    q['f_141']=999999;q['structure_age']=999
    builder.base(q)
    pd.testing.assert_frame_equal(before,builder.base(x))


def test_cluster_interval_captures_correlated_errors():
    groups=np.repeat(np.arange(20),30);y=np.tile(np.arange(3),200)
    pred=np.where(groups<10,y,(y+1)%3);p=np.eye(3)[pred]
    row=interval(y,p,draws=200);cluster=interval(y,p,groups=groups,draws=200)
    assert cluster['high']-cluster['low']>2*(row['high']-row['low'])


def test_existing_outputs_are_preserved(tmp_path,monkeypatch):
    import research_0680.phase7.audit as module
    marker=tmp_path/'marker';marker.write_text('preserve')
    monkeypatch.setattr(module,'OUT',tmp_path)
    with pytest.raises(SystemExit,match='fresh'):main()
    assert marker.read_text()=='preserve'


def test_transport_identity_without_exposure_shift():
    from research_0680.phase7.transport_risk import transport
    y=np.repeat(np.arange(3),100);small=np.r_[np.arange(100)<20,np.arange(100)<40,np.arange(100)<60]
    correct=(np.arange(300)%3!=0).astype(float)
    r=transport(y,correct,small,float(np.mean([small[y==k].mean() for k in range(3)])))
    assert abs(r['delta'])<1e-10
    assert abs(r['odds_shift'])<1e-10


def test_transport_matches_stated_target_exposure():
    from research_0680.phase7.transport_risk import transport
    y=np.repeat(np.arange(3),100);small=np.r_[np.arange(100)<20,np.arange(100)<40,np.arange(100)<60]
    correct=(np.arange(300)%3!=0).astype(float)
    r=transport(y,correct,small,.5)
    assert np.isclose(np.mean(r['assumed_target_small_by_class']),.5)
    assert 0<=r['transported_score']<=1
