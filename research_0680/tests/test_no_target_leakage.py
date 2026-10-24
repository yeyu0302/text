import numpy as np
import pandas as pd
from research_0680.src.common import read_data
from research_0680.src.leakage_safe_te import NestedTE
from research_0680.src.conditional_knn import ConditionalKNN
from research_0680.src.train_baseline import FeatureFactory,BASE
from src.features import keys,row_features

def sample():
    x,y,_,_=read_data();idx=np.concatenate([np.flatnonzero(y==k)[:30] for k in range(3)])
    return x.iloc[idx].reset_index(drop=True),y[idx]

def test_te_own_label_invariance():
    x,y=sample();k=keys(x,row_features(x));other=y.copy();other[5]=(other[5]+1)%3
    for mode in ['plain','balanced_log16']:
        a,_=NestedTE(mode).fit_transform(k,y,k.iloc[:3]);b,_=NestedTE(mode).fit_transform(k,other,k.iloc[:3])
        np.testing.assert_array_equal(a.iloc[5],b.iloc[5])

def test_knn_crossfit_isolation():
    x,y=sample();obj=ConditionalKNN(5);a,b=obj.fit_transform(x,y,x.iloc[:3])
    for split in obj.crossfit_audit:assert not set(split['fit_ids'])&set(split['query_ids'])
    for own,ids in zip(x.id.iloc[:3],obj.neighbor_ids):assert own not in ids

def test_knn_crossfit_own_label_invariance():
    x,y=sample();changed=y.copy();changed[5]=(changed[5]+1)%3
    a,_=ConditionalKNN(5).fit_transform(x,y,x.iloc[:3])
    b,_=ConditionalKNN(5).fit_transform(x,changed,x.iloc[:3])
    np.testing.assert_array_equal(a.iloc[5],b.iloc[5])

def test_validation_labels_and_covariates_do_not_affect_fit_features():
    x,y=sample();q=x.iloc[-12:].copy();a,b=FeatureFactory(BASE).fit_transform(x.iloc[:-12],y[:-12],q)
    q['f_141']=99999;q['structure_age']=999;q['roofing_kind']='UNKNOWN'
    c,d=FeatureFactory(BASE).fit_transform(x.iloc[:-12],y[:-12],q)
    pd.testing.assert_frame_equal(a,c);assert list(b)==list(d)
