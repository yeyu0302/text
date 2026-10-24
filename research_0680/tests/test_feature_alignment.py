import numpy as np
from research_0680.src.common import read_data,outer_folds
from research_0680.src.build_proxy_features import ProxyBuilder,trusted,CLEAN
from src.evaluation import prior_correct

def test_outer_ids_exactly_once_and_sklearn_folds():
    x,y,te,s=read_data();f=outer_folds(y,42)
    assert set(f)==set(range(5)) and len(f)==15000 and x.id.is_unique
    assert te.id.equals(s.id)

def test_prior_transform():
    p=prior_correct(np.array([[.2,.3,.5],[.5,.1,.4]]),np.array([1,2,4])/7)
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1) and p.min()>=0

def test_v5_keeps_trusted_measurements():
    x,y,_,_=read_data();x=x.iloc[:300].reset_index(drop=True)
    builder=ProxyBuilder('V5').fit(x);a,_=builder.transform(x)
    for col,target in CLEAN.items():
        good=trusted(x,col)
        np.testing.assert_array_equal(a.loc[good,target],x.loc[good,col])

def test_actual_outer_outputs_are_complete_and_aligned():
    import pandas as pd,json
    from research_0680.src.common import RESULTS
    x,y,_,_=read_data();count=0
    for file in RESULTS.rglob('complete.json'):
        metadata=json.loads(file.read_text())
        if 'seed' not in metadata or 'stage' in metadata:continue
        path=file.parent/'oof_predictions.csv'
        if not path.exists():continue
        p=pd.read_csv(path);assert len(p)==15000 and p.id.equals(x.id)
        np.testing.assert_array_equal(p.label,y+1)
        np.testing.assert_array_equal(p.fold,outer_folds(y,int(metadata['seed'])))
        assert np.isfinite(p[['p1','p2','p3']]).all().all() and np.allclose(p[['p1','p2','p3']].sum(1),1)
        for selected in file.parent.glob('selection_*.json'):
            z=json.loads(selected.read_text());assert not set(z['train_ids'])&set(z['val_ids'])
        count+=1
    assert count>=1

def test_original_training_and_previous_results_not_modified():
    from research_0680.src.common import ROOT
    import json,hashlib
    m=json.loads((ROOT/'results/reproducibility.json').read_text())
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value for name,value in m['files'].items())
