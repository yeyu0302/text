from pathlib import Path
import hashlib,json,time,subprocess,platform,importlib.metadata,os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold,KFold
from src.evaluation import metrics,prior_correct

ROOT=Path(__file__).resolve().parents[2]
PROJECT=ROOT/'research_0680'
RESULTS=Path(os.environ.get('RESEARCH0680_RESULTS',str(PROJECT/'results'))).resolve()
REPORTS=Path(os.environ.get('RESEARCH0680_REPORTS',str(PROJECT/'reports'))).resolve()

def dump(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')

def read_data():
    train,test,sample=[pd.read_csv(ROOT/n) for n in ['train.csv','test.csv','sample_submission.csv']]
    assert train.id.is_unique and test.id.is_unique and sample.id.is_unique
    assert set(train.label)=={1,2,3} and len(train)==15000 and len(test)==5100
    assert train.label.value_counts().sort_index().tolist()==[2142,4285,8573]
    assert test.id.equals(sample.id) and list(sample.columns)==['id','label']
    expected=PROJECT/'configs/expected_dataset_hashes.json'
    if expected.exists():
        for name,value in json.loads(expected.read_text()).items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==value,f'Dataset version mismatch: {name}'
    return train.drop(columns='label'),train.label.to_numpy()-1,test,sample

def outer_folds(y,seed):
    out=np.full(len(y),-1,int)
    for f,(_,v) in enumerate(StratifiedKFold(5,shuffle=True,random_state=seed).split(y,y)):out[v]=f
    return out

def cf_splits(n,seed=913,n_splits=3):
    # Label-independent assignment is essential to the own-label intervention test.
    return list(KFold(n_splits,shuffle=True,random_state=seed).split(np.arange(n)))

def full_metrics(y,p):
    ans=metrics(y,p);pred=p.argmax(1);cm=confusion_matrix(y,pred,labels=[0,1,2])
    ans.update(level2_to1=int(cm[1,0]),level2_to3=int(cm[1,2]),
               predicted_counts=np.bincount(pred,minlength=3).tolist(),confusion_matrix=cm.tolist())
    return ans

def bootstrap(y,a,b,draws=1000,seed=738):
    rng=np.random.default_rng(seed);d=(a.argmax(1)==y).astype(float)-(b.argmax(1)==y).astype(float)
    groups=[np.flatnonzero(y==k) for k in range(3)];values=[]
    # Ordinary paired row bootstrap, retaining class-specific recall denominators.
    for _ in range(draws):
        idx=rng.integers(0,len(y),len(y))
        values.append(np.mean([d[idx[y[idx]==k]].mean() for k in range(3)]))
    return dict(delta=float(np.mean([d[g].mean() for g in groups])),ci_low=float(np.quantile(values,.025)),ci_high=float(np.quantile(values,.975)),draws=draws)

def save_run(dest,ids,y,p,folds,test_ids=None,tp=None,manifest=None):
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-6)
    table=pd.DataFrame(dict(id=ids,label=y+1,p1=p[:,0],p2=p[:,1],p3=p[:,2],pred=p.argmax(1)+1,fold=folds))
    assert len(table)==len(set(ids))
    table.to_csv(dest/'oof_predictions.csv',index=False)
    fm=[dict(fold=int(f),**full_metrics(y[folds==f],p[folds==f])) for f in np.unique(folds)]
    pd.DataFrame(fm).to_csv(dest/'fold_metrics.csv',index=False)
    pd.DataFrame(full_metrics(y,p)['confusion_matrix'],index=['true1','true2','true3'],columns=['pred1','pred2','pred3']).to_csv(dest/'confusion_matrix.csv')
    original=pd.read_csv(ROOT/'train.csv').set_index('id').reindex(np.asarray(ids))
    subgroups=[]
    for col in ['roofing_kind','f_90','f_109','f_141','f_182']:
        for value,idx in pd.Series(np.arange(len(ids))).groupby(original[col].fillna('MISSING').astype(str).to_numpy()).groups.items():
            idx=np.asarray(list(idx));mask=y[idx]==1
            subgroups.append(dict(feature=col,value=str(value),n=len(idx),level2_n=int(mask.sum()),
                level2_to1=int(np.sum((y[idx]==1)&(p[idx].argmax(1)==0))),level2_to3=int(np.sum((y[idx]==1)&(p[idx].argmax(1)==2))),
                recall2=float(np.mean(p[idx][mask].argmax(1)==1)) if mask.any() else None))
    pd.DataFrame(subgroups).to_csv(dest/'subgroup_metrics.csv',index=False)
    dump(dest/'metrics.json',full_metrics(y,p))
    if tp is not None:
        assert np.isfinite(tp).all() and np.allclose(tp.sum(1),1,atol=1e-6) and np.min(tp)>=0 and np.max(tp)<=1
        pd.DataFrame(dict(id=test_ids,p1=tp[:,0],p2=tp[:,1],p3=tp[:,2])).to_csv(dest/'test_probabilities.csv',index=False)
    dump(dest/'experiment_manifest.json',manifest or {})

def manifest(extra=None):
    data={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'train.csv',ROOT/'test.csv',ROOT/'sample_submission.csv']}
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT/'src').glob('*.py')}
    commit=None;dirty=None
    if (ROOT/'.git').exists():
        commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        dirty=bool(subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip())
    return dict(git_commit=commit,working_tree_dirty=dirty,
                code_version_note='Git commit is execution base; per-file source hashes identify newly written, possibly uncommitted research code.',
                dataset_hashes=data,source_hashes=sources,python=platform.python_version(),
                packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','catboost']},
                **(extra or {}))
