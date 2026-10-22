"""Strict outer-fold calibration and ensemble comparisons.

Calibration models train only within each outer training set. Other outer folds'
OOF predictions are NEVER used to tune the current fold's weights/calibration:
those predictions would themselves depend on current validation labels.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.train import features, fit_predict
from src.evaluation import metrics, prior_correct, exact_assignment

ROOT=Path(__file__).resolve().parents[1]


def normalize(p):
    return p/p.sum(axis=1,keepdims=True)


def rank_average(ps):
    return normalize(np.mean([np.column_stack([rankdata(p[:,k])/len(p) for k in range(3)]) for p in ps],axis=0))


def blend(ps,weights):
    return np.einsum('m,mnk->nk',weights,np.stack(ps))


def weight_grid(n):
    import itertools
    return [np.asarray(w)/4 for w in itertools.product(range(5),repeat=n) if sum(w)==4]


def main(names,tag='F00'):
    configs=[json.loads((ROOT/'results/predictions'/n/'summary.json').read_text()) for n in names]
    arrays=[np.load(ROOT/'results/predictions'/n/'predictions.npz') for n in names]
    y,folds=arrays[0]['y'],arrays[0]['folds']
    assert all(np.array_equal(z['folds'],folds) for z in arrays)
    tr=pd.read_csv(ROOT/'train.csv').drop(columns='label');te=pd.read_csv(ROOT/'test.csv')
    dest=ROOT/'results'/tag;dest.mkdir(exist_ok=True)
    if (dest/'comparison.csv').exists() and (dest/'calibration.json').exists():
        previous=json.loads((dest/'calibration.json').read_text())
        if previous['models']!=names:
            raise ValueError('Completed calibration experiment has different models; choose a new tag')
        if (dest/'balanced_decision_summary.csv').exists():
            print('CACHED',tag,flush=True)
            return
    outputs={k:np.zeros((len(y),3)) for k in ['plain','prior','multipliers','weighted','rank','stack']}
    test_outputs={k:np.zeros((len(te),3)) for k in outputs}
    records=[];params=[];calibration_cache=[]
    t0=time.perf_counter()
    for f in range(5):
        fit=np.flatnonzero(folds!=f);val=np.flatnonzero(folds==f)
        inner,cal=train_test_split(fit,test_size=.25,stratify=y[fit],random_state=800+f)
        cp=[];vp=[];tp=[];rv=[];rt=[]
        for cfg,arr in zip(configs,arrays):
            cache=dest/f'cal_{f}_{cfg["name"]}.npz'
            if cache.exists():
                saved=np.load(cache)
                assert np.array_equal(saved['cal'],cal)
                q=saved['p']
            else:
                a,b=features(tr.iloc[inner],y[inner],tr.iloc[cal],cfg['family'],cfg['seed'],cfg['smoothing'],cfg['aggregate'],cfg['raw_mode'],cfg['native'])
                auxiliary_params=dict(cfg['params'])
                if cfg['model'] in ('xgb','lgb'):
                    auxiliary_params['n_jobs']=4
                if cfg['model']=='cat':
                    auxiliary_params['thread_count']=4
                model_seed=cfg['seed']+10*f if cfg['model']=='hist' else cfg['seed']+f
                p=fit_predict(cfg['model'],a,y[inner],b,auxiliary_params,model_seed)
                q=prior_correct(p,np.bincount(y[inner],minlength=3)/len(inner))
                np.savez_compressed(cache,p=q,cal=cal)
            cp.append(q);vp.append(arr['p'][val]);rv.append(arr['raw'][val])
            # Test predictions below are averages of outer models, not the
            # particular current fold. Validation decisions remain strictly
            # nested; test blends average the five calibration rules.
            tp.append(arr['test']);rt.append(arr['raw_test'])
        candidates=weight_grid(len(names))
        weights=max(candidates,key=lambda w:metrics(y[cal],blend(cp,w))['balanced_accuracy'])
        cmean=np.mean(cp,axis=0)
        mults=[np.array([1.,w1,w2]) for w1 in [.85,1.,1.15] for w2 in [.85,1.,1.15]]
        mult=max(mults,key=lambda m:metrics(y[cal],normalize(cmean*m))['balanced_accuracy'])
        stack=LogisticRegression(C=.01,max_iter=500,class_weight='balanced',random_state=42)
        stack.fit(np.hstack([np.log(np.clip(p,1e-8,1)) for p in cp]),y[cal])
        val_results=dict(plain=np.mean(rv,axis=0),prior=np.mean(vp,axis=0),
                         multipliers=normalize(np.mean(vp,axis=0)*mult),weighted=blend(vp,weights),
                         rank=rank_average(vp),stack=stack.predict_proba(np.hstack([np.log(np.clip(p,1e-8,1)) for p in vp])))
        test_results=dict(plain=np.mean(rt,axis=0),prior=np.mean(tp,axis=0),
                          multipliers=normalize(np.mean(tp,axis=0)*mult),weighted=blend(tp,weights),
                          rank=rank_average(tp),stack=stack.predict_proba(np.hstack([np.log(np.clip(p,1e-8,1)) for p in tp])))
        for method,p in val_results.items():
            outputs[method][val]=p
            test_outputs[method]+=test_results[method]/5
            records.append(dict(method=method,fold=f,**metrics(y[val],p)))
        params.append(dict(fold=f,calibration_rows=len(cal),fit_rows=len(inner),weights=weights.tolist(),multipliers=mult.tolist(),
                           stacking_C=.01,calibration_seed=800+f))
        print(tag,'fold',f,'weights',weights,'multipliers',mult,flush=True)
    np.savez_compressed(dest/'predictions.npz',**outputs,**{k+'_test':p for k,p in test_outputs.items()},y=y,folds=folds)
    pd.DataFrame(records).to_csv(dest/'fold_metrics.csv',index=False)
    pd.DataFrame([dict(method=k,**metrics(y,p)) for k,p in outputs.items()]).to_csv(dest/'comparison.csv',index=False)
    (dest/'calibration.json').write_text(json.dumps(dict(models=names,folds=params,runtime_seconds=time.perf_counter()-t0,
        cv_seed=configs[0]['seed'],code_hash=hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src').glob('*.py')))).hexdigest(),
        auxiliary_model_threads=4,
        test_blending='Average five independent outer calibration rules applied to fold-mean test probabilities'),indent=2))
    decision_comparison(outputs,y,folds,dest)
    print(pd.read_csv(dest/'comparison.csv').to_string(index=False),flush=True)


def decision_comparison(outputs,y,folds,dest):
    rows=[]
    for f in range(5):
        val=np.flatnonzero(folds==f)
        n=min(np.sum(y[val]==k) for k in range(3))
        for repeat in range(5):
            rng=np.random.default_rng(10000+100*f+repeat)
            idx=np.concatenate([rng.choice(val[y[val]==k],n,replace=False) for k in range(3)])
            rng.shuffle(idx)
            for method in ['plain','prior','multipliers','weighted','rank','stack']:
                p=outputs[method][idx]
                pred=p.argmax(1)
                rows.append(dict(fold=f,repeat=repeat,n=len(idx),probability_method=method,decision='argmax',accuracy=float(np.mean(y[idx]==pred))))
                if method in ('prior','weighted','stack'):
                    for objective in ['accuracy','log']:
                        pred=exact_assignment(p,[n]*3,objective)
                        rows.append(dict(fold=f,repeat=repeat,n=len(idx),probability_method=method,decision='exact_'+objective,accuracy=float(np.mean(y[idx]==pred))))
    pd.DataFrame(rows).to_csv(dest/'balanced_decisions.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['probability_method','decision']).accuracy.agg(['mean','std','min','max'])
    summary.to_csv(dest/'balanced_decision_summary.csv')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+',required=True);p.add_argument('--tag',default='F00')
    a=p.parse_args();main(a.models,a.tag)
