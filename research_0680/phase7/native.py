"""One fixed new native categorical route; no repeated baseline classifier fits."""
import json,os,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from src.features import FeatureBuilder
from research_0680.src.common import read_data,full_metrics,bootstrap,dump,outer_folds,save_run,PROJECT

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE7_NATIVE_RESULTS',str(HERE/'native_results')))
PROTOCOL=json.loads((HERE/'native_protocol.json').read_text())

def predict(x,y,q):
    builder=FeatureBuilder('proxy',native=True).fit_base(x)
    a,b=builder.base(x),builder.base(q)
    clf=LGBMClassifier(**PROTOCOL['classifier']).fit(a,y,categorical_feature='auto')
    return clf.predict_proba(b)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol_hash=hashlib.sha256((HERE/'native_protocol.json').read_bytes()).hexdigest()
    if (OUT/'screen.json').exists():
        old=json.loads((OUT/'screen.json').read_text());assert old['protocol_sha256']==protocol_hash
        if not old['promoted']:print('Cached No-Go; no refit');return
    else:
        x,y,test,_=read_data();t,v=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=847)
        cached=np.load(PROJECT/'phase3/results/screen/probabilities.npz');assert np.array_equal(cached['ids'],x.id.iloc[v])
        tick=time.perf_counter();p=predict(x.iloc[t],y[t],x.iloc[v]);reference=cached['reference']
        m=full_metrics(y[v],p);delta=m['balanced_accuracy']-full_metrics(y[v],reference)['balanced_accuracy']
        np.savez_compressed(OUT/'screen_predictions.npz',ids=x.id.iloc[v],probabilities=p,reference=reference)
        result=dict(protocol_sha256=protocol_hash,metrics=m,delta_ba=delta,runtime_seconds=time.perf_counter()-tick,promoted=delta>.002)
        dump(OUT/'screen.json',result);print('NATIVE SCREEN',result,flush=True)
        if not result['promoted']:return
    x,y,test,_=read_data();rows=[]
    for seed in [42,2026]:
        folds=outer_folds(y,seed);dest=OUT/f'seed{seed}';dest.mkdir(exist_ok=True);p=np.zeros((len(x),3));tp=np.zeros((len(test),3))
        ref=pd.read_csv(PROJECT/f'results/00_baseline/fixed_baseline_seed{seed}/oof_predictions.csv');assert ref.id.equals(x.id)
        for fold in range(5):
            t=np.flatnonzero(folds!=fold);v=np.flatnonzero(folds==fold);cache=dest/f'fold{fold}.npz'
            if not cache.exists():
                q=pd.concat([x.iloc[v],test],ignore_index=True);probs=predict(x.iloc[t],y[t],q)
                np.savez_compressed(cache,val_ids=x.id.iloc[v],p=probs[:len(v)],test=probs[len(v):])
            c=np.load(cache);assert np.array_equal(c['val_ids'],x.id.iloc[v]);p[v]=c['p'];tp+=c['test']/5
        save_run(dest,x.id,y,p,folds,test.id,tp,dict(protocol=PROTOCOL,protocol_sha256=protocol_hash))
        row=dict(seed=seed,**full_metrics(y,p),**bootstrap(y,p,ref[['p1','p2','p3']].to_numpy()))
        rows.append(row);dump(OUT/'comparison.json',rows);print('NATIVE OUTER',row,flush=True)
        if row['delta']<=.001 or row['ci_low']<=0:break
    passed=len(rows)==2 and all(r['delta']>.001 and r['ci_low']>0 for r in rows)
    dump(OUT/'gate.json',dict(passed=passed,submission_exported=False,reason='No automatic export; applicability/provenance review also required'))

if __name__=='__main__':main()
