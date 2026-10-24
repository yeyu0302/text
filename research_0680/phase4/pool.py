"""Reproduce fixed pooling and submit-ready experimental CSVs from saved predictions."""
import json,os
from pathlib import Path
import numpy as np
import pandas as pd
from src.evaluation import exact_assignment
from research_0680.src.common import PROJECT,read_data,full_metrics,bootstrap,save_run,manifest

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE4_POOL_RESULTS',str(HERE/'pooling_results')))
PROTOCOL=json.loads((HERE/'pooling_protocol.json').read_text())

def pool(ps):
    arithmetic=np.mean(ps,axis=0)
    geometric=np.exp(np.mean(np.log(np.clip(ps,1e-9,1)),axis=0));geometric/=geometric.sum(1,keepdims=True)
    return .5*(arithmetic+geometric)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    names=['submission_candidate_phase4_pool_argmax.csv','submission_candidate_phase4_pool_balanced1700.csv']
    assert not any((OUT/n).exists() for n in names),'Use a new output directory to preserve previous candidates'
    folders=[Path(os.environ.get(env,str(PROJECT/relative))) for env,relative in [
        ('RESEARCH0680_PHASE3_RESULTS','phase3/results'),('RESEARCH0680_PHASE4_RESULTS','phase4/results'),
        ('RESEARCH0680_PHASE4_PAIR_RESULTS','phase4/pairwise_results')]]
    x,y,test,sample=read_data();formal=[];panels=[];testparts=[]
    for seed in [42,2026]:
        tables=[pd.read_csv(f/f'nested_seed{seed}/oof_predictions.csv') for f in folders]
        assert all(t.id.equals(x.id) for t in tables)
        p=pool([t[['p1','p2','p3']].to_numpy() for t in tables]);folds=tables[0].fold.to_numpy()
        ref=pd.read_csv(folders[0]/f'nested_seed{seed}/reference_probabilities.csv')[['p1','p2','p3']].to_numpy()
        testtables=[pd.read_csv(f/f'nested_seed{seed}/test_probabilities.csv') for f in folders]
        assert all(t.id.equals(test.id) for t in testtables)
        tp=pool([t[['p1','p2','p3']].to_numpy() for t in testtables]);testparts.append(tp)
        save_run(OUT/f'seed{seed}',x.id,y,p,folds,test.id,tp,manifest(dict(pooling_protocol=PROTOCOL,
                 scope='exploratory fixed hybrid pool of previously inspected component rules')))
        formal.append(dict(seed=seed,**full_metrics(y,p),**bootstrap(y,p,ref)))
        for fold in range(5):
            ids=np.flatnonzero(folds==fold);n=min(np.sum(y[ids]==k) for k in range(3))
            for repeat in range(5):
                rng=np.random.default_rng(seed+5000+fold*100+repeat)
                idx=np.concatenate([rng.choice(ids[y[ids]==k],n,replace=False) for k in range(3)]);rng.shuffle(idx)
                for method,pp in [('pool',p),('reference',ref)]:
                    for decision,pred in [('argmax',pp[idx].argmax(1)),('exact_log',exact_assignment(pp[idx],[n]*3,'log'))]:
                        panels.append(dict(seed=seed,fold=fold,repeat=repeat,method=method,decision=decision,n=len(idx),accuracy=float(np.mean(pred==y[idx]))))
    pd.DataFrame(formal).to_csv(OUT/'oof_comparison.csv',index=False)
    pd.DataFrame(panels).to_csv(OUT/'balanced_panels.csv',index=False)
    pd.DataFrame(panels).groupby(['seed','method','decision']).accuracy.agg(['mean','std','min','max']).to_csv(OUT/'balanced_summary.csv')
    tp=np.mean(testparts,axis=0)
    pd.DataFrame(dict(id=test.id,p1=tp[:,0],p2=tp[:,1],p3=tp[:,2])).to_csv(OUT/'candidate_probabilities.csv',index=False)
    for name,pred in zip(names,[tp.argmax(1),exact_assignment(tp,[1700]*3,'log')]):
        table=sample.copy();table['label']=pred+1;assert table.label.isin([1,2,3]).all();table.to_csv(OUT/name,index=False)
    print('Exported experimental candidates; no measured Kaggle score',flush=True)

if __name__=='__main__':run()
