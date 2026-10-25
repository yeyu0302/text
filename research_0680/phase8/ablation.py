"""Small paired implementation-fix experiment; never label it V5/V11 reproduction."""
import os,json,time,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from research_0680.src.common import read_data,dump,full_metrics,bootstrap,outer_folds,save_run
from .source_audit import provided
from .fold_base import FoldBase
from .fixes import SandwichFeatures,TE_PREFIXES

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE8_RESULTS',str(HERE/'ablation_results')))
PROTOCOL_PATH=Path(os.environ.get('RESEARCH0680_PHASE8_PROTOCOL',str(HERE/'ablation_protocol.json')))
PROTOCOL=json.loads(PROTOCOL_PATH.read_text())

def predict(x,y,q):
    m=provided();factory=FoldBase().fit(x);base_a=factory.base;base_b=factory.transform(q);all_base=pd.concat([base_a,base_b],ignore_index=True)
    n=len(x);Y=np.vstack([np.eye(3)[y],np.zeros((len(q),3))]);labels=np.r_[y+1,np.zeros(len(q),int)]
    ta,tb,_=m.build_nested_fold_te(all_base,Y,labels,np.zeros((len(all_base),1)),np.arange(n),np.arange(n,len(all_base)),np.array([],int),n_inner=5,seed=42,include_seismic_res=False)
    ta=ta.reset_index(drop=True);tb=tb.reset_index(drop=True)
    te_cols=[c for c in ta if '_bal_p' in c]
    a=np.column_stack([base_a[factory.numerics].to_numpy(float),ta[te_cols].to_numpy()]);b=np.column_stack([base_b[factory.numerics].to_numpy(float),tb[te_cols].to_numpy()])
    theta=[];outputs={};original_minimize=m.minimize
    def captured(*args,**kwargs):
        result=original_minimize(*args,**kwargs);theta.append(dict(success=bool(result.success),status=int(result.status),iterations=int(result.nit),message=str(result.message),gradient_inf_norm=float(np.max(abs(result.jac))),objective=float(result.fun)));return result
    m.minimize=captured
    for name in ['zero_hazard_control','active_hazard_fix']:
        train_te=ta.copy();query_te=tb.copy()
        if name=='zero_hazard_control':
            for prefix in TE_PREFIXES:
                for k in [1,3]:train_te[f'{prefix}_bal_p{k}']=1/3;query_te[f'{prefix}_bal_p{k}']=1/3
        features=SandwichFeatures().fit(base_a,train_te)
        aa=np.column_stack([a,features.transform(base_a,train_te)]);bb=np.column_stack([b,features.transform(base_b,query_te)])
        aa,bb=m.build_spline_matrix(aa,bb)
        p=m.fit_predict_bohning_softmax(aa,y+1,bb,C=PROTOCOL['model']['C'],max_iter=PROTOCOL['model']['max_iter'])
        outputs[name]=m.calibrate_for_balanced_prior(p,np.bincount(y,minlength=3)/len(y))
    return outputs,dict(optimizers=theta,numeric_columns=factory.numerics,te_columns=te_cols,training_rows=n,query_rows=len(q),true_query_labels_passed=False)

def main():
    OUT.mkdir(parents=True,exist_ok=True);x,y,test,sample=read_data();protocol_hash=hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest()
    screen_path=OUT/'screen.json'
    if not screen_path.exists() and any(OUT.iterdir()):raise SystemExit('Choose a fresh result directory; preserve partial outputs')
    if screen_path.exists():
        screen=json.loads(screen_path.read_text());assert screen['protocol_sha256']==protocol_hash
        if not screen['promoted']:print('Cached No-Go; no refit');return
    else:
        t,v=train_test_split(np.arange(len(x)),test_size=.25,stratify=y,random_state=847);tick=time.perf_counter();p,meta=predict(x.iloc[t],y[t],x.iloc[v]);metrics={k:full_metrics(y[v],a) for k,a in p.items()};delta=metrics['active_hazard_fix']['balanced_accuracy']-metrics['zero_hazard_control']['balanced_accuracy']
        screen=dict(protocol_sha256=protocol_hash,metrics=metrics,delta_ba=delta,details=meta,runtime_seconds=time.perf_counter()-tick,promoted=delta>.002 and all(r['success'] for r in meta['optimizers']))
        np.savez_compressed(OUT/'screen_predictions.npz',ids=x.id.iloc[v],**p);dump(screen_path,screen);print('FIX SCREEN',screen,flush=True)
        if not screen['promoted']:return
    comparisons=[];test_predictions=[]
    for seed in [42,2026]:
        folds=outer_folds(y,seed);dest=OUT/f'seed{seed}';dest.mkdir(exist_ok=True);p={k:np.zeros((len(x),3)) for k in ['zero_hazard_control','active_hazard_fix']};tp={k:np.zeros((len(test),3)) for k in p};converged=True
        for fold in range(5):
            t=np.flatnonzero(folds!=fold);v=np.flatnonzero(folds==fold);cache=dest/f'fold{fold}.npz'
            if not cache.exists():
                result,meta=predict(x.iloc[t],y[t],pd.concat([x.iloc[v],test],ignore_index=True));np.savez_compressed(cache,val_ids=x.id.iloc[v],**result);dump(dest/f'fold{fold}.json',meta)
            data=np.load(cache);assert np.array_equal(data['val_ids'],x.id.iloc[v]);converged&=all(r['success'] for r in json.loads((dest/f'fold{fold}.json').read_text())['optimizers'])
            for k in p:p[k][v]=data[k][:len(v)];tp[k]+=data[k][len(v):]/5
        for k in p:save_run(dest/k,x.id,y,p[k],folds,test.id,tp[k],dict(protocol=PROTOCOL,protocol_sha256=protocol_hash))
        row=dict(seed=seed,**full_metrics(y,p['active_hazard_fix']),**bootstrap(y,p['active_hazard_fix'],p['zero_hazard_control']),all_converged=converged);comparisons.append(row);dump(OUT/'comparison.json',comparisons);test_predictions.append(tp['active_hazard_fix']);print('FIX OUTER',row,flush=True)
        if not (row['delta']>.001 and row['ci_low']>0 and row['balanced_accuracy']>.680 and converged):break
    passed=len(comparisons)==2 and all(r['delta']>.001 and r['ci_low']>0 and r['balanced_accuracy']>.680 and r['all_converged'] for r in comparisons)
    dump(OUT/'gate.json',dict(passed=passed,guaranteed_kaggle_score=False))
    if passed:
        path=OUT/'submission_implementation_fix_candidate.csv';assert not path.exists();sample['label']=np.mean(test_predictions,axis=0).argmax(1)+1;sample.to_csv(path,index=False)

if __name__=='__main__':main()
