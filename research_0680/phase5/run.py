"""Joint multilevel physical model. Only two new model designs are fitted."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split,StratifiedKFold
from src.features import FeatureBuilder,row_features,REAL_BIN_COLS,REAL_CAT_COLS
from research_0680.src.common import PROJECT,read_data,outer_folds,full_metrics,bootstrap,dump,save_run,manifest

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE5_RESULTS',str(HERE/'results')))
PROTOCOL=json.loads((HERE/'protocol.json').read_text())
REFERENCE=PROJECT/'results/00_baseline'
INNER_REFERENCE=PROJECT/'phase3/results'

def digest(ids):return hashlib.sha256(np.asarray(ids).tobytes()).hexdigest()

def designs(x,q):
    """No labels accepted: all categorical/group designs are X-only."""
    x=x.reset_index(drop=True);q=q.reset_index(drop=True)
    builder=FeatureBuilder('groups').fit_base(x)
    a=builder._struct(x,row_features(x));b=builder._struct(q,row_features(q))
    physical=['clean_age','clean_storey','clean_area','clean_height','log_clean_age','log_clean_area',
        'height_per_storey','slenderness','volume_proxy','f_119_clean','weak_masonry_score','superstructure_sum',
        'age_proxy_mean','storey_proxy_mean','area_proxy_mean','height_proxy_mean']
    physical+=['f_103_imp' if c=='f_103' else 'f_175_imp' if c=='f_175' else c for c in REAL_BIN_COLS]
    aa=a[physical].to_numpy(float);bb=b[physical].to_numpy(float)
    median=np.nanmedian(aa,axis=0);aa=np.where(np.isfinite(aa),aa,median);bb=np.where(np.isfinite(bb),bb,median)
    scale=np.quantile(aa,.75,axis=0)-np.quantile(aa,.25,axis=0)
    scale=np.where(scale>1e-5,scale,np.std(aa,axis=0));scale=np.where(scale>1e-5,scale,1)
    aa=np.clip((aa-median)/scale,-5,5);bb=np.clip((bb-median)/scale,-5,5)
    nonlinear=[i for i in range(len(physical)) if len(np.unique(aa[:,i]))>10]
    cat=OneHotEncoder(handle_unknown='ignore',sparse_output=True).fit(x[REAL_CAT_COLS].fillna('MISSING').astype(str))
    da=[sparse.csr_matrix(np.hstack([aa,.15*aa[:,nonlinear]**2])),cat.transform(x[REAL_CAT_COLS].fillna('MISSING').astype(str))]
    db=[sparse.csr_matrix(np.hstack([bb,.15*bb[:,nonlinear]**2])),cat.transform(q[REAL_CAT_COLS].fillna('MISSING').astype(str))]
    groups={}
    for group,weight in PROTOCOL['geo_scales'].items():
        encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=True).fit(x[[group]].astype(str))
        ta=encoder.transform(x[[group]].astype(str));tb=encoder.transform(q[[group]].astype(str));groups[group]=(ta,tb)
        da.append(weight*ta);db.append(weight*tb)
    basic=(sparse.hstack(da,format='csr'),sparse.hstack(db,format='csr'))
    for group,columns,key in [('f_90',PROTOCOL['macro_slopes'],'macro'),('f_109',PROTOCOL['meso_slopes'],'meso')]:
        ta,tb=groups[group]
        for name in columns:
            j=physical.index(name);weight=PROTOCOL['slope_scales'][key]
            da.append(weight*ta.multiply(aa[:,j,None]));db.append(weight*tb.multiply(bb[:,j,None]))
    varying=(sparse.hstack(da,format='csr'),sparse.hstack(db,format='csr'))
    return dict(joint_intercepts=basic,joint_varying_slopes=varying)

def predict_pair(x,y,q):
    results={}
    for name,(a,b) in designs(x,q).items():
        model=LogisticRegression(C=.05,class_weight='balanced',max_iter=600,tol=1e-4).fit(a,y)
        assert max(model.n_iter_)<600,'Optimizer failed to converge'
        results[name]=model.predict_proba(b)
    return results

def predict_selected(x,y,q,choice):
    a,b=designs(x,q)[choice]
    model=LogisticRegression(C=.05,class_weight='balanced',max_iter=600,tol=1e-4).fit(a,y)
    assert max(model.n_iter_)<600
    return model.predict_proba(b)

def cached_inner(seed,fold,train_ids,val_ids):
    path=INNER_REFERENCE/f'nested_seed{seed}'/f'choice{fold}.json';record=json.loads(path.read_text())
    assert record['train_id_sha256']==digest(train_ids) and record['validation_id_sha256']==digest(val_ids)
    values=record['trace']['fold_scores']['reference']
    return float(np.mean(values)),dict(source=str(path.relative_to(PROJECT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scores=values)

def outer(seed):
    x,y,test,_=read_data();folds=outer_folds(y,seed);dest=OUT/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return json.loads((dest/'complete.json').read_text())
    p=np.zeros((len(x),3));tp=np.zeros((len(test),3));tick=time.perf_counter()
    for fold in range(5):
        t=np.flatnonzero(folds!=fold);v=np.flatnonzero(folds==fold);cache=dest/f'fold{fold}.npz'
        if not cache.exists():
            baseline,provenance=cached_inner(seed,fold,x.id.iloc[t],x.id.iloc[v]);values={m:[] for m in PROTOCOL['models']}
            for fit,val in StratifiedKFold(2,shuffle=True,random_state=391).split(x.iloc[t],y[t]):
                preds=predict_pair(x.iloc[t[fit]],y[t[fit]],x.iloc[t[val]])
                for m in values:values[m].append(full_metrics(y[t[val]],preds[m])['balanced_accuracy'])
            means={m:float(np.mean(s)) for m,s in values.items()};choice=max(means,key=means.get)
            if means[choice]<=baseline+.001:choice='reference'
            if choice=='reference':
                old=np.load(REFERENCE/f'fixed_baseline_seed{seed}'/f'fold_{fold}.npz');assert np.array_equal(old['val_ids'],x.id.iloc[v])
                result=old['p'];test_result=old['test']
            else:
                result=predict_selected(x.iloc[t],y[t],pd.concat([x.iloc[v],test],ignore_index=True),choice)
                result,test_result=result[:len(v)],result[len(v):]
            np.savez_compressed(cache,p=result,test=test_result,ids=x.id.iloc[v])
            dump(dest/f'choice{fold}.json',dict(choice=choice,means=means,inner_scores=values,reference=provenance,
                train_id_sha256=digest(x.id.iloc[t]),val_id_sha256=digest(x.id.iloc[v])))
            print('FOLD',seed,fold,choice,full_metrics(y[v],result),flush=True)
        z=np.load(cache);assert np.array_equal(z['ids'],x.id.iloc[v]);p[v]=z['p'];tp+=z['test']/5
    baseline=pd.read_csv(REFERENCE/f'fixed_baseline_seed{seed}'/'oof_predictions.csv');assert baseline.id.equals(x.id)
    save_run(dest,x.id,y,p,folds,test.id,tp,manifest(dict(protocol=PROTOCOL,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='No TE in new models; all cached reference inner-score and validation-ID hashes verified')))
    row=dict(seed=seed,**full_metrics(y,p),**bootstrap(y,p,baseline[['p1','p2','p3']].to_numpy()),runtime_seconds=time.perf_counter()-tick)
    dump(dest/'complete.json',row);print('DONE',row,flush=True);return row

def run():
    OUT.mkdir(parents=True,exist_ok=True);x,y,test,sample=read_data()
    t,v=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=847)
    old=np.load(INNER_REFERENCE/'screen/probabilities.npz');assert np.array_equal(old['ids'],x.id.iloc[v]);base=old['reference']
    tick=time.perf_counter();p=predict_pair(x.iloc[t],y[t],x.iloc[v]);rows=[]
    np.savez_compressed(OUT/'screen_probabilities.npz',ids=x.id.iloc[v],**p)
    for name,prob in p.items():rows.append(dict(model=name,scope='screen_3750',**full_metrics(y[v],prob),**bootstrap(y[v],prob,base)))
    pd.DataFrame(rows).to_csv(OUT/'screen.csv',index=False)
    dump(OUT/'manifest.json',manifest(dict(protocol=PROTOCOL,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),screen_runtime_seconds=time.perf_counter()-tick)))
    print('SCREEN',[(r['model'],r['balanced_accuracy'],r['delta']) for r in rows],flush=True)
    if max(r['delta'] for r in rows)<=.002:
        dump(OUT/'gate.json',dict(promoted=False,reason='No new screening gain > .002; stopped after two fits',new_model_fits=2));return
    primary=outer(42);formal=[primary]
    if primary['delta']>.001 and primary['ci_low']>0:formal.append(outer(2026))
    pd.DataFrame(formal).to_csv(OUT/'comparison.csv',index=False)
    passed=len(formal)==2 and all(r['delta']>.001 and r['ci_low']>0 for r in formal)
    dump(OUT/'gate.json',dict(promoted=True,passed=passed,reason='No additional failed route reruns or weak-score submission generation'))
    if passed:
        # The validated fold-model ensemble is saved unchanged; no TE or scope swap.
        tables=[pd.read_csv(OUT/f'seed{s}/test_probabilities.csv') for s in [42,2026]]
        for table in tables:assert table.id.equals(test.id)
        prob=np.mean([table[['p1','p2','p3']].to_numpy() for table in tables],axis=0)
        sample['label']=prob.argmax(1)+1;path=OUT/'submission_candidate_phase5.csv';assert not path.exists();sample.to_csv(path,index=False)

if __name__=='__main__':run()
