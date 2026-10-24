"""Two compact models, fixed blends, inner selection, outer validation and gate."""
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.preprocessing import SplineTransformer,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,train_test_split
from research_0680.src.common import read_data,dump,full_metrics,bootstrap,outer_folds,save_run,manifest,PROJECT
from research_0680.src.train_baseline import BASE,CORE,FeatureFactory,model_predict
from src.features import REAL_CAT_COLS
from src.evaluation import prior_correct

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE3_RESULTS',str(HERE/'results')))
PROTOCOL=json.loads((HERE/'protocol.json').read_text())
CHOICES=PROTOCOL['choices']

def predictions(x,y,q):
    """Every learned transform fits x only; query may contain val+test."""
    x=x.reset_index(drop=True);q=q.reset_index(drop=True)
    a,b=FeatureFactory(BASE).fit_transform(x,y,q)
    prior=np.bincount(y,minlength=3)/len(y)
    result={'reference':prior_correct(model_predict(a,y,b),prior)}
    columns=CORE+[c for c in a if c.startswith('onehot_')]
    aa=a[columns].fillna(0).to_numpy(float);bb=b[columns].fillna(0).to_numpy(float)
    scaler=StandardScaler().fit(aa);aa=np.clip(scaler.transform(aa),-8,8);bb=np.clip(scaler.transform(bb),-8,8)
    continuous=[j for j in range(aa.shape[1]) if len(np.unique(aa[:,j]))>=12]
    spline=SplineTransformer(n_knots=4,degree=2,knots='quantile',include_bias=False,extrapolation='linear')
    sa=spline.fit_transform(aa[:,continuous]);sb=spline.transform(bb[:,continuous])
    # Raw linear terms preserve extrapolation and categorical indicators.
    sa=np.hstack([aa,sa]);sb=np.hstack([bb,sb])
    model=LogisticRegression(C=.1,max_iter=600,tol=1e-4).fit(sa,y)
    assert max(model.n_iter_)<600,'GAM optimizer did not converge'
    result['gam']=prior_correct(model.predict_proba(sb),prior)
    ca=a[CORE].fillna(0).copy();cb=b[CORE].fillna(0).copy()
    for c in REAL_CAT_COLS:
        ca['cat_'+c]=x[c].fillna('MISSING').astype(str)
        cb['cat_'+c]=q[c].fillna('MISSING').astype(str)
    model=CatBoostClassifier(**PROTOCOL['catboost'],loss_function='MultiClass',random_seed=913,
                             verbose=False,allow_writing_files=False)
    model.fit(ca,y,cat_features=['cat_'+c for c in REAL_CAT_COLS])
    result['cat']=prior_correct(model.predict_proba(cb),prior)
    for name in CHOICES:
        if '+' in name:
            left,right=name.split('+');result[name]=.5*(result[left]+result[right])
    for p in result.values():assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
    return result

def choose(x,y):
    values={c:[] for c in CHOICES}
    for t,v in StratifiedKFold(2,shuffle=True,random_state=391).split(x,y):
        p=predictions(x.iloc[t],y[t],x.iloc[v])
        for c in CHOICES:values[c].append(full_metrics(y[v],p[c])['balanced_accuracy'])
    scores={c:float(np.mean(s)) for c,s in values.items()};best=max(scores,key=scores.get)
    if scores[best]<=scores['reference']+.001:best='reference'
    return best,dict(means=scores,fold_scores=values)

def screen():
    x,y,_,_=read_data();t,v=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=847)
    dest=OUT/'screen';dest.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
    cache=dest/'probabilities.npz'
    if cache.exists():
        z=np.load(cache);p={c:z[c] for c in CHOICES}
    else:
        p=predictions(x.iloc[t],y[t],x.iloc[v]);np.savez_compressed(cache,ids=x.id.iloc[v],**p)
        dump(dest/'manifest.json',manifest(dict(protocol=PROTOCOL,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='exploratory holdout 3750, not full OOF',runtime_seconds=time.perf_counter()-tick)))
    rows=[dict(choice=c,scope='exploratory_3750',**full_metrics(y[v],p[c]),**bootstrap(y[v],p[c],p['reference'])) for c in CHOICES]
    pd.DataFrame(rows).to_csv(OUT/'screen.csv',index=False)
    print('SCREEN',[(r['choice'],r['balanced_accuracy']) for r in rows],flush=True)
    return max(r['delta'] for r in rows)>.001

def outer(seed):
    dest=OUT/f'nested_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return json.loads((dest/'complete.json').read_text())
    x,y,test,_=read_data();folds=outer_folds(y,seed);p=np.zeros((len(x),3));base=p.copy();tp=np.zeros((len(test),3))
    tick=time.perf_counter()
    for fold in range(5):
        t=np.flatnonzero(folds!=fold);v=np.flatnonzero(folds==fold);cache=dest/f'fold{fold}.npz'
        if cache.exists():z=np.load(cache)
        else:
            start=time.perf_counter();choice,trace=choose(x.iloc[t],y[t])
            values=predictions(x.iloc[t],y[t],pd.concat([x.iloc[v],test],ignore_index=True))
            z=dict(p=values[choice][:len(v)],reference=values['reference'][:len(v)],test=values[choice][len(v):],ids=x.id.iloc[v])
            np.savez_compressed(cache,**z)
            dump(dest/f'choice{fold}.json',dict(choice=choice,trace=trace,train_id_sha256=hashlib.sha256(x.id.iloc[t].to_numpy().tobytes()).hexdigest(),
                validation_id_sha256=hashlib.sha256(x.id.iloc[v].to_numpy().tobytes()).hexdigest(),runtime_seconds=time.perf_counter()-start))
            print('OUTER',seed,fold,choice,full_metrics(y[v],z['p']),flush=True)
        assert np.array_equal(z['ids'],x.id.iloc[v]);p[v]=z['p'];base[v]=z['reference'];tp+=z['test']/5
    meta=manifest(dict(protocol=PROTOCOL,seed=seed,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_seconds=time.perf_counter()-tick))
    save_run(dest,x.id,y,p,folds,test.id,tp,meta)
    pd.DataFrame(dict(id=x.id,p1=base[:,0],p2=base[:,1],p3=base[:,2])).to_csv(dest/'reference_probabilities.csv',index=False)
    r=dict(seed=seed,**full_metrics(y,p),**bootstrap(y,p,base),runtime_seconds=time.perf_counter()-tick)
    dump(dest/'complete.json',r);print('DONE',r,flush=True);return r

def finalize(rows):
    passed=len(rows)==2 and all(r['delta']>.001 and r['ci_low']>0 for r in rows)
    decision=dict(passed=passed,protocol=PROTOCOL,candidate=None)
    if passed:
        path=OUT/'submission_candidate_phase3.csv';assert not path.exists(),'Preserve existing output'
        x,y,test,sample=read_data();choice,trace=choose(x,y);p=predictions(x,y,test)[choice]
        sample['label']=p.argmax(1)+1;sample.to_csv(path,index=False)
        dump(OUT/'full_selection.json',dict(choice=choice,trace=trace));decision['candidate']=str(path)
    dump(OUT/'submission_gate.json',decision)

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    if not screen():
        dump(OUT/'submission_gate.json',dict(passed=False,reason='All screening gains <= .001; stop without outer training'))
        return
    rows=[outer(42),outer(2026)];pd.DataFrame(rows).to_csv(OUT/'nested_comparison.csv',index=False);finalize(rows)

if __name__=='__main__':run()
