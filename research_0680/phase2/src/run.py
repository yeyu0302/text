import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split
from research_0680.src.common import ROOT,PROJECT,read_data,outer_folds,full_metrics,bootstrap,dump,manifest,save_run
from .methods import predict,PRED_CACHE,PHYSICAL_BUNDLES
from .paths import PHASE,OUT,REFERENCE_RESULTS
REF={'kind':'reference'}

def phase_manifest(extra):
    import hashlib
    return manifest(dict(phase2_source_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (PHASE/'src').glob('*.py')},
                         phase2_protocol=json.loads((PHASE/'config.json').read_text()),**extra))

def clear():PRED_CACHE.clear();PHYSICAL_BUNDLES.clear()

def choose(x,y,rule):
    splits=list(StratifiedKFold(2,shuffle=True,random_state=391).split(x,y));trace=[]
    def score(config):
        scores=[]
        for t,v in splits:
            p=predict(x.iloc[t],y[t],x.iloc[v],config);scores.append(full_metrics(y[v],p)['balanced_accuracy'])
        return float(np.mean(scores)),scores
    baseline,base_scores=score(REF)
    if rule=='posterior':
        candidates=[REF,{'kind':'posterior'}];values=[]
        for c in candidates:
            mean,folds=score(c);values.append(dict(config=c,score=mean,fold_scores=folds))
        best=max(values,key=lambda r:r['score']);selected=best['config'] if best['score']>baseline+.001 else REF
        return selected,values
    penalties=[]
    for penalty in [2,8,32]:
        c={'kind':'hierarchy','penalty':penalty};mean,folds=score(c);penalties.append(dict(config=c,score=mean,fold_scores=folds))
    best=max(penalties,key=lambda r:r['score']);penalty=best['config']['penalty'];weights=[]
    for weight in [0,.25,.5,.75,1]:
        c=REF if weight==0 else ({'kind':'hierarchy','penalty':penalty} if weight==1 else {'kind':'mixture','penalty':penalty,'weight':weight})
        mean,folds=score(c);weights.append(dict(config=c,score=mean,fold_scores=folds))
    winner=max(weights,key=lambda r:r['score']);choice=winner['config'] if winner['score']>baseline+.001 else REF
    return choice,dict(penalty_stage=penalties,mixture_stage=weights,reference_score=baseline)

def screening():
    x,y,test,sample=read_data();fit,val=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=847)
    configs=[REF,{'kind':'posterior'},{'kind':'physical'}]+[{'kind':'hierarchy','penalty':a} for a in [2,8,32]]+[
        {'kind':'mixture','penalty':8,'weight':w} for w in [.25,.5,.75]]
    rows=[];baseline=None
    for i,c in enumerate(configs):
        pth=OUT/'screening'/f'case{i}';pth.mkdir(parents=True,exist_ok=True);tick=time.perf_counter()
        if (pth/'metrics.json').exists():
            p=pd.read_csv(pth/'oof_predictions.csv')[['p1','p2','p3']].to_numpy()
        else:
            p=predict(x.iloc[fit],y[fit],x.iloc[val],c)
            save_run(pth,x.id.iloc[val],y[val],p,np.zeros(len(val),int),manifest=phase_manifest(dict(config=c,scope='screening only',fit_ids=x.id.iloc[fit].tolist(),validation_ids=x.id.iloc[val].tolist())))
        if baseline is None:baseline=p
        r=dict(case=i,config=c,**full_metrics(y[val],p),**bootstrap(y[val],p,baseline),runtime_seconds=time.perf_counter()-tick)
        rows.append(r);print('PHASE2 SCREEN',i,c,r['balanced_accuracy'],flush=True)
        pd.DataFrame(rows).to_csv(OUT/'screening_summary.csv',index=False)

def outer(rule,seed):
    dest=OUT/f'nested_{rule}_seed{seed}';dest.mkdir(parents=True,exist_ok=True)
    if (dest/'complete.json').exists():return json.loads((dest/'complete.json').read_text())
    x,y,test,sample=read_data();folds=outer_folds(y,seed);p=np.zeros((len(y),3));tp=np.zeros((len(test),3));start=time.perf_counter();choices=[]
    for f in range(5):
        t=np.flatnonzero(folds!=f);v=np.flatnonzero(folds==f);saved=dest/f'fold{f}.npz'
        if saved.exists():
            z=np.load(saved);assert np.array_equal(z['ids'],x.id.iloc[v]);p[v]=z['p'];tp+=z['test']/5
            choices.append(json.loads((dest/f'choice{f}.json').read_text()));continue
        clear();tick=time.perf_counter();choice,trace=choose(x.iloc[t],y[t],rule)
        pred=predict(x.iloc[t],y[t],pd.concat([x.iloc[v],test],ignore_index=True),choice)
        p[v]=pred[:len(v)];tp+=pred[len(v):]/5
        np.savez_compressed(saved,p=pred[:len(v)],test=pred[len(v):],ids=x.id.iloc[v].to_numpy())
        item=dict(fold=f,config=choice,trace=trace,train_ids=x.id.iloc[t].tolist(),val_ids=x.id.iloc[v].tolist(),runtime_seconds=time.perf_counter()-tick)
        choices.append(item);dump(dest/f'choice{f}.json',item)
        print('PHASE2',rule,seed,'fold',f,full_metrics(y[v],p[v]),'seconds',time.perf_counter()-tick,flush=True)
    basepath=REFERENCE_RESULTS/'00_baseline'/f'fixed_baseline_seed{seed}'/'oof_predictions.csv'
    if not basepath.exists():
        from research_0680.src.run_experiments import evaluate_outer
        from research_0680.src.train_baseline import BASE
        evaluate_outer('fixed_baseline',BASE,seed)
    base=pd.read_csv(basepath);assert base.id.equals(x.id)
    b=base[['p1','p2','p3']].to_numpy()
    save_run(dest,x.id,y,p,folds,test.id,tp,phase_manifest(dict(rule=rule,seed=seed,choices=choices,
        note='Outer labels never enter proxy prior fitting, physical crossfit, regional MAP or mixture/penalty choice',runtime_seconds=time.perf_counter()-start)))
    metrics=dict(rule=rule,seed=seed,**full_metrics(y,p),**bootstrap(y,p,b),runtime_seconds=time.perf_counter()-start)
    dump(dest/'complete.json',metrics);print('PHASE2 DONE',metrics,flush=True);return metrics

def run_final():
    rows=[]
    # Pre-specified user stop condition: do not promote a route with no positive
    # screening effect. The conditional rule still chooses ALL penalties/weights
    # afresh inside each outer training set, not the screening winner.
    screen=pd.read_csv(OUT/'screening_summary.csv')
    posterior_gain=float(screen.loc[screen['case']==1,'delta'].iloc[0])
    rules=['conditional'] if posterior_gain<=.001 else ['posterior','conditional']
    dump(OUT/'screening_go_no_go.json',dict(posterior_gain=posterior_gain,posterior_promoted=posterior_gain>.001,
        formal_rules=rules,reason='Stop weak route at screening; no global winning penalty or weight is passed into outer selection'))
    for rule in rules:
        r=outer(rule,42);rows.append(r)
        # Confirmation is for the same frozen algorithm, not a new hyperparameter search.
        rows.append(outer(rule,2026))
    pd.DataFrame(rows).to_csv(OUT/'nested_comparison.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['screen','final'],required=True);a=p.parse_args()
    if a.phase=='screen':screening()
    else:run_final()
