"""Sequential S0-S6 orchestration; nested selection never sees outer labels."""
import argparse,copy,json,time,resource
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import balanced_accuracy_score
from .common import *
from .train_baseline import BASE,predict_pipeline,FEATURE_CACHE

def cfg(**kw):return {**BASE,**kw}

def evaluate_outer(name,config=None,seed=42,rule=None):
    folder='00_baseline' if name=='fixed_baseline' else {'S2':'02_proxy_ablation','S3':'03_te_ablation','S4':'04_knn_ablation','S5':'04_knn_ablation'}.get(name[:2],'05_final_candidates')
    dest=RESULTS/folder/f'{name}_seed{seed}'
    if (dest/'complete.json').exists():return json.loads((dest/'complete.json').read_text())
    x,y,test,sample=read_data();folds=outer_folds(y,seed);p=np.zeros((len(y),3));tp=np.zeros((len(test),3));decisions=[]
    dest.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    for f in range(5):
        t=np.flatnonzero(folds!=f);v=np.flatnonzero(folds==f);file=dest/f'fold_{f}.npz'
        if file.exists():
            z=np.load(file);assert np.array_equal(z['val_ids'],x.id.iloc[v]);p[v]=z['p'];tp+=z['test']/5
            decisions.append(json.loads((dest/f'selection_{f}.json').read_text()));continue
        tick=time.perf_counter();FEATURE_CACHE.clear()
        choice=copy.deepcopy(config or BASE)
        if rule:choice,trace=select_rule(x.iloc[t],y[t],rule)
        else:trace=[]
        # No validation labels are passed into this call, including selection.
        apply=pd.concat([x.iloc[v],test],ignore_index=True)
        pred,factory=predict_pipeline(x.iloc[t],y[t],apply,choice)
        p[v]=pred[:len(v)];tp+=pred[len(v):]/5
        np.savez_compressed(file,p=pred[:len(v)],test=pred[len(v):],val_ids=x.id.iloc[v])
        decision=dict(fold=f,config=choice,inner_trace=trace,train_ids=x.id.iloc[t].tolist(),val_ids=x.id.iloc[v].tolist(),
                      runtime_seconds=time.perf_counter()-tick,features=factory.columns if factory else 'consistent fold-ensemble component feature sets')
        dump(dest/f'selection_{f}.json',decision);decisions.append(decision)
        if factory:
            factory.te_builder.distribution.to_csv(dest/f'te_distribution_fold{f}.csv',index=False)
            pd.DataFrame(factory.proxy_builder.calibration).to_csv(dest/f'proxy_calibration_fold{f}.csv',index=False)
        print(name,seed,'fold',f,full_metrics(y[v],p[v]),'seconds',round(time.perf_counter()-tick,1),flush=True)
    save_run(dest,x.id,y,p,folds,test.id,tp,manifest(dict(name=name,seed=seed,outer_folds=folds.tolist(),
        config=config,selection_rule=rule,decisions=[{k:v for k,v in d.items() if k not in ['train_ids','val_ids']} for d in decisions],
        command=f'python -m research_0680.src.run_experiments --phase {"baseline" if name=="fixed_baseline" else "final"}',
        runtime_seconds=time.perf_counter()-start,peak_memory_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)))
    metrics_=full_metrics(y,p);metrics_.update(name=name,seed=seed,runtime_seconds=time.perf_counter()-start,path=str(dest))
    if name!='fixed_baseline':
        baseline=pd.read_csv(RESULTS/'00_baseline'/f'fixed_baseline_seed{seed}'/'oof_predictions.csv')
        delta=bootstrap(y,p,baseline[['p1','p2','p3']].to_numpy());metrics_.update(delta)
        pd.DataFrame([delta]).to_csv(dest/'paired_delta_bootstrap.csv',index=False)
    dump(dest/'complete.json',metrics_);return metrics_

def menu_proxy(current):return [dict(current,proxy=v) for v in ['V0','V1','V2','V3','V4','V5']]
def menu_te(current):return [dict(current,te=v) for v in ['plain','balanced','balanced_log','balanced_log8','balanced_log16','balanced_log32']]
def menu_knn(current):return [dict(current,knn=v) for v in ['none','region_k10','region_k20','region_k40']]
def menu_ablate(current):return [dict(current,ablate=v) for v in ['none','one_hot','te_interactions','proxy_raw','group_aggregate']]
def menu_model(current):return [dict(current,model=v) for v in ['spline','ordinal','hgb','blend']]

def select_rule(x,y,rule):
    current=dict(BASE);trace=[]
    menus=[('proxy',menu_proxy)]
    if rule=='adaptive':menus += [('te',menu_te),('knn',menu_knn),('ablate',menu_ablate),('model',menu_model)]
    splits=list(StratifiedKFold(2,shuffle=True,random_state=1907).split(x,y))
    for stage,menu in menus:
        scores=[]
        for c in menu(current):
            values=[]
            for t,v in splits:
                prob,_=predict_pipeline(x.iloc[t],y[t],x.iloc[v],c)
                values.append(float(balanced_accuracy_score(y[v],prob.argmax(1))))
            scores.append((float(np.mean(values)),c,values))
        # Retain the current simpler recipe unless mean inner gain exceeds .001.
        base=next(s for s in scores if s[1]==current)
        best=max(scores,key=lambda s:s[0])
        if best[0]>base[0]+.001:current=best[1]
        trace.append(dict(stage=stage,choices=[dict(config=c,mean_ba=s,fold_ba=v) for s,c,v in scores],selected=dict(current)))
    return current,trace

def screening():
    x,y,test,sample=read_data();fit,val=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=847)
    paths={'S2':'02_proxy_ablation','S3':'03_te_ablation','S4':'04_knn_ablation','S5':'04_knn_ablation'}
    schedules=[('S2',[(f'proxy_{v}',cfg(proxy=v)) for v in ['V0','V1','V2','V3','V4','V5']]),
               ('S3',[('te_'+v,cfg(te=v)) for v in ['plain','balanced','balanced_log','balanced_log8','balanced_log16','balanced_log32']]+[('fold_ensemble',cfg(strategy='fold_ensemble'))]),
               ('S4',[('knn_'+v,cfg(knn=v)) for v in ['none','region_k10','region_k20','region_k40']]+[('knn_only',cfg(knn='region_k20',ablate='all_te')),('no_te',cfg(ablate='all_te'))]),
               ('S5',[(v,cfg(ablate=v)) for v in ['none','one_hot','te_interactions','proxy_raw','proxy_uncertainty','group_aggregate','binary_crosses']]+
                [('uncertainty_control',cfg(proxy='V3')),('remove_uncertainty',cfg(proxy='V3',ablate='proxy_uncertainty')),
                 ('ordinal',cfg(model='ordinal')),('hgb',cfg(model='hgb')),('blend',cfg(model='blend')),
                 ('soft_weights',cfg(weights='soft')),('residual',cfg(residual=True))])]
    rows=[];baseline=None
    for stage,experiments in schedules:
        for name,c in experiments:
            dest=RESULTS/paths[stage]/(stage+'_'+name);dest.mkdir(parents=True,exist_ok=True)
            if (dest/'complete.json').exists():
                result=json.loads((dest/'complete.json').read_text());prob=pd.read_csv(dest/'oof_predictions.csv')[['p1','p2','p3']].to_numpy()
            else:
                tick=time.perf_counter()
                try:
                    prob,factory=predict_pipeline(x.iloc[fit],y[fit],x.iloc[val],c)
                    result=dict(stage=stage,name=name,status='complete',**full_metrics(y[val],prob),runtime_seconds=time.perf_counter()-tick,
                                config=c,interpretation='EXPLORATORY SINGLE DEVELOPMENT SPLIT; not unbiased OOF')
                    save_run(dest,x.id.iloc[val],y[val],prob,np.zeros(len(val),int),manifest=manifest(dict(config=c,fit_ids=x.id.iloc[fit].tolist(),
                        validation_ids=x.id.iloc[val].tolist(),seed=847,feature_list=factory.columns if factory else 'fold components',command='python -m research_0680.src.run_experiments --phase screening')))
                    if factory:factory.te_builder.distribution.to_csv(dest/'te_distribution.csv',index=False)
                    dump(dest/'complete.json',result)
                except Exception as exc:
                    result=dict(stage=stage,name=name,status='failed',error=repr(exc),config=c);dump(dest/'failure.json',result);rows.append(result);print(result,flush=True);continue
            if baseline is None:baseline=prob
            result.update(bootstrap(y[val],prob,baseline));rows.append(result)
            pd.DataFrame(rows).to_csv(RESULTS/'ablation_summary.csv',index=False)
            print('SCREEN',stage,name,result.get('balanced_accuracy'),result['delta'],flush=True)
        FEATURE_CACHE.clear()
    return rows

def final():
    rows=[]
    for rule in ['proxy','adaptive']:
        row=evaluate_outer('nested_'+rule,seed=42,rule=rule);rows.append(row)
    # At most two rules; repeat only those showing a positive development direction.
    promising=[r for r in rows if r['delta']>0]
    if promising:
        evaluate_outer('fixed_baseline',BASE,2026)
        for r in promising:rows.append(evaluate_outer(r['name'],seed=2026,rule=r['name'].replace('nested_','')))
    pd.DataFrame(rows).to_csv(RESULTS/'nested_rule_comparison.csv',index=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['baseline','screening','final'],required=True);args=p.parse_args()
    if args.phase=='baseline':evaluate_outer('fixed_baseline',BASE,42)
    elif args.phase=='screening':screening()
    else:final()
