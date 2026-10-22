"""Create validated submissions, full-data refits and paired uncertainty artifacts.

Primary is the fixed equal blend of the independently confirmed finalist model
families. Validation of that fixed rule is separate from choosing/tuning a rule
using the same OOF rows. Full-data refits form an independently motivated
alternative, with no claim that its test accuracy is locally measurable.
"""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from src.train import ROOT, features, fit_predict, refresh_log
from src.evaluation import metrics, prior_correct, exact_assignment


def paired_bootstrap(y,a,b,seed=500,draws=1000):
    rng=np.random.default_rng(seed)
    difference=(a.argmax(1)==y).astype(float)-(b.argmax(1)==y).astype(float)
    values=[]
    for k in range(3):
        d=difference[y==k]
        values.append(d[rng.integers(0,len(d),size=(draws,len(d)))].mean(axis=1))
    delta=np.mean(values,axis=0)
    return dict(delta=float(np.mean([difference[y==k].mean() for k in range(3)])),
                ci95_low=float(np.quantile(delta,.025)),ci95_high=float(np.quantile(delta,.975)),draws=draws)


def validate_submission(sub,test):
    assert list(sub.columns)==['id','label']
    assert len(sub)==len(test)==5100 and sub.id.equals(test.id)
    assert sub.id.is_unique and not sub.isna().any().any()
    assert sub.label.isin([1,2,3]).all()
    assert sub.label.value_counts().sort_index().to_dict()=={1:1700,2:1700,3:1700}


def main():
    plan=json.loads((ROOT/'configs/confirmation_plan.json').read_text())
    configs=plan['selected']
    original=[np.load(ROOT/'results/predictions'/c['name']/'predictions.npz') for c in configs]
    confirmation=[np.load(ROOT/'results/predictions'/('R2026_'+c['name'])/'predictions.npz') for c in configs]
    all_configs=list(configs)
    decision_plan=json.loads((ROOT/'configs/decision_plan.json').read_text())
    method=decision_plan['method']
    blend42=np.mean([z['p'] for z in original],axis=0)
    blend2026=np.mean([z['p'] for z in confirmation],axis=0)
    y=original[0]['y']
    # Freeze model-family retention from development seed 42 only. Independent
    # seed 2026 reports stability and never changes the selection rule.
    keep_ensemble=method!='prior' or metrics(y,blend42)['balanced_accuracy'] >= metrics(y,original[0]['p'])['balanced_accuracy']+.001
    if not keep_ensemble:
        configs,original,confirmation=configs[:1],original[:1],confirmation[:1]
    p=np.mean([z['p'] for z in original],axis=0)
    q=np.mean([z['p'] for z in confirmation],axis=0)
    test_p=np.mean([z['test'] for z in original]+[z['test'] for z in confirmation],axis=0)
    if method!='prior':
        calibrated42=np.load(ROOT/'results/F00/predictions.npz')
        calibrated2026=np.load(ROOT/'results/F01/predictions.npz')
        p,q=calibrated42[method],calibrated2026[method]
        test_p=(calibrated42[method+'_test']+calibrated2026[method+'_test'])/2
    # Both seeds have the same preselected algorithms and feature recipes.
    # Averaging across seeds reduces split variance without label-based weights.
    y,folds=original[0]['y'],original[0]['folds']
    tr=pd.read_csv(ROOT/'train.csv');te=pd.read_csv(ROOT/'test.csv')
    table=pd.DataFrame({'id':tr.id,'label':y+1,'fold_seed42':folds,'fold_seed2026':confirmation[0]['folds']})
    for cfg,z,r in zip(configs,original,confirmation):
        for k in range(3):
            table[f'{cfg["name"]}_p{k+1}']=z['p'][:,k]
            table[f'R2026_{cfg["name"]}_p{k+1}']=r['p'][:,k]
    for k in range(3):
        table[f'final_seed42_p{k+1}']=p[:,k]
        table[f'final_seed2026_p{k+1}']=q[:,k]
    table.to_csv(ROOT/'results/oof_predictions.csv',index=False)
    rows=[dict(seed=42,**metrics(y,p)),dict(seed=2026,**metrics(y,q))]
    pd.DataFrame(rows).to_csv(ROOT/'results/finalist_validation.csv',index=False)
    uncertainty=[]
    for name,arr in [('B00_all_raw',np.load(ROOT/'results/predictions/B00_all_raw/predictions.npz')),
                     ('B03_hierarchy',np.load(ROOT/'results/predictions/B03_hierarchy/predictions.npz'))]:
        uncertainty.append(dict(comparison='seed42 fixed ensemble minus '+name,**paired_bootstrap(y,p,arr['p'])))
    base=np.load(ROOT/'results/predictions/R2026_baseline_hierarchy/predictions.npz')
    uncertainty.append(dict(comparison='seed2026 fixed ensemble minus baseline hierarchy',**paired_bootstrap(y,q,base['p'])))
    for cfg,z,r in zip(configs,original,confirmation):
        uncertainty.append(dict(comparison='seed42 ensemble minus '+cfg['name'],**paired_bootstrap(y,p,z['p'])))
        uncertainty.append(dict(comparison='seed2026 ensemble minus '+cfg['name'],**paired_bootstrap(y,q,r['p'])))
    pd.DataFrame(uncertainty).to_csv(ROOT/'results/paired_bootstrap.csv',index=False)
    # Independent seed provides a fresh evaluation of both predeclared exact
    # objectives. Preserve each repetition, do not treat draws as independent folds.
    balanced=[]
    for seed,prob,z in [(42,p,original[0]),(2026,q,confirmation[0])]:
        for f in range(5):
            val=np.flatnonzero(z['folds']==f);n=min(np.sum(y[val]==k) for k in range(3))
            for repeat in range(5):
                rng=np.random.default_rng(seed*1000+f*10+repeat)
                idx=np.concatenate([rng.choice(val[y[val]==k],n,replace=False) for k in range(3)])
                for objective in ['argmax','accuracy','log']:
                    pred=prob[idx].argmax(1) if objective=='argmax' else exact_assignment(prob[idx],[n]*3,objective)
                    balanced.append(dict(seed=seed,fold=f,repeat=repeat,objective=objective,n=len(idx),accuracy=float(np.mean(pred==y[idx]))))
    balanced=pd.DataFrame(balanced);balanced.to_csv(ROOT/'results/final_balanced_decisions.csv',index=False)
    bs=balanced.groupby(['seed','objective']).accuracy.agg(['mean','std'])
    bs.to_csv(ROOT/'results/final_balanced_summary.csv')
    # Select on development seed 42 only; seed 2026 evaluates the frozen choice.
    objective='log' if bs.loc[(42,'log'),'mean'] > bs.loc[(42,'accuracy'),'mean']+.002 else 'accuracy'
    pred=exact_assignment(test_p,[1700]*3,objective)
    primary=pd.DataFrame({'id':te.id,'label':pred+1});validate_submission(primary,te)
    if (ROOT/'results/best_submission.csv').exists():
        raise FileExistsError('Preserve existing best_submission.csv; use a fresh results directory for another delivery')
    primary.to_csv(ROOT/'results/best_submission.csv',index=False)
    np.save(ROOT/'results/final_test_probabilities.npy',test_p)
    # Equivalent leakage-safe full-label training: four inner crossfits construct
    # training TE; test statistics use all labeled training rows exclusively.
    full=[];full_rows=[]
    for cfg in configs:
        start=time.perf_counter()
        a,b=features(tr.drop(columns='label'),y,te,cfg['family'],42,cfg['smoothing'],cfg['aggregate'],cfg['raw_mode'],cfg['native'])
        params=dict(cfg['params'])
        if cfg['model'] in ('xgb','lgb'):
            params['n_jobs']=4
        if cfg['model']=='cat':
            params['thread_count']=4
        raw=fit_predict(cfg['model'],a,y,b,params,42)
        full.append(prior_correct(raw,np.bincount(y)/len(y)))
        full_rows.append(dict(model=cfg['name'],train_rows=len(y),features=a.shape[1],runtime_seconds=time.perf_counter()-start))
        print('Full-data refit',cfg['name'],full_rows[-1],flush=True)
    full_p=np.mean(full,axis=0)
    alt=pd.DataFrame({'id':te.id,'label':exact_assignment(full_p,[1700]*3,objective)+1})
    validate_submission(alt,te)
    alt.to_csv(ROOT/'results/alternative_submission.csv',index=False)
    np.save(ROOT/'results/full_refit_probabilities.npy',full_p)
    pd.DataFrame(full_rows).to_csv(ROOT/'results/full_refit_runtime.csv',index=False)
    selected=dict(models=configs,weights='Equal fixed model-family weights' if method=='prior' else 'Outer-training calibration rules; see F00/F01 calibration.json',
                  probability_method=method,decision_plan=decision_plan,
                  cross_seed_weights='Equal weights across seeds 42 and 2026',
                  ensemble_retained=keep_ensemble,
                  ensemble_retention_rule='Require +0.001 development-seed BA versus the leading model; freeze selection; report independent-seed stability without adapting',
                  considered_models=[c['name'] for c in all_configs],
                  decision_objective=objective,counts=[1700]*3,
                  primary='Cross-seed fold predictions with the frozen probability recipe, exact transportation LP',
                  alternative='Full-data refits, equal model weights, prior correction and the same decision objective',
                  validation=rows,disagreement_rows=int((primary.label!=alt.label).sum()),
                  kaggle_public_score_user_confirmed=.676,new_kaggle_score=None)
    (ROOT/'configs/selected_solution.json').write_text(json.dumps(selected,indent=2))
    refresh_log()
    print('Submissions validated',json.dumps(rows),flush=True)


if __name__=='__main__':
    main()
