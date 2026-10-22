"""Cached, auditable five-fold experiments. python -m src.train --help"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from src.features import FeatureBuilder
from src.evaluation import metrics, prior_correct, folds_for

ROOT = Path(__file__).resolve().parents[1]
CFG = json.loads((ROOT/'configs/experiments.json').read_text())


def source_replace(source,old,new,count=1):
    if source.count(old)!=count:
        raise RuntimeError(f'Legacy source changed; audit adapter substitution: {old}')
    return source.replace(old,new)


def legacy_nested(x, y, apply, seed):
    """Faithful nested original FE, explicitly transductive in X only.

    Fix global class prior inside each inner TE fit. Unsupervised train+apply
    statistics match the legacy transductive assumption for this audit control.
    """
    import optimize_pipeline as old
    source = inspect.getsource(old.build_clean_and_engineered_features)
    source = source_replace(source,'def compute_te(fit_idx, apply_idx):',
                            'def compute_te(fit_idx, apply_idx):\n            global_prior = Y_tr_oh[fit_idx].mean(axis=0)')
    source = source_replace(source,'p_bal = apply_prior_shift(te_all, global_prior, alpha=1.0)',
                            'encoding_priors = np.tile(global_prior, (len(te_all), 1))\n'
                            '        for inner_fold in np.unique(folds_arr):\n'
                            '            encoding_priors[np.flatnonzero(folds_arr == inner_fold)] = Y_tr_oh[folds_arr != inner_fold].mean(axis=0)\n'
                            '        p_bal = te_all / encoding_priors\n'
                            '        p_bal /= p_bal.sum(axis=1, keepdims=True)')
    namespace = dict(vars(old))
    exec(compile(source, '<nested-legacy-feature-audit>', 'exec'), namespace)
    x, apply = x.copy(), apply.copy()
    x['label'] = y+1
    a,b = namespace['build_clean_and_engineered_features'](x, apply, y, folds_for(y,seed,4))
    return a.reset_index(drop=True),b.reset_index(drop=True)


def features(tr, y, apply, family, seed, smoothing, aggregate, raw_mode, native):
    if family == 'legacy_nested':
        return legacy_nested(tr,y,apply,seed)
    if family == 'core':
        a,b=FeatureBuilder('full',seed,smoothing,aggregate,raw_mode,False).fit_transform(tr,y,apply)
        tree=ast.parse((ROOT/'optimize_pipeline.py').read_text())
        cols=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
                  and any(isinstance(t,ast.Name) and t.id=='gbdt_cols' for t in n.targets))
        return a[cols],b[cols]
    return FeatureBuilder(family,seed,smoothing,aggregate,raw_mode,native).fit_transform(tr,y,apply)


def custom_predict(kind, a, y, b, priors, seed):
    """Reuse original audited training math; train on a, predict b only.

    Execute just the designated fold; no validation labels supplied. Scaling and
    quantile estimates now fit training rows only. Source hash is recorded.
    """
    import optimize_pipeline as old
    fn = old.fit_predict_spline_softmax if kind == 'spline' else old.fit_predict_fast_hist_gbdt
    source = inspect.getsource(fn)
    source = source_replace(source,'n_folds = len(np.unique(folds_arr))','n_folds = 1')
    source = source_replace(source,'for f in np.unique(folds_arr):','for f in [1]:')
    if kind=='hist':
        source = source_replace(source,'42 + int(f) * 10', str(seed))
        source = source_replace(source,'valid = col[~np.isnan(col)]', 'fit_col = X_tr[:, j][folds_arr == 0]\n        valid = fit_col[~np.isnan(fit_col)]')
    else:
        source = source_replace(source,'np.median(X_tr_raw, axis=0)', 'np.median(X_tr_raw[folds_arr == 0], axis=0)')
        source = source_replace(source,'np.percentile(X_tr_raw,', 'np.percentile(X_tr_raw[folds_arr == 0],',2)
        source = source_replace(source,'np.std(X_tr_raw, axis=0)', 'np.std(X_tr_raw[folds_arr == 0], axis=0)')
    namespace = dict(vars(old))
    exec(compile(source, '<fold-local-legacy-model>', 'exec'), namespace)
    if kind == 'hist' and 'clean_age' in a:
        tree = ast.parse((ROOT/'optimize_pipeline.py').read_text())
        cols = next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id == 'gbdt_cols' for t in n.targets))
        a,b = a[cols],b[cols]
    x = pd.concat([a,b],ignore_index=True)
    fake_y = np.r_[y,np.zeros(len(b),dtype=int)]
    f = np.r_[np.zeros(len(a),dtype=int), np.ones(len(b),dtype=int)]
    p,_ = namespace[fn.__name__](x,fake_y,f,b,priors)
    return p[len(a):]


def fit_predict(model, a, y, b, params, seed):
    if model in ('hist','spline'):
        return custom_predict(model,a,y,b,np.bincount(y,minlength=3)/len(y),seed)
    if model == 'xgb':
        from xgboost import XGBClassifier
        est = XGBClassifier(**params,random_state=seed,enable_categorical=True)
    elif model == 'lgb':
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(**params,random_state=seed)
    elif model == 'cat':
        from catboost import CatBoostClassifier
        est = CatBoostClassifier(**params,random_seed=seed)
        cat_cols = list(a.select_dtypes('category').columns)
        if cat_cols:
            a,b = a.copy(),b.copy()
            for c in cat_cols:
                a[c] = a[c].astype(str).fillna('MISSING')
                b[c] = b[c].astype(str).fillna('MISSING')
        else:
            # CatBoost needs finite numeric values; tree-aware missing handling
            # differs from XGBoost but the sentinel is fixed, not fitted on holdout.
            a,b = a.fillna(-999),b.fillna(-999)
        est.fit(a,y,cat_features=cat_cols)
        return est.predict_proba(b)
    elif model == 'spline_standard':
        # Preserve the original clipped linear/quadratic feature map while
        # replacing its fixed-curvature heuristic with a converged standard
        # multinomial optimizer. C=1/35 matches the original L2 coefficient;
        # the standard intercept is unpenalized rather than penalized by 1e-3.
        raw=a.fillna(0).to_numpy(float)
        median=np.median(raw,axis=0)
        iqr=np.percentile(raw,75,axis=0)-np.percentile(raw,25,axis=0)
        scale=np.where(iqr>1e-5,iqr,np.std(raw,axis=0));scale=np.where(scale>1e-5,scale,1.)
        za=np.clip((raw-median)/scale,-5,5)
        zb=np.clip((b.fillna(0).to_numpy(float)-median)/scale,-5,5)
        a=np.hstack([za,.15*za**2]);b=np.hstack([zb,.15*zb**2])
        est=LogisticRegression(**params,random_state=seed)
    elif model == 'logistic':
        impute, scale = SimpleImputer(), StandardScaler()
        a = scale.fit_transform(impute.fit_transform(a))
        b = scale.transform(impute.transform(b))
        est = LogisticRegression(C=.02,max_iter=500,random_state=seed)
    else:
        raise ValueError(model)
    est.fit(a,y)
    if model in ('logistic','spline_standard') and np.any(est.n_iter_ >= est.max_iter):
        raise RuntimeError('Logistic regression did not converge; increase the explicit iteration budget')
    return est.predict_proba(b)


def run(name, family='full', model='xgb', seed=42, overrides=None, smoothing=1., aggregate='mean', raw_mode='both', native=False):
    dest = ROOT/'results/predictions'/name
    dest.mkdir(parents=True,exist_ok=True)
    code_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src').glob('*.py')))+(ROOT/'optimize_pipeline.py').read_bytes()).hexdigest()
    params = {**CFG.get(model,{}),**(overrides or {})}
    config = dict(name=name,family=family,model=model,seed=seed,params=params,smoothing=smoothing,
                  aggregate=aggregate,raw_mode=raw_mode,native=native,code_hash=code_hash)
    if (dest/'summary.json').exists():
        saved = json.loads((dest/'summary.json').read_text())
        # Never silently overwrite a completed experiment.
        if any(saved[k] != v for k,v in config.items() if k != 'code_hash'):
            raise ValueError(f'Existing experiment {name} has a different configuration')
        print('CACHED',name,saved['balanced_accuracy'],flush=True)
        return saved
    tr, te = pd.read_csv(ROOT/'train.csv'),pd.read_csv(ROOT/'test.csv')
    y = tr.label.to_numpy()-1
    x = tr.drop(columns='label')
    folds = folds_for(y,seed)
    raw = np.zeros((len(x),3)); corrected = raw.copy(); test = np.zeros((len(te),3)); raw_test=test.copy()
    rows=[]; times=[]; n_features=[]
    start=time.perf_counter()
    for f in range(5):
        tick=time.perf_counter()
        train,val = np.flatnonzero(folds!=f),np.flatnonzero(folds==f)
        # Validation labels are deliberately not passed to feature or model code.
        apply=pd.concat([x.iloc[val],te],ignore_index=True)
        a,b=features(x.iloc[train],y[train],apply,family,seed,smoothing,aggregate,raw_mode,native)
        fit_seed=seed+10*f if model=='hist' else seed+f
        p=fit_predict(model,a,y[train],b,params,fit_seed)
        assert p.shape==(len(apply),3) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
        q=prior_correct(p,np.bincount(y[train],minlength=3)/len(train))
        raw[val],corrected[val]=p[:len(val)],q[:len(val)]
        raw_test += p[len(val):]/5; test += q[len(val):]/5
        elapsed=time.perf_counter()-tick;times.append(elapsed);n_features.append(a.shape[1])
        rows.append(dict(fold=f,runtime_seconds=elapsed,features=a.shape[1],model_seed=fit_seed,**metrics(y[val],q[:len(val)])))
        print(name,'fold',f,'BA',round(rows[-1]['balanced_accuracy'],6),'seconds',round(elapsed,1),flush=True)
    np.savez_compressed(dest/'predictions.npz',raw=raw,p=corrected,test=test,raw_test=raw_test,y=y,folds=folds)
    pd.DataFrame(rows).to_csv(dest/'fold_metrics.csv',index=False)
    summary={**config,**metrics(y,corrected), 'fold_std':float(np.std([r['balanced_accuracy'] for r in rows],ddof=1)),
             'runtime_seconds':time.perf_counter()-start,'features_min':min(n_features),'features_max':max(n_features),
             'hypothesis':CFG['hypotheses'].get(family,'Compare model, regularization or encoding design'),
             'status':'complete','validation':'nested fold-local; fixed tree budgets; no outer early stopping',
             'raw_metrics':metrics(y,raw)}
    (dest/'summary.json').write_text(json.dumps(summary,indent=2))
    print('DONE',name,summary['balanced_accuracy'],flush=True)
    refresh_log()
    return summary


def refresh_log():
    rows=[json.loads(p.read_text()) for p in sorted((ROOT/'results/predictions').glob('*/summary.json'))]
    for r in rows:
        r['params']=json.dumps(r['params'],sort_keys=True)
        r['raw_metrics']=json.dumps(r['raw_metrics'],sort_keys=True)
        r['eligible_for_final_selection']=r['family']!='legacy_nested'
    legacy_path=ROOT/'results/legacy_reproduction/predictions.npz'
    if legacy_path.exists():
        z=np.load(legacy_path)
        meta=json.loads((ROOT/'results/legacy_reproduction/metadata.json').read_text())
        for model,key in [('spline','spline'),('hist','hist'),('ensemble','ensemble')]:
            per_fold=[metrics(z['y'][z['folds']==f],z[key][z['folds']==f])['balanced_accuracy'] for f in range(5)]
            rows.append(dict(name='A00_leaked_'+model,family='legacy_leaked',model=model,seed=42,
                             **metrics(z['y'],z[key]),fold_std=float(np.std(per_fold,ddof=1)),
                             runtime_seconds=meta['runtime_seconds'],runtime_scope='shared entire legacy reproduction pipeline',
                             params=json.dumps(meta['models']),features_min=55 if model=='hist' else 240,
                             validation='KNOWN CROSS-FOLD LABEL LEAKAGE: diagnostic reproduction only',
                             hypothesis='Reproduce historical implementation; not eligible for optimization',
                             eligible_for_final_selection=False,status='complete'))
    for comparison_path in sorted((ROOT/'results').glob('F*/comparison.csv')):
        parent=comparison_path.parent
        meta=json.loads((parent/'calibration.json').read_text())
        fold_metrics=pd.read_csv(parent/'fold_metrics.csv')
        for row in pd.read_csv(comparison_path).to_dict('records'):
            method=row.pop('method')
            rows.append(dict(name=parent.name+'_'+method,family='nested_calibration',model=method,
                             seed=meta['cv_seed'],**row,params=json.dumps(meta),code_hash=meta.get('code_hash'),
                             fold_std=float(fold_metrics.loc[fold_metrics.method==method,'balanced_accuracy'].std()),
                             runtime_seconds=meta['runtime_seconds'],runtime_scope='shared calibration and ensemble pipeline',
                             validation='Strict outer-training-only calibration holdout',
                             hypothesis='Decision or ensemble rule fitted without outer holdout labels',
                             eligible_for_final_selection=method in ['prior','multipliers','weighted','stack'],status='complete'))
    table=pd.DataFrame(rows)
    if len(table):
        baseline=table.loc[table.name=='B00_all_raw','balanced_accuracy']
        table['delta_vs_raw_baseline']=table.balanced_accuracy-(baseline.iloc[0] if len(baseline) else np.nan)
        table.to_csv(ROOT/'results/experiment_log.csv',index=False)
        table['rank_within_seed']=table.groupby('seed').balanced_accuracy.rank(ascending=False,method='min')
        table.sort_values(['seed','balanced_accuracy'],ascending=[True,False]).to_csv(ROOT/'results/model_comparison.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True);p.add_argument('--family',default='full')
    p.add_argument('--model',default='xgb');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--params',default='{}');p.add_argument('--smoothing',type=float,default=1.)
    p.add_argument('--aggregate',default='mean',choices=['mean','median','trimmed'])
    p.add_argument('--raw-mode',default='both',choices=['both','repaired'])
    p.add_argument('--native',action='store_true')
    a=p.parse_args()
    run(a.name,a.family,a.model,a.seed,json.loads(a.params),a.smoothing,a.aggregate,a.raw_mode,a.native)
