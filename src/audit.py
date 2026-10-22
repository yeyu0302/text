"""Descriptive audits and controlled proxy reconstruction, no predictive selection."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from src.features import AGE_PROXIES, STOREY_PROXIES, AREA_PROXIES, HEIGHT_PROXIES
from src.evaluation import exact_assignment
import optimize_pipeline as old

ROOT=Path(__file__).resolve().parents[1]


def main():
    tr,te=pd.read_csv(ROOT/'train.csv'),pd.read_csv(ROOT/'test.csv')
    allx=pd.concat([tr.drop(columns='label'),te],ignore_index=True)
    rows=[]
    for name,df in [('train',tr),('test',te),('combined',allx)]:
        for parent,child in [('f_90','f_109'),('f_109','f_141')]:
            mapping=df.groupby(child)[parent].nunique()
            rows.append(dict(dataset=name,parent=parent,child=child,parent_groups=df[parent].nunique(),
                             child_groups=df[child].nunique(),violating_children=int((mapping>1).sum()),
                             unseen_test_children=int((~te[child].isin(tr[child])).sum())))
    pd.DataFrame(rows).to_csv(ROOT/'results/hierarchy_audit.csv',index=False)
    proxy_rows=[];correlations=[];anomalies=[]
    definitions=[('structure_age',AGE_PROXIES,lambda s:s.between(0,200),10),
                 ('storey_count',STOREY_PROXIES,lambda s:s.between(1,7),1),
                 ('footprint_area',AREA_PROXIES,lambda s:s.between(50,1450),50),
                 ('f_165',HEIGHT_PROXIES,lambda s:s.between(4,64),4)]
    for source,cols,valid,step in definitions:
        mask=valid(tr[source]) & tr[cols].notna().all(axis=1)
        df=tr.loc[mask].reset_index(drop=True)
        source_values=df[source].to_numpy(float)
        proxies=df[cols].to_numpy(float)
        fit,hold=train_test_split(np.arange(len(df)),test_size=.3,random_state=163)
        for c in cols:
            correlations.append(dict(source=source,proxy=c,n_valid=len(df),pearson=df[source].corr(df[c]),spearman=df[source].corr(df[c],method='spearman')))
        # Weights learned only on apparently valid reconstruction fit records.
        # No target label used. Source is masked on holdout during reconstruction.
        residual=proxies[fit]-source_values[fit,None]
        weights=1/np.maximum(np.var(residual,axis=0),1e-6);weights/=weights.sum()
        aggregations={'mean':proxies.mean(1),'median':np.median(proxies,axis=1),
                      'trimmed':trim_mean(proxies,proportiontocut=1/len(cols),axis=1) if len(cols)>2 else proxies.mean(1),
                      'weighted':proxies@weights}
        if source=='f_165':
            aggregations['legacy_weighted_clipped']=.28*(df.f_186+df.f_151+df.f_20)+.16*df.f_80.clip(-40,80)
        for method,values in aggregations.items():
            # Fit affine bias correction on fit records; compare on held-out valid records.
            calibrator=Ridge(alpha=1).fit(np.asarray(values)[fit,None],source_values[fit])
            for calibrated in [False,True]:
                estimate=calibrator.predict(np.asarray(values)[hold,None]) if calibrated else np.asarray(values)[hold]
                proxy_rows.append(dict(source=source,method=method,calibrated=calibrated,fit_rows=len(fit),holdout_rows=len(hold),
                                       masked_mae=mean_absolute_error(source_values[hold],estimate),
                                       masked_rmse=np.sqrt(mean_squared_error(source_values[hold],estimate)),
                                       rounded_exact_rate=float(np.mean(np.round(estimate/step)*step==source_values[hold]))))
        invalid=~valid(allx[source])
        mean_proxy=allx[cols].mean(axis=1)
        for value in sorted(allx.loc[invalid,source].dropna().unique()):
            subset=allx[source]==value
            anomalies.append(dict(source=source,unusual_value=float(value),n=int(subset.sum()),
                                  proxy_median=float(mean_proxy[subset].median()),
                                  raw_distance=float(abs(mean_proxy[subset]-value).median()),
                                  sign_flip_distance=float(abs(mean_proxy[subset]-abs(value)).median()),
                                  divide10_distance=float(abs(mean_proxy[subset]-value/10).median())))
    pd.DataFrame(proxy_rows).to_csv(ROOT/'results/proxy_reconstruction.csv',index=False)
    pd.DataFrame(correlations).to_csv(ROOT/'results/proxy_correlations.csv',index=False)
    pd.DataFrame(anomalies).to_csv(ROOT/'results/anomaly_evidence.csv',index=False)
    # Row-local corruption trials compare actual legacy repair decisions, rather
    # than assuming the corruption flag or its original value is known.
    trials=[]
    from src.features import row_features
    for source,cols,valid,step in definitions:
        good=tr.loc[valid(tr[source])].head(3000).reset_index(drop=True)
        rng=np.random.default_rng(163)
        good=good.iloc[rng.permutation(len(good))[:1000]].reset_index(drop=True)
        target=good[source].to_numpy()
        outcol={'structure_age':'clean_age','storey_count':'clean_storey','footprint_area':'clean_area','f_165':'clean_height'}[source]
        modes=['missing'] if source=='structure_age' else (['zero','sentinel'] if source=='storey_count' else (['times10'] if source=='footprint_area' else ['sign_flip','missing']))
        for mode in modes:
            corrupted=good.copy()
            corrupted[source]=np.nan if mode=='missing' else (0 if mode=='zero' else (99 if mode=='sentinel' else target*(10 if mode=='times10' else -1)))
            for method in ['mean','median','trimmed']:
                estimate=row_features(corrupted,method)[outcol].to_numpy()
                trials.append(dict(source=source,corruption=mode,aggregate=method,n=len(good),
                                   mae=float(np.nanmean(abs(target-estimate))),
                                   exact_rate=float(np.mean(target==estimate))))
    pd.DataFrame(trials).to_csv(ROOT/'results/controlled_corruption.csv',index=False)
    # A deterministic failure case for legacy cardinality repair (ties remain
    # across all samples; capped 50 single-row moves cannot meet all quotas).
    tied=np.tile([.98,.01,.01],(300,1))
    legacy,_=old.enforce_exact_cardinality(tied,[100]*3)
    rng=np.random.default_rng(19)
    counter=None
    for attempt in range(100):
        p=rng.dirichlet([1,1,1],size=12)
        greedy,_=old.enforce_exact_cardinality(p,[4]*3)
        exact=exact_assignment(p,[4]*3,'log')
        gs=np.log(p)[np.arange(12),greedy].sum();es=np.log(p)[np.arange(12),exact].sum()
        if es>gs+1e-7:
            counter=dict(attempt=attempt,greedy_log_objective=gs,exact_log_objective=es,
                         greedy_counts=np.bincount(greedy,minlength=3).tolist(),p=p.tolist())
            break
    (ROOT/'results/cardinality_audit.json').write_text(json.dumps(dict(
        tied_legacy_counts=np.bincount(legacy,minlength=3).tolist(),requested=[100]*3,
        suboptimal_counterexample=counter),indent=2))
    packages=['numpy','pandas','scipy','scikit-learn','xgboost-cpu','lightgbm','catboost','pytest']
    meta=dict(python=platform.python_version(),packages={p:importlib.metadata.version(p) for p in packages},
              files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'train.csv',ROOT/'test.csv',ROOT/'sample_submission.csv',ROOT/'optimize_pipeline.py',ROOT/'best_params_optimized.json',ROOT/'methodology_report.md']},
              train_shape=tr.shape,test_shape=te.shape,train_counts=tr.label.value_counts().sort_index().to_dict(),
              duplicate_train_feature_rows=int(tr.drop(columns=['id','label']).duplicated().sum()),
              id_used_as_predictor=False,external_data_used=False)
    (ROOT/'results/reproducibility.json').write_text(json.dumps(meta,indent=2))
    print('Audits complete',flush=True)


if __name__=='__main__':
    main()
