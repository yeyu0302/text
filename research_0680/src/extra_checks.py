"""Bounded extra checks: exact schema, conditional negative controls and repeatability."""
import ast,json,time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from .common import *
from .train_baseline import BASE,predict_pipeline,FEATURE_CACHE
from .build_proxy_features import trusted
from .error_attribution import encode_unsupervised

def run():
    x,y,test,sample=read_data();dest=RESULTS/'01_data_forensics'
    source=PROJECT/'results/00_baseline/provided/optimize_pipeline_user.py'
    if source.exists():
        tree=ast.parse(source.read_text());cols=next(ast.literal_eval(n.value) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='REAL_BINARY_COLS' for t in n.targets))
        schema=[]
        for c in cols:
            values=x[c].dropna();numeric=pd.to_numeric(values,errors='coerce')
            schema.append(dict(feature=c,dtype=str(x[c].dtype),non_numeric_count=int(numeric.isna().sum()),
                               is_actually_binary=bool(values.isin([0,1]).all()),examples=values.head(5).tolist()))
        dump(RESULTS/'00_baseline/provided_schema_audit.json',schema)
        upload=PROJECT/'results/00_baseline/provided/submission_user_0679.csv'
        dump(RESULTS/'00_baseline/uploaded_version_mapping.json',dict(user_supplied_submission=str(upload),
            sha256=__import__('hashlib').sha256(upload.read_bytes()).hexdigest(),user_reported_kaggle=.679,
            generation_code_header='V6 Dual Plan',parameter_header='V5 13Model Hybrid',
            exact_generation_reproduced=False,reason="Unmodified supplied code fails in base cleaning when 'c3' is converted to float",provided_oof_available=False))
    # Correct explicit encoder ordering: numeric then categorical, not input order.
    candidates=['f_51','f_5','f_125','f_90'];ordered=[c for c in candidates if x[c].dtype!=object]+[c for c in candidates if x[c].dtype==object]
    target=(~trusted(x,'structure_age')|~trusted(x,'storey_count')|~trusted(x,'f_165')).astype(int).to_numpy()
    rows=[];rng=np.random.default_rng(571)
    for f,(t,v) in enumerate(StratifiedKFold(3,shuffle=True,random_state=571).split(x,target)):
        a,b=encode_unsupervised(x.iloc[t][candidates],x.iloc[v][candidates])
        a=np.column_stack([a,rng.normal(size=len(t))]);b=np.column_stack([b,rng.normal(size=len(v))])
        model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,l2_regularization=50,early_stopping=False,random_state=571).fit(a,target[t])
        initial=roc_auc_score(target[v],model.predict_proba(b)[:,1])
        for j,col in enumerate(ordered+['negative_random_control']):
            perm=b.copy()
            for _,idx in x.iloc[v].reset_index(drop=True).groupby('f_90').indices.items():perm[idx,j]=rng.permutation(perm[idx,j])
            rows.append(dict(fold=f,feature=col,encoded_column=j,conditional_permutation_auc_drop=initial-roc_auc_score(target[v],model.predict_proba(perm)[:,1])))
    pd.DataFrame(rows).to_csv(dest/'contamination_conditional_permutation_corrected.csv',index=False)
    # Pairwise-complete residual correlations handle missing sensors explicitly.
    from .build_proxy_features import GROUPS
    residual=[]
    for source,cols in GROUPS.items():
        good=trusted(x,source)
        r=x.loc[good,cols].subtract(x.loc[good,source],axis=0)
        c=r.corr(min_periods=100).to_numpy();off=c[~np.eye(len(cols),dtype=bool)]
        residual.append(dict(source=source,mean_off_diagonal=float(np.nanmean(off)),available_pairs=int(np.isfinite(off).sum()),method='pairwise-complete silver-reference residuals'))
    pd.DataFrame(residual).to_csv(dest/'proxy_residual_correlations_corrected.csv',index=False)
    # Refit both transformations and classifier twice on the same fixed small split.
    fit=np.concatenate([np.flatnonzero(y==k)[:500] for k in range(3)]);val=np.concatenate([np.flatnonzero(y==k)[500:600] for k in range(3)])
    FEATURE_CACHE.clear();start=time.perf_counter();a,_=predict_pipeline(x.iloc[fit],y[fit],x.iloc[val],BASE)
    FEATURE_CACHE.clear();b,_=predict_pipeline(x.iloc[fit],y[fit],x.iloc[val],BASE)
    difference=float(abs(a-b).max());assert np.allclose(a,b,atol=1e-10)
    dump(dest/'repeatability.json',dict(max_absolute_probability_difference=difference,runtime_seconds=time.perf_counter()-start,train_rows=len(fit),query_rows=len(val),fresh_transform_and_model_refits=2))
    print('Extra checks complete',difference,flush=True)

if __name__=='__main__':run()
