"""Diagnostic LMMSE on silver references, not a new damage classifier."""
import os,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from research_0680.src.common import read_data,dump
from research_0680.src.build_proxy_features import trusted
from src.features import AGE_PROXIES,HEIGHT_PROXIES,row_features

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE6_RESULTS',str(HERE/'results')))
SOURCES={'structure_age':AGE_PROXIES,'f_165':HEIGHT_PROXIES}

def fit_linear(ref,source):
    cols=SOURCES[source];good=trusted(ref,source)
    a=ref.loc[good,cols].to_numpy(float);target=ref.loc[good,source].to_numpy(float)
    median=np.nanmedian(a,axis=0);a=np.where(np.isfinite(a),a,median)
    mean=a.mean(0);mean_target=target.mean();center=a-mean
    cov=center.T@center/len(a);cross=center.T@(target-mean_target)/len(a)
    coef=np.linalg.solve(cov+1e-6*np.eye(len(cols)),cross)
    return dict(median=median,mean=mean,mean_target=mean_target,coef=coef,silver_n=len(a))

def predict_linear(params,query,source):
    a=query[SOURCES[source]].to_numpy(float)
    return params['mean_target']+(np.where(np.isfinite(a),a,params['median'])-params['mean'])@params['coef']

def run():
    OUT.mkdir(parents=True,exist_ok=True)
    train,y,test,_=read_data()
    x=pd.concat([train.assign(dataset='train'),test.assign(dataset='test')],ignore_index=True)
    assert len(x)==20100
    errors=[];parameters={}
    for source,cols in SOURCES.items():
        full=fit_linear(x,source);parameters[source]=full
        p=np.zeros(len(x));fold_id=np.full(len(x),-1)
        for fold,(t,v) in enumerate(KFold(5,shuffle=True,random_state=813).split(x)):
            p[v]=predict_linear(fit_linear(x.iloc[t],source),x.iloc[v],source);fold_id[v]=fold
        good=trusted(x,source).to_numpy()&x[cols].notna().any(axis=1).to_numpy()
        query=x.copy();query[source]=np.nan
        legacy=row_features(query)['clean_age' if source=='structure_age' else 'clean_height'].to_numpy()
        full_pred=predict_linear(full,x,source);truth=x[source].to_numpy(float)
        means=x[cols].mean(axis=1).to_numpy()
        estimates={'raw_proxy_mean':means,'legacy_masked_repair':legacy,'lmmse_oof':p,'lmmse_full20100_in_sample':full_pred}
        for method,estimate in estimates.items():
            for i in np.flatnonzero(good):
                errors.append(dict(dataset=x.dataset.iloc[i],id=int(x.id.iloc[i]),source=source,fold=int(fold_id[i]),method=method,
                                   silver_value=truth[i],estimate=estimate[i],error=estimate[i]-truth[i]))
    records=pd.DataFrame(errors);records.to_csv(OUT/'measurement_errors.csv',index=False)
    summary=[]
    for (source,method),part in records.groupby(['source','method']):
        for name,mask in [('all',np.ones(len(part),bool)),('age_ge80' if source=='structure_age' else 'height_ge28',part.silver_value.to_numpy()>=(80 if source=='structure_age' else 28))]:
            sub=part.loc[mask];e=sub.error.to_numpy()
            summary.append(dict(source=source,method=method,subset=name,n=len(e),rmse=float(np.sqrt(np.mean(e**2))),mae=float(np.mean(abs(e))),bias=float(np.mean(e))))
    result=pd.DataFrame(summary)
    for i,row in result.iterrows():
        baseline=result.loc[(result.source==row.source)&(result.method=='raw_proxy_mean')&(result.subset==row.subset),'rmse'].iloc[0]
        result.loc[i,'rmse_reduction_vs_proxy_mean_pct']=100*(1-row.rmse/baseline)
        legacy=result.loc[(result.source==row.source)&(result.method=='legacy_masked_repair')&(result.subset==row.subset),'rmse'].iloc[0]
        result.loc[i,'rmse_reduction_vs_legacy_pct']=100*(1-row.rmse/legacy)
    serial={source:{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in par.items()} for source,par in parameters.items()}
    result.to_csv(OUT/'wiener_summary.csv',index=False);dump(OUT/'full20100_linear_mmse_parameters.json',serial)
    dump(OUT/'measurement_protocol.json',dict(rows=20100,fit_uses_damage_labels=False,folds=5,seed=813,
        interpretation='LMMSE affine estimator; equals full MMSE only under appropriate joint distribution assumptions',
        silver_reference='Main fields within predeclared valid ranges; not confirmed latent ground truth',
        full_fit_scope='Explicitly transductive X-only fit on train+test, in-sample reconstruction reported separately',
        oof_scope='Held-out physical main value never enters coefficient/prior fitting; only proxies predict it',
        target_claims=dict(age_rmse_reduction_pct=22.2,height_rmse_reduction_pct=38.9),
        limitation='Actual corrupted main values have no known true physical values; their RMSE and the supplied unpublished filter cannot be directly verified. No damage classifier was retrained.'))
    print(result.to_string(index=False),flush=True)

if __name__=='__main__':run()
