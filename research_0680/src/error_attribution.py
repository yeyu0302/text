"""S1 evidence: neighborhood ambiguity, measurement residuals, label suspicion, shift.

All predictive claims here are diagnostic. No labels are corrected and none of
these pooled diagnostic rankings are fed into final outer-fold selection.
"""
import json,time
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp,mannwhitneyu,skew,kurtosis,chi2_contingency
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.features import SELECTED,REAL_CAT_COLS,GEO_COLS
from .common import *
from .build_proxy_features import ProxyBuilder,GROUPS,CLEAN,trusted
from .conditional_knn import ConditionalKNN

def encode_unsupervised(fit,query):
    cats=list(fit.select_dtypes('object').columns);numeric=[c for c in fit if c not in cats]
    encoder=OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1)
    a=fit[numeric].to_numpy(float);b=query[numeric].to_numpy(float)
    if cats:
        a=np.hstack([a,encoder.fit_transform(fit[cats].fillna('MISSING').astype(str))]);b=np.hstack([b,encoder.transform(query[cats].fillna('MISSING').astype(str))])
    return a,b

def fdr(values):
    p=np.asarray(values,float);order=np.argsort(p);q=p[order]*len(p)/np.arange(1,len(p)+1)
    result=np.empty(len(p));result[order]=np.minimum(1,np.minimum.accumulate(q[::-1])[::-1]);return result

def bootstrap_diff(a,b,seed=42):
    rng=np.random.default_rng(seed);v=[]
    for _ in range(1000):v.append(float(rng.choice(a,len(a),replace=True).mean()-rng.choice(b,len(b),replace=True).mean()))
    return np.quantile(v,[.025,.975]).tolist()

def run():
    dest=RESULTS/'01_data_forensics';dest.mkdir(exist_ok=True)
    x,y,test,sample=read_data();oof=pd.read_csv(RESULTS/'00_baseline/fixed_baseline_seed42/oof_predictions.csv')
    assert oof.id.equals(x.id);p=oof[['p1','p2','p3']].to_numpy();folds=oof.fold.to_numpy();pred=p.argmax(1)
    traces=[];uncertainty=[];calibrations=[];repair_rows=[];start=time.perf_counter()
    for fold in range(5):
        fit=np.flatnonzero(folds!=fold);val=np.flatnonzero(folds==fold)
        proxy=ProxyBuilder('V3').fit(x.iloc[fit]);clean,u=proxy.transform(x.iloc[val].reset_index(drop=True))
        u['id']=x.id.iloc[val].to_numpy();uncertainty.append(u);calibrations+= [dict(fold=fold,**r) for r in proxy.calibration]
        repair_rows += [dict(fold=fold,**r) for r in proxy.last_repairs]
        obj=ConditionalKNN(40).fit(x.iloc[fit],y[fit])
        for scope in ['global','f_109','region']:
            for k in [10,20,40]:
                obj.k=k;obj.scope=scope;features=obj.transform(x.iloc[val]);q=features[['knn_p1','knn_p2','knn_p3']].to_numpy()
                a=pd.DataFrame(dict(id=x.id.iloc[val].to_numpy(),true_label=y[val]+1,oof_pred=pred[val]+1,oof_p2=p[val,1],
                    neighbor_entropy=features.neighbor_entropy,neighbor_level2_rate=q[:,1],
                    neighbor_label_disagreement=1-q[np.arange(len(val)),y[val]],neighbor_effective_n=features.neighbor_effective_n,
                    knn_distance=features.knn_distance,fallback=obj.fallbacks,scope=scope,k=k,fold=fold))
                traces.append(a)
        print('S1 neighborhood fold',fold,'seconds',round(time.perf_counter()-start,1),flush=True)
    neighbors=pd.concat(traces,ignore_index=True);neighbors.to_csv(dest/'neighbor_diagnostics.csv',index=False)
    unc=pd.concat(uncertainty,ignore_index=True).set_index('id').reindex(x.id).reset_index()
    assert unc.id.equals(x.id)
    unc.to_csv(dest/'proxy_uncertainty_outer_fitted.csv',index=False)
    pd.DataFrame(calibrations).to_csv(dest/'proxy_calibration_by_fold.csv',index=False)
    dump(dest/'repair_counts_and_examples.json',repair_rows)
    # Genuine independent historical model fits, aligned by id; no label corrections.
    independent=ROOT/'results/predictions/C01_full_cat/predictions.npz'
    independent_p=np.load(independent)['p'] if independent.exists() else None
    suspicion=pd.DataFrame(dict(id=x.id,true_label=y+1,p_true=p[np.arange(len(y)),y],pred=pred+1))
    suspicion['suspicion_score']=1-suspicion.p_true
    if independent_p is not None:
        suspicion['independent_model_p_true']=independent_p[np.arange(len(y)),y]
        suspicion['cross_model_confident_disagreement']=(suspicion.p_true<.1)&(suspicion.independent_model_p_true<.1)
    suspicion=suspicion.sort_values('suspicion_score',ascending=False)
    suspicion.to_csv(dest/'suspicious_labels_all.csv',index=False)
    suspicion.query('true_label == 2 and p_true < .1').to_csv(dest/'level2_high_confidence_errors.csv',index=False)
    # Cleaned physical matches. Such matches omit latent factors; they are not proof of mislabels.
    pb=ProxyBuilder('V0').fit(x);clean,_=pb.transform(x)
    physical=clean[['clean_age','clean_storey','clean_area','clean_height']].round(0)
    for c in ['roofing_kind','f_182','f_214','f_199','f_109','f_141']:physical[c]=x[c].fillna('MISSING')
    signature=pd.util.hash_pandas_object(physical,index=False)
    duplicates=pd.DataFrame(dict(id=x.id,signature=signature,label=y+1,fold=folds))
    counts=duplicates.groupby('signature').label.nunique();conflict=counts[counts>1].index
    duplicates[duplicates.signature.isin(conflict)].to_csv(dest/'conflicting_physical_matches.csv',index=False)
    # Forensic descriptions are explicitly global exploratory summaries, never model features.
    proxy_stats=[];residual_corr=[]
    for source,cols in GROUPS.items():
        good=trusted(x,source)
        residual=x.loc[good,cols].to_numpy()-x.loc[good,source].to_numpy()[:,None]
        correlation=np.corrcoef(residual,rowvar=False)
        residual_corr.append(dict(source=source,mean_off_diagonal=float(correlation[~np.eye(len(cols),dtype=bool)].mean())))
        for c in cols:
            a=x[c].dropna();proxy_stats.append(dict(source=source,proxy=c,q01=a.quantile(.01),q50=a.median(),q99=a.quantile(.99),
                skew=skew(a),excess_kurtosis=kurtosis(a),missing=x[c].isna().mean(),duplicates=int(x[c].duplicated().sum()),
                silver_correlation=x.loc[good,c].corr(x.loc[good,source]),peer_correlation=float(x[cols].corr()[c].drop(c).mean())))
    pd.DataFrame(proxy_stats).to_csv(dest/'proxy_forensics.csv',index=False)
    pd.DataFrame(residual_corr).to_csv(dest/'proxy_residual_correlations.csv',index=False)
    # Control proxy-error associations for available structural/group covariates.
    wrong=(pred!=1)&(y==1);right=(pred==1)&(y==1);rows=[]
    controls=pd.DataFrame(dict(age=clean.clean_age,storey=clean.clean_storey,area=np.log1p(clean.clean_area),height=clean.clean_height,
                               f182=x.f_182,f214=x.f_214,group=x.f_109.astype(str),roof=x.roofing_kind.fillna('MISSING')))
    controls=pd.get_dummies(controls,dtype=float).fillna(0)
    base=np.flatnonzero(y==1);scaler=StandardScaler().fit(controls.iloc[base]);control=scaler.transform(controls.iloc[base])
    label=(pred[base]!=1).astype(int)
    for c in [c for c in unc if c.endswith('proxy_std') or c.endswith('proxy_missing') or c.endswith('main_gap')]:
        values=unc[c].to_numpy(float);l,h=bootstrap_diff(values[wrong],values[right]);test_stat=mannwhitneyu(values[wrong],values[right],alternative='two-sided')
        residual=np.zeros(len(base));z=(values[base]-values[base].mean())/(values[base].std()+1e-8)
        plain_loss=[];extra_loss=[]
        for t,v in StratifiedKFold(3,shuffle=True,random_state=91).split(base,label):
            a=LogisticRegression(C=.1,max_iter=500).fit(control[t],label[t]);b=LogisticRegression(C=.1,max_iter=500).fit(np.column_stack([control[t],z[t]]),label[t])
            plain_loss.append(log_loss(label[v],a.predict_proba(control[v])))
            extra_loss.append(log_loss(label[v],b.predict_proba(np.column_stack([control[v],z[v]]))))
        rows.append(dict(feature=c,wrong_mean=values[wrong].mean(),correct_mean=values[right].mean(),difference_ci_low=l,difference_ci_high=h,
                         p_value=test_stat.pvalue,controlled_cv_logloss_gain=np.mean(plain_loss)-np.mean(extra_loss)))
    assoc=pd.DataFrame(rows);assoc['fdr_q']=fdr(assoc.p_value);assoc.to_csv(dest/'proxy_level2_error_associations.csv',index=False)
    # Subgroups: errors cannot be inferred just from recall_2 improvements.
    subgroup=[]
    for name,values in [('roof',x.roofing_kind.fillna('MISSING')),('age_bin',pd.cut(clean.clean_age,[-1,10,30,60,200,1000]).astype(str)),('f109',x.f_109.astype(str)),('f182_storey',x.f_182.astype(str)+'_'+clean.clean_storey.round().astype(str))]:
        for value,idx in pd.Series(np.arange(len(y))).groupby(values).groups.items():
            idx=np.asarray(list(idx));mask=y[idx]==1
            if mask.sum():subgroup.append(dict(group=name,value=str(value),n=len(idx),level2_n=int(mask.sum()),
                level2_to1=int(np.sum((y[idx]==1)&(pred[idx]==0))),level2_to3=int(np.sum((y[idx]==1)&(pred[idx]==2))),recall2=float(np.mean(pred[idx][mask]==1))))
    pd.DataFrame(subgroup).to_csv(dest/'subgroup_metrics.csv',index=False)
    # Domain classifier, with and without training-class prior standardization.
    combined=pd.concat([x[SELECTED],test[SELECTED]],ignore_index=True);domain=np.r_[np.zeros(len(x)),np.ones(len(test))]
    domain_rows=[];auc_rows=[]
    for balanced in [False,True]:
        dp=np.zeros(len(combined));weights=np.ones(len(combined))
        for f,(t,v) in enumerate(StratifiedKFold(3,shuffle=True,random_state=58).split(combined,domain)):
            w=np.ones(len(t));is_train=t<len(y);prior=np.bincount(y[t[is_train]],minlength=3)/is_train.sum()
            if balanced:w[is_train]=(1/3)/prior[y[t[is_train]]]
            a,b=encode_unsupervised(combined.iloc[t],combined.iloc[v])
            model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,max_depth=3,l2_regularization=50,early_stopping=False,random_state=58)
            model.fit(a,domain[t],sample_weight=w);dp[v]=model.predict_proba(b)[:,1]
        if balanced:weights[:len(y)]=(1/3)/(np.bincount(y,minlength=3)/len(y))[y]
        auc_rows.append(dict(prior_standardized=balanced,roc_auc=roc_auc_score(domain,dp,sample_weight=weights)))
        domain_rows.append(pd.DataFrame(dict(row=np.arange(len(domain)),domain=domain,p_test=dp,prior_standardized=balanced)))
    pd.DataFrame(auc_rows).to_csv(dest/'adversarial_auc.csv',index=False);pd.concat(domain_rows).to_csv(dest/'adversarial_predictions.csv',index=False)
    drift=[]
    for c in SELECTED:
        if x[c].dtype==object or x[c].nunique()<20:
            left=x[c].fillna('MISSING').astype(str).value_counts(normalize=True);right_=test[c].fillna('MISSING').astype(str).value_counts(normalize=True)
            ix=left.index.union(right_.index);a=left.reindex(ix,fill_value=0).to_numpy();b=right_.reindex(ix,fill_value=0).to_numpy()
            psi=float(np.sum((a-b)*np.log((a+1e-5)/(b+1e-5))));table_=np.stack([a*len(x),b*len(test)])
            pv=chi2_contingency(table_)[1];smd=np.nan;ks=np.nan
        else:
            a=x[c].dropna().to_numpy();b=test[c].dropna().to_numpy();ks,pv=ks_2samp(a,b);smd=(np.mean(a)-np.mean(b))/np.sqrt((np.var(a)+np.var(b))/2+1e-8)
            bins=np.unique(np.quantile(a,np.linspace(0,1,11)));bins[0],bins[-1]=-np.inf,np.inf
            aa=np.histogram(a,bins)[0]/len(a);bb=np.histogram(b,bins)[0]/len(b);psi=float(np.sum((aa-bb)*np.log((aa+1e-5)/(bb+1e-5))))
        drift.append(dict(feature=c,psi=psi,ks=ks,p_value=pv,standardized_mean_difference=smd,train_missing=x[c].isna().mean(),test_missing=test[c].isna().mean()))
    d=pd.DataFrame(drift);d['fdr_q']=fdr(d.p_value);d.to_csv(dest/'distribution_shift.csv',index=False)
    # H6: omitted features predict observable contamination? Conditional negative control.
    candidates=['f_51','f_5','f_125'];target=(~trusted(x,'structure_age')|~trusted(x,'storey_count')|~trusted(x,'f_165')).astype(int).to_numpy()
    importance=[];rng=np.random.default_rng(571)
    for t,v in StratifiedKFold(3,shuffle=True,random_state=571).split(x,target):
        a,b=encode_unsupervised(x.iloc[t][candidates+['f_90']],x.iloc[v][candidates+['f_90']])
        a=np.column_stack([a,rng.normal(size=len(t))]);b=np.column_stack([b,rng.normal(size=len(v))])
        model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,l2_regularization=50,early_stopping=False).fit(a,target[t]);base_auc=roc_auc_score(target[v],model.predict_proba(b)[:,1])
        for j,col in enumerate(candidates+['f_90','negative_random_control']):
            perm=b.copy()
            for group,idx in x.iloc[v].reset_index(drop=True).groupby('f_90').indices.items():perm[idx,j]=rng.permutation(perm[idx,j])
            importance.append(dict(feature=col,conditional_permutation_auc_drop=base_auc-roc_auc_score(target[v],model.predict_proba(perm)[:,1])))
    pd.DataFrame(importance).to_csv(dest/'contamination_conditional_permutation.csv',index=False)
    # Neighborhood summaries include scope/k stability and inferential comparisons.
    nr=[]
    for (scope,k),g in neighbors.groupby(['scope','k']):
        a=g[(g.true_label==2)&(g.oof_pred!=2)].neighbor_entropy.to_numpy();b=g[(g.true_label==2)&(g.oof_pred==2)].neighbor_entropy.to_numpy();ci=bootstrap_diff(a,b)
        nr.append(dict(scope=scope,k=int(k),wrong_entropy=a.mean(),correct_entropy=b.mean(),delta_ci_low=ci[0],delta_ci_high=ci[1],
                       mean_distance=g.knn_distance.mean(),mean_effective_n=g.neighbor_effective_n.mean(),fallback_rate=g.fallback.mean()))
    pd.DataFrame(nr).to_csv(dest/'neighbor_summary.csv',index=False)
    plt.figure(figsize=(8,4));g=neighbors.query('scope == "region" and k == 20 and true_label == 2')
    plt.hist([g[g.oof_pred==2].neighbor_entropy,g[g.oof_pred!=2].neighbor_entropy],bins=25,density=True,label=['Correct Level 2','Wrong Level 2'],alpha=.65)
    plt.xlabel('Train-neighbor posterior entropy');plt.legend();plt.tight_layout();plt.savefig(dest/'level2_neighbor_entropy.png',dpi=160);plt.close()
    plt.figure(figsize=(8,4));plt.scatter(unc.structure_age_proxy_std[y==1],p[y==1,1],c=(pred[y==1]!=1),s=5,alpha=.3)
    plt.xlabel('Age calibrated sensor spread');plt.ylabel('OOF P(Level 2)');plt.tight_layout();plt.savefig(dest/'level2_proxy_spread.png',dpi=160);plt.close()
    dump(dest/'complete.json',dict(runtime_seconds=time.perf_counter()-start,conflicting_signatures=len(conflict),
        exact_raw_duplicate_rows=int(x.drop(columns='id').duplicated().sum()),
        warning='All label-noise rankings and pooled associations are exploratory; no labels changed; historical independent model has different outer folds.'))
    print('S1 complete',auc_rows,flush=True)

if __name__=='__main__':run()
