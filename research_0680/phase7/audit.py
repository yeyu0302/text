"""Cached-prediction diagnostics. Meta cross-fitting is NOT nested base validation."""
import os,json,itertools,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import roc_auc_score,balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold,GroupKFold,train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from src.features import row_features,REAL_CAT_COLS,REAL_BIN_COLS,GEO_COLS,PROXIES
from src.evaluation import exact_assignment
from research_0680.src.common import read_data,full_metrics,dump,PROJECT
from research_0680.phase6.shift_audit import solve_prior

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE7_RESULTS',str(HERE/'results_reproduced')))
NAMES={'v5':('v5_oof','v5_test'),'v11':('v11_oof','v11_test'),'bagged':('v5_v11_bagged','v5_v11_bagged')}

def probs(table,prefix):
    p=table[[f'{prefix}_p{k}' for k in [1,2,3]]].to_numpy(float)
    if not np.isfinite(p).all() or p.min()<0 or not np.allclose(p.sum(1),1,atol=2e-6,rtol=0):raise ValueError('Invalid probability matrix')
    return p/p.sum(1,keepdims=True)

def load():
    x,y,test,sample=read_data()
    o=pd.read_csv(HERE/'evidence/v5_v11_oof_probabilities.csv');q=pd.read_csv(HERE/'evidence/v5_v11_test_probabilities.csv')
    s=pd.read_csv(HERE/'evidence/submission_bagged_argmax_noquota.csv')
    if not(o.id.equals(x.id) and q.id.equals(test.id) and s.id.equals(sample.id)):raise ValueError('ID alignment mismatch')
    if not np.array_equal(o.label.to_numpy()-1,y):raise ValueError('OOF truth does not match train labels')
    a={n:probs(o,v[0]) for n,v in NAMES.items()};b={n:probs(q,v[1]) for n,v in NAMES.items()}
    if not np.array_equal(b['bagged'].argmax(1)+1,s.label):raise ValueError('Submission does not match probability argmax')
    return x,y,test,a,b

def interval(y,p,groups=None,seed=831,draws=1000):
    """Conditional uncertainty: neither refit nor selection bias included."""
    rng=np.random.default_rng(seed);correct=(p.argmax(1)==y).astype(float);values=[]
    weights=np.eye(3)[y]
    if groups is None:
        buckets=[correct[y==k] for k in range(3)]
        for _ in range(draws):values.append(np.mean([v[rng.integers(len(v),size=len(v))].mean() for v in buckets]))
    else:
        unique,codes=np.unique(groups,return_inverse=True)
        n=np.vstack([np.bincount(codes,weights=weights[:,k],minlength=len(unique)) for k in range(3)])
        c=np.vstack([np.bincount(codes,weights=weights[:,k]*correct,minlength=len(unique)) for k in range(3)])
        for _ in range(draws):
            picks=rng.integers(len(unique),size=len(unique));values.append(np.mean(c[:,picks].sum(1)/n[:,picks].sum(1)))
    return dict(low=float(np.quantile(values,.025)),high=float(np.quantile(values,.975)),point=float(np.mean([correct[y==k].mean() for k in range(3)])))

def calibrated(theta,p):return softmax(np.log(np.clip(p,1e-12,1))*np.exp(theta[0])+np.r_[theta[1:],0],axis=1)

def fit_calibration(p,y):
    # Balanced CE targets uniform-prior posterior; do not apply inverse source prior twice.
    w=1/np.bincount(y,minlength=3)[y];w/=w.mean()
    def loss(theta):return np.average(-np.log(np.maximum(calibrated(theta,p)[np.arange(len(y)),y],1e-12)),weights=w)
    fit=minimize(loss,np.zeros(3),method='L-BFGS-B',bounds=[(-1,1),(-1,1),(-1,1)],options={'ftol':1e-12})
    if not fit.success:raise RuntimeError(fit.message)
    return fit.x

def summary(x,y,test,a,b):
    best=pd.read_csv(PROJECT/'phase6/evidence/submission_3.csv');assert best.id.equals(test.id)
    rows=[];uncertainty=[]
    for name in NAMES:
        m=full_metrics(y,a[name]);rows.append(dict(model=name,**m,test_counts=np.bincount(b[name].argmax(1),minlength=3).tolist(),changed_from_best679=int(np.sum(b[name].argmax(1)+1!=best.label))))
        for scope,groups in [('row',None),('village_cluster',x.f_141.to_numpy())]:uncertainty.append(dict(model=name,resampling=scope,**interval(y,a[name],groups)))
    dump(OUT/'metrics.json',rows);pd.DataFrame(uncertainty).to_csv(OUT/'conditional_ba_intervals.csv',index=False)
    c1=a['v5'].argmax(1)==y;c2=a['v11'].argmax(1)==y
    dump(OUT/'diversity.json',dict(correctness_correlation=float(np.corrcoef(c1.astype(float),c2.astype(float))[0,1]),oof_disagreements=int(np.sum(a['v5'].argmax(1)!=a['v11'].argmax(1))),test_disagreements=int(np.sum(b['v5'].argmax(1)!=b['v11'].argmax(1))),id_ranges=dict(train=[int(x.id.min()),int(x.id.max())],test=[int(test.id.min()),int(test.id.max())]),bagged_formula='.65 * V5 + .35 * V11, within CSV rounding'))
    subgroup=[];p=a['bagged'];pred=p.argmax(1);n=x.f_141.map(x.f_141.value_counts());clean=row_features(x)
    definitions={'all':np.ones(len(x),bool),'village_n_le15':(n<=15).to_numpy(),'village_n_gt15':(n>15).to_numpy(),'age_ge80':(clean.clean_age>=80).to_numpy(),'height_ge28':(clean.clean_height>=28).to_numpy(),'age_flag':(x.structure_age.isna()|(x.structure_age>200)|(x.structure_age<0)).to_numpy()}
    for group,mask in definitions.items():
        part=y[mask];h=pred[mask];subgroup.append(dict(group=group,n=int(mask.sum()),level2_n=int(np.sum(part==1)),level2_recall=float(np.mean(h[part==1]==1)),**full_metrics(part,p[mask])))
    dump(OUT/'subgroups.json',subgroup)
    # Confidence is not a classifier selection criterion. These conditional intervals
    # do not include leaderboard split, base refitting, or repeated-search uncertainty.
    dump(OUT/'metric_scope.json',dict(user_reported_submission_kaggle_accuracy=.672,base_oof_fold_metadata_available=False,training_code_available=False,confidence_of_kaggle_above_068='Not established',test_label_prior_known=False,strict_nested_meta_validation_possible_from_these_files=False))

def calibration_audit(x,y,a):
    rows=[];foldparams=[];p=a['bagged'];base=full_metrics(y,p)['balanced_accuracy']
    for name,split in [('stratified42',StratifiedKFold(5,shuffle=True,random_state=42).split(x,y)),('stratified2026',StratifiedKFold(5,shuffle=True,random_state=2026).split(x,y)),('village_group',GroupKFold(5).split(x,y,x.f_141))]:
        out=np.zeros_like(p)
        for fold,(t,v) in enumerate(split):
            theta=fit_calibration(p[t],y[t]);out[v]=calibrated(theta,p[v]);foldparams.append(dict(split=name,fold=fold,theta=theta.tolist()))
        rows.append(dict(split=name,**full_metrics(y,out),delta_vs_bagged=full_metrics(y,out)['balanced_accuracy']-base))
        np.savez_compressed(OUT/f'calibration_{name}.npz',p=out)
    dump(OUT/'calibration_results.json',rows);dump(OUT/'calibration_parameters.json',foldparams)
    dump(OUT/'calibration_scope.json',dict(scope='Meta-calibrator labels cross-fitted only; base OOF generation is unknown',not_strict_nested='Other training-row OOF predictions may have been generated by base models trained on a meta-validation label; fold metadata would not by itself supply missing inner predictions',selection='Three fixed diagnostic splits, no calibration/grid parameter search, no candidate export'))

def prior_audit(y,a,b):
    rng=np.random.default_rng(923);rows=[];marginals=[]
    for name in NAMES:
        p=a[name];q=b[name];buckets=[p[y==k] for k in range(3)]
        c=np.column_stack([v.mean(0) for v in buckets]);estimate,_=solve_prior(c,q.mean(0));draws=[]
        for _ in range(1000):
            cb=np.column_stack([v[rng.integers(len(v),size=len(v))].mean(0) for v in buckets]);tb=q[rng.integers(len(q),size=len(q))].mean(0);draws.append(solve_prior(cb,tb)[0])
        lo,hi=np.quantile(draws,[.025,.975],axis=0)
        for k in range(3):rows.append(dict(model=name,level=k+1,estimated_prior=estimate[k],low=lo[k],high=hi[k],contains_one_third=bool(lo[k]<=1/3<=hi[k]),condition_number=np.linalg.cond(c)))
        for k in range(3):marginals.append(dict(model=name,level=k+1,train_oof_mean=p.mean(0)[k],uniform_reweighted_oof_mean=c.mean(1)[k],test_mean=q.mean(0)[k]))
    pd.DataFrame(rows).to_csv(OUT/'bbse_diagnostic.csv',index=False);pd.DataFrame(marginals).to_csv(OUT/'probability_marginals.csv',index=False)
    dump(OUT/'bbse_scope.json',dict(assumptions=['label shift: class-conditional X distribution stable','source/test classifier and transformations match','confusion matrix invertible'],unverified=['Training implementation and original fold IDs','Fold versus full-fit target encoding consistency','Uniform test class prior'],bootstrap='Conditional source-within-class and target row resampling; no fold scope matching possible from uploaded metadata; no retraining variance',action='Diagnostic only; no BBSE submission correction'))

def cached_comparisons(y,a,b):
    base=a['bagged'];rows=[]
    for name in ['results/00_baseline/fixed_baseline_seed42','phase3/results/nested_seed42','phase4/pooling_results/seed42','phase4/results/nested_seed42','phase4/pairwise_results/nested_seed42','phase5/kernel_results/seed42']:
        df=pd.read_csv(PROJECT/name/'oof_predictions.csv');assert np.array_equal(df.label.to_numpy()-1,y)
        p=df[['p1','p2','p3']].to_numpy()
        for w in [.25,.5]:
            m=full_metrics(y,(1-w)*base+w*p);rows.append(dict(cache=name,weight=w,**m,delta= m['balanced_accuracy']-full_metrics(y,base)['balanced_accuracy']))
    dump(OUT/'cache_blends.json',rows)
    # Exact integer assignment, no classifier fit. Compare identity to best only;
    # equality of class counts is not evidence that these are the same model.
    best=pd.read_csv(PROJECT/'phase6/evidence/submission_3.csv').label.to_numpy()-1;quota=[]
    for name in NAMES:
        for objective in ['accuracy','log']:
            pred=exact_assignment(b[name],[1700]*3,objective)
            quota.append(dict(model=name,objective=objective,changed_from_best679=int(np.sum(pred!=best)),changed_from_argmax=int(np.sum(pred!=b[name].argmax(1)))))
    dump(OUT/'quota_identity_only.json',quota)


def domain_audit(x,y,test):
    """X-only domain classifier; NOT a damage prediction model."""
    cat=REAL_CAT_COLS+GEO_COLS;numeric=REAL_BIN_COLS+PROXIES
    def features(df):
        clean=row_features(df);f=pd.concat([df[cat+numeric].reset_index(drop=True),clean.reset_index(drop=True)],axis=1)
        for c in cat:f[c]=f[c].fillna('MISSING').astype(str)
        return f
    source=features(x);target=features(test);rng=np.random.default_rng(623);n=np.bincount(y).min();balanced=np.concatenate([rng.choice(np.flatnonzero(y==k),n,replace=False) for k in range(3)])
    results=[]
    for name,ids in [('original_source',np.arange(len(y))),('class_balanced_source',balanced)]:
        data=pd.concat([source.iloc[ids],target],ignore_index=True);domain=np.r_[np.zeros(len(ids),int),np.ones(len(test),int)];t,v=train_test_split(np.arange(len(data)),test_size=.3,stratify=domain,random_state=12683)
        numeric_cols=[c for c in data if c not in cat]
        prep=ColumnTransformer([('numeric',make_pipeline(SimpleImputer(strategy='median'),StandardScaler()),numeric_cols),('cat',OneHotEncoder(handle_unknown='ignore',min_frequency=10),cat)])
        clf=make_pipeline(prep,LogisticRegression(C=.1,class_weight='balanced',max_iter=1000,tol=1e-5)).fit(data.iloc[t],domain[t]);score=clf.predict_proba(data.iloc[v])[:,1]
        results.append(dict(scope=name,source_rows=len(ids),target_rows=len(test),validation_rows=len(v),auc=roc_auc_score(domain[v],score),max_optimizer_iterations=int(clf[-1].n_iter_.max())))
    dump(OUT/'domain_diagnostic.json',dict(results=results,scope='One fixed X-only logistic domain diagnostic; ID excluded, fold-local preprocessing; class-balanced source uses downsampling',limitation='Near-chance AUC does not prove no shift; discrimination is not proof of TE mismatch or an actionable importance weight. No damage classifier fitted.'))

def main():
    if OUT.exists() and any(OUT.iterdir()):raise SystemExit('Choose a fresh RESEARCH0680_PHASE7_RESULTS directory; preserve old results')
    OUT.mkdir(parents=True,exist_ok=True);x,y,test,a,b=load()
    summary(x,y,test,a,b);calibration_audit(x,y,a);prior_audit(y,a,b);cached_comparisons(y,a,b);domain_audit(x,y,test)
    dump(OUT/'complete.json',dict(finished=True,no_new_submission_exported=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('Completed: identity, OOF error, uncertainty, calibration, priors, cache diversity, quota identity, domain audit.',flush=True)

if __name__=='__main__':main()
