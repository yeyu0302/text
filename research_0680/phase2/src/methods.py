import numpy as np
import hashlib,json
import pandas as pd
from scipy.special import softmax,logsumexp
from sklearn.linear_model import LogisticRegression
from research_0680.src.build_proxy_features import ProxyBuilder,GROUPS,CLEAN,trusted
from research_0680.src.common import cf_splits
from research_0680.src.train_baseline import FeatureFactory,BASE,CORE,model_predict
from src.features import REAL_CAT_COLS,REAL_BIN_COLS,PROXIES,GEO_COLS,keys
from research_0680.src.leakage_safe_te import NestedTE
from src.evaluation import prior_correct
PRED_CACHE={};PHYSICAL_BUNDLES={}

def cache_key(x,y,q,config):
    return hashlib.sha256(pd.util.hash_pandas_object(x,index=False).values.tobytes()+np.asarray(y).tobytes()+
        pd.util.hash_pandas_object(q,index=False).values.tobytes()+json.dumps(config,sort_keys=True).encode()).hexdigest()

class PosteriorProxy(ProxyBuilder):
    """Empirical silver-reference prior times independent Student-t sensor likelihoods.

    Measurement sources and independence are hypotheses, not established truths.
    Only reference X fits grids, residual scales and prior masses. Query main
    measurements are preserved when trusted; latent uncertainty remains explicit.
    """
    def __init__(self):super().__init__('V0')
    def fit(self,x):
        super().fit(x);self.priors={}
        for source in GROUPS:
            a=x.loc[trusted(x,source),source].to_numpy(float)
            step={'structure_age':10,'storey_count':1,'footprint_area':50,'f_165':4}[source]
            grid,n=np.unique(np.round(a/step)*step,return_counts=True)
            self.priors[source]=(grid,(n+.5)/(n.sum()+.5*len(n)))
        return self
    def transform(self,x):
        base,unc=super().transform(x)
        for source,cols in GROUPS.items():
            grid,prior=self.priors[source];par=self.params[source]
            bias,slope,var=par['params'].T;observed=x[cols].to_numpy(float)
            expected=bias[None,:]+grid[:,None]*slope[None,:]
            sigma=np.sqrt(var)*abs(slope)
            r=(observed[:,None,:]-expected[None,:,:])/(sigma[None,None,:]+1e-6)
            # Student t, df=4: robustness without clipping away heavy-tail uncertainty.
            loglik=-2.5*np.log1p(r*r/4)-np.log(sigma[None,None,:]+1e-6)
            loglik=np.where(np.isfinite(observed[:,None,:]),loglik,0)
            logp=np.log(prior)[None,:]+np.sum(loglik,axis=2);post=softmax(logp,axis=1)
            mean=post@grid;variance=np.maximum(post@(grid**2)-mean**2,0)
            good=trusted(x,source).to_numpy();raw=x[source].to_numpy(float)
            target=CLEAN[source]
            replacement=mean.copy()
            if source=='f_165':replacement=np.where((raw<0)&(raw>=-64),abs(raw),replacement)
            if source=='footprint_area':
                ten=(raw>=1500)&(np.remainder(raw,500)==0)&(abs(raw/10-mean)<abs(raw-mean));replacement=np.where(ten,raw/10,replacement)
            base[target]=np.where(good,raw,replacement)
            name={'structure_age':'age_proxy_mean','storey_count':'storey_proxy_mean','footprint_area':'area_proxy_mean','f_165':'height_proxy_mean'}[source]
            base[name]=mean
            unc[source+'_posterior_std']=np.sqrt(variance)
            unc[source+'_posterior_entropy']=-np.sum(post*np.log(np.clip(post,1e-12,1)),axis=1)
            unc[source+'_posterior_main_gap']=np.where(np.isfinite(raw),raw-mean,0)
        base['age_blended']=base.clean_age;base['storey_blended']=base.clean_storey;base['height_blended']=base.clean_height
        base['log_clean_age']=np.log1p(base.clean_age.clip(lower=0));base['log_clean_area']=np.log1p(base.clean_area)
        return base,unc

class PosteriorFactory(FeatureFactory):
    def fit_transform(self,x,y,query):
        x=x.reset_index(drop=True);query=query.reset_index(drop=True);self.ref=x
        self.proxy_builder=PosteriorProxy().fit(x);self.clean_ref,_=self.proxy_builder.transform(x)
        self.impute={}
        for c,condition in [('f_103','f_113'),('f_175','f_104')]:
            self.impute[c]=(x.groupby(['f_141',condition])[c].mean(),x.groupby(condition)[c].mean(),float(x[c].mean()),condition)
        s=self._struct(x,self.clean_ref.copy());self.group_maps={}
        for g in GEO_COLS:
            cols=['clean_age','clean_storey','clean_area','clean_height','f_182','f_214','f_199','weak_masonry_score']
            self.group_maps[g]=(x[g].value_counts(),s.groupby(x[g])[cols].mean(),s[cols].mean())
        self.levels={c:sorted(x[c].fillna('MISSING').astype(str).unique()) for c in REAL_CAT_COLS}
        a=self.transform_base(x);b=self.transform_base(query)
        _,u=self.proxy_builder.transform(x);_,uq=self.proxy_builder.transform(query)
        # Only posterior moments are additional; measurement-spread diagnostics are unchanged.
        extra=[c for c in u if '_posterior_' in c]
        for c in extra:a['post_'+c]=u[c].to_numpy();b['post_'+c]=uq[c].to_numpy()
        self.te_builder=NestedTE('plain',self.seed);qc,_=self.proxy_builder.transform(query)
        ta,tb=self.te_builder.fit_transform(keys(x,self.clean_ref),y,keys(query,qc))
        a=pd.concat([a,ta],axis=1);b=pd.concat([b,tb],axis=1)
        cols=list(dict.fromkeys(CORE+[c for c in a if c.startswith('onehot_') or c in PROXIES or c.startswith('post_')]))
        self.columns=cols
        return a[cols],b[cols]

def physical_matrix_fit(x,query,posterior=False):
    proxy=(PosteriorProxy() if posterior else ProxyBuilder('V0')).fit(x)
    a,ua=proxy.transform(x.reset_index(drop=True));b,ub=proxy.transform(query.reset_index(drop=True))
    names=['clean_age','clean_storey','clean_area','clean_height','age_proxy_mean','storey_proxy_mean','area_proxy_mean','height_proxy_mean']
    aa=a[names].copy();bb=b[names].copy()
    for c in REAL_BIN_COLS:aa[c]=x[c].fillna(.5).to_numpy();bb[c]=query[c].fillna(.5).to_numpy()
    for c in REAL_CAT_COLS:
        levels=sorted(x[c].fillna('MISSING').astype(str).unique())
        for v in levels:
            aa[c+'_'+v]=(x[c].fillna('MISSING').astype(str)==v).to_numpy().astype(float)
            bb[c+'_'+v]=(query[c].fillna('MISSING').astype(str)==v).to_numpy().astype(float)
    for frame in [aa,bb]:
        frame['height_per_storey']=frame.clean_height/frame.clean_storey.clip(lower=1)
        frame['slenderness']=frame.clean_height/np.sqrt(frame.clean_area.clip(lower=50))
        frame['age_x_floor']=np.log1p(frame.clean_age)*frame.clean_storey
    return aa,bb

def physical_predict(x,y,query,posterior=False,kind='spline'):
    a,b=physical_matrix_fit(x.reset_index(drop=True),query.reset_index(drop=True),posterior)
    return model_predict(a,y,b,kind)

def fit_offset(base,y,prior,penalty):
    """Convex class-intercept MAP with sum-zero contrasts and Gaussian parent prior."""
    logbase=np.log(np.clip(base,1e-9,1));offset=prior.copy()
    one=np.eye(3)[y]
    def objective(v):
        logits=logbase+v
        return float(np.sum(logsumexp(logits,axis=1)-logits[np.arange(len(y)),y])+.5*penalty*np.sum((v-prior)**2))
    for _ in range(30):
        p=softmax(logbase+offset,axis=1)
        grad=(p-one).sum(0)+penalty*(offset-prior)
        h=np.diag(p.sum(0))-p.T@p+penalty*np.eye(3)
        step=np.linalg.solve(h,grad)
        if max(abs(step))>2:step*=2/max(abs(step))
        scale=1.;before=objective(offset)
        while scale>1/128:
            trial=offset-scale*step;trial-=trial.mean()
            if objective(trial)<=before+1e-9:break
            scale*=.5
        offset=trial
        if max(abs(step))<1e-6:break
    return np.clip(offset,-4,4)

class ConditionalHierarchy:
    def __init__(self,penalty=8,posterior=False):self.penalty=penalty;self.posterior=posterior
    def fit_predict(self,x,y,query):
        x=x.reset_index(drop=True);query=query.reset_index(drop=True);y=np.asarray(y)
        key=cache_key(x,y,query,{'posterior':self.posterior})
        if key in PHYSICAL_BUNDLES:oof,raw=PHYSICAL_BUNDLES[key]
        else:
            oof=np.zeros((len(x),3))
            for t,v in cf_splits(len(x),913):oof[v]=physical_predict(x.iloc[t],y[t],x.iloc[v],self.posterior)
            raw=physical_predict(x,y,query,self.posterior)
            if len(PHYSICAL_BUNDLES)>8:PHYSICAL_BUNDLES.pop(next(iter(PHYSICAL_BUNDLES)))
            PHYSICAL_BUNDLES[key]=(oof,raw)
        inherited_fit=np.zeros((len(x),3));inherited_query=np.zeros((len(query),3));self.offset_records=[]
        for group in ['f_90','f_109','f_141']:
            fit_offset_array=inherited_fit.copy();query_offset_array=inherited_query.copy()
            for key,idx in x.groupby(group).indices.items():
                idx=np.asarray(idx);parent=inherited_fit[idx].mean(0)
                value=fit_offset(oof[idx],y[idx],parent,self.penalty)
                fit_offset_array[idx]=value
                mask=query[group].to_numpy()==key;query_offset_array[mask]=value
                self.offset_records.append(dict(group=group,key=int(key),n=len(idx),offset=value.tolist(),parent=parent.tolist()))
            inherited_fit,inherited_query=fit_offset_array,query_offset_array
        p=softmax(np.log(np.clip(raw,1e-9,1))+inherited_query,axis=1)
        return prior_correct(p,np.bincount(y,minlength=3)/len(y))

def predict(x,y,query,config):
    key=cache_key(x,y,query,config)
    if key in PRED_CACHE:return PRED_CACHE[key]
    p=_predict(x,y,query,config)
    assert np.isfinite(p).all() and np.min(p)>=0 and np.allclose(p.sum(1),1)
    if len(PRED_CACHE)>30:PRED_CACHE.pop(next(iter(PRED_CACHE)))
    PRED_CACHE[key]=p
    return p

def _predict(x,y,query,config):
    from research_0680.src.train_baseline import predict_pipeline
    mode=config['kind']
    if mode=='reference':return predict_pipeline(x,y,query,BASE)[0]
    if mode=='posterior':
        a,b=PosteriorFactory(BASE).fit_transform(x,y,query)
        return prior_correct(model_predict(a,y,b),np.bincount(y,minlength=3)/len(y))
    if mode=='physical':return prior_correct(physical_predict(x,y,query,config.get('posterior',False)),np.bincount(y,minlength=3)/len(y))
    if mode=='hierarchy':return ConditionalHierarchy(config.get('penalty',8),config.get('posterior',False)).fit_predict(x,y,query)
    hierarchical=predict(x,y,query,{'kind':'hierarchy','penalty':config.get('penalty',8),'posterior':config.get('posterior',False)})
    reference=predict_pipeline(x,y,query,BASE)[0]
    weight=config.get('weight',.5)
    return (1-weight)*reference+weight*hierarchical
