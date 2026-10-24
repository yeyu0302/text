"""Fold-local feature and model factory shared by every controlled experiment."""
import ast,copy,time
import hashlib,json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from src.features import FeatureBuilder,SELECTED,PROXIES,REAL_CAT_COLS,REAL_BIN_COLS,GEO_COLS,keys
from src.evaluation import prior_correct
from .common import ROOT
from .build_proxy_features import ProxyBuilder
from .leakage_safe_te import NestedTE
from .conditional_knn import ConditionalKNN

BASE=dict(proxy='V0',te='plain',knn='none',model='spline',ablate='none',strategy='full',weights='none',residual=False)
CORE=next(ast.literal_eval(n.value) for n in ast.walk(ast.parse((ROOT/'optimize_pipeline.py').read_text()))
          if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='gbdt_cols' for t in n.targets))
FEATURE_CACHE={}

class FeatureFactory(FeatureBuilder):
    def __init__(self,cfg,seed=913):
        super().__init__('full',seed)
        self.cfg=cfg;self.seed=seed
    def fit_transform(self,x,y,query):
        x=x.reset_index(drop=True);query=query.reset_index(drop=True)
        self.ref=x;self.proxy_builder=ProxyBuilder(self.cfg['proxy']).fit(x)
        self.clean_ref,_=self.proxy_builder.transform(x)
        self.impute={}
        for c,condition in [('f_103','f_113'),('f_175','f_104')]:
            self.impute[c]=(x.groupby(['f_141',condition])[c].mean(),x.groupby(condition)[c].mean(),float(x[c].mean()),condition)
        s=self._struct(x,self.clean_ref.copy());self.group_maps={}
        for g in GEO_COLS:
            cols=['clean_age','clean_storey','clean_area','clean_height','f_182','f_214','f_199','weak_masonry_score']
            self.group_maps[g]=(x[g].value_counts(),s.groupby(x[g])[cols].mean(),s[cols].mean())
        self.levels={c:sorted(x[c].fillna('MISSING').astype(str).unique()) for c in REAL_CAT_COLS}
        a=self.transform_base(x);b=self.transform_base(query)
        self.te_builder=NestedTE(self.cfg['te'],self.seed)
        qclean,_=self.proxy_builder.transform(query)
        ta,tb=self.te_builder.fit_transform(keys(x,self.clean_ref),y,keys(query,qclean))
        a=pd.concat([a,ta],axis=1);b=pd.concat([b,tb],axis=1)
        # Frozen Gemini core, raw sensors, categorical indicators: no new global label selection.
        extra=[c for c in a if c.startswith('onehot_') or c in PROXIES or c.startswith('unc_')]
        cols=list(dict.fromkeys(CORE+extra))
        if self.cfg['knn']!='none':
            k=int(self.cfg['knn'].split('k')[-1]);knn=ConditionalKNN(k,seed=self.seed)
            ka,kb=knn.fit_transform(x,y,query)
            a=pd.concat([a,ka],axis=1);b=pd.concat([b,kb],axis=1);cols+=ka.columns.tolist()
            self.knn_builder=knn
        mode=self.cfg['ablate']
        if mode=='one_hot':cols=[c for c in cols if not c.startswith('onehot_')]
        if mode=='te_interactions':cols=[c for c in cols if not(c.startswith('i') and '_te_' in c)]
        if mode=='all_te':cols=[c for c in cols if '_te_' not in c]
        if mode=='proxy_raw' or self.cfg['proxy']=='V1':cols=[c for c in cols if c not in PROXIES]
        if mode=='proxy_uncertainty':cols=[c for c in cols if not c.startswith('unc_')]
        if mode=='group_aggregate':cols=[c for c in cols if not any(s in c for s in ['_mean_','_diff_','_count'])]
        if mode=='binary_crosses':cols=[c for c in cols if c not in ['weak_masonry_score','superstructure_sum']]
        self.columns=cols
        return a[cols].replace([np.inf,-np.inf],np.nan),b[cols].replace([np.inf,-np.inf],np.nan)
    def transform_base(self,x):
        clean,unc=self.proxy_builder.transform(x);out=self._struct(x,clean.copy()).to_dict('series')
        for c in PROXIES:out[c]=x[c].to_numpy()
        for c in REAL_CAT_COLS:
            s=x[c].fillna('MISSING').astype(str)
            out[c+'_freq']=s.map(self.ref[c].fillna('MISSING').astype(str).value_counts(normalize=True)).fillna(0)
            for v in self.levels[c]:out['onehot_'+c+'_'+v]=(s==v).astype(float)
        for g,(count,mapping,fallback) in self.group_maps.items():
            out[g+'_count']=np.log1p(x[g].map(count).fillna(0))
            for c in mapping:
                mean=x[g].map(mapping[c]).fillna(fallback[c]);out[g+'_mean_'+c]=mean
                if c in ['clean_age','clean_height','clean_area','weak_masonry_score']:out[g+'_diff_'+c]=out[c]-mean
        if self.cfg['proxy']=='V3':
            for c in unc:out['unc_'+c]=unc[c].to_numpy()
        return pd.DataFrame(out,index=x.index).reset_index(drop=True)

def model_predict(a,y,b,kind='spline',weights=None,seed=42):
    if kind=='blend':
        return .5*model_predict(a,y,b,'spline',weights,seed)+.5*model_predict(a,y,b,'hgb',weights,seed)
    if kind=='hgb':
        clf=HistGradientBoostingClassifier(max_iter=160,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=35,
                                           l2_regularization=20,early_stopping=False,random_state=seed)
        clf.fit(a,y,sample_weight=weights);return clf.predict_proba(b)
    raw=a.fillna(0).to_numpy(float);test=b.fillna(0).to_numpy(float)
    med=np.median(raw,axis=0);iqr=np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0)
    scale=np.where(iqr>1e-5,iqr,np.std(raw,axis=0));scale=np.where(scale>1e-5,scale,1)
    za=np.clip((raw-med)/scale,-5,5);zb=np.clip((test-med)/scale,-5,5)
    aa=np.hstack([za,.15*za**2]);bb=np.hstack([zb,.15*zb**2])
    if kind=='ordinal':
        probs=[]
        for k in [0,1]:
            clf=LogisticRegression(C=1/35,max_iter=700,tol=1e-4).fit(aa,(y<=k).astype(int),sample_weight=weights)
            probs.append(clf.predict_proba(bb)[:,1])
        low,high=np.minimum(probs[0],probs[1]),np.maximum(probs[0],probs[1])
        p=np.column_stack([low,high-low,1-high]);return p/p.sum(1,keepdims=True)
    clf=LogisticRegression(C=1/35,max_iter=700,tol=1e-4).fit(aa,y,sample_weight=weights)
    if max(clf.n_iter_)>=700:raise RuntimeError('Multinomial optimizer did not converge')
    return clf.predict_proba(bb)

def predict_pipeline(x,y,query,cfg=None,seed=913,diagnostics=False):
    cfg={**BASE,**(cfg or {})};x=x.reset_index(drop=True);query=query.reset_index(drop=True);y=np.array(y)
    if cfg['strategy']=='fold_ensemble':
        from .common import cf_splits
        parts=[]
        for t,_ in cf_splits(len(y),seed):
            p,_=predict_pipeline(x.iloc[t],y[t],query,{**cfg,'strategy':'full'},seed)
            parts.append(p)
        return np.mean(parts,axis=0),None
    feature_cfg={k:cfg[k] for k in ['proxy','te','knn','ablate']}
    key=hashlib.sha256(pd.util.hash_pandas_object(x,index=False).values.tobytes()+np.asarray(y).tobytes()+
                       pd.util.hash_pandas_object(query,index=False).values.tobytes()+json.dumps(feature_cfg,sort_keys=True).encode()+str(seed).encode()).hexdigest()
    if key in FEATURE_CACHE:a,b,factory=FEATURE_CACHE[key]
    else:
        factory=FeatureFactory(cfg,seed);a,b=factory.fit_transform(x,y,query)
        if len(FEATURE_CACHE)>=12:FEATURE_CACHE.pop(next(iter(FEATURE_CACHE)))
        FEATURE_CACHE[key]=(a,b,factory)
    weights=None
    if cfg['weights']=='soft' or cfg['residual']:
        from .common import cf_splits
        inner=np.zeros((len(y),3))
        for t,v in cf_splits(len(y),seed,2):
            inner[v],_=predict_pipeline(x.iloc[t],y[t],x.iloc[v],{**cfg,'weights':'none','residual':False},seed)
        if cfg['weights']=='soft':
            weights=np.ones(len(y));risk=inner[np.arange(len(y)),y]
            candidates=np.flatnonzero(risk<.1);candidates=candidates[np.argsort(risk[candidates])[:int(.05*len(y))]];weights[candidates]=.8
    p=model_predict(a,y,b,cfg['model'],weights,seed)
    p=prior_correct(p,np.bincount(y,minlength=3)/len(y))
    if cfg['residual']:
        # Inputs and residual targets are cross-fitted inside the allowed fit set.
        clean,_=factory.proxy_builder.transform(x);qc,_=factory.proxy_builder.transform(query)
        cols=['clean_age','clean_storey','clean_area','clean_height']
        tx=np.hstack([clean[cols].fillna(0),inner]);qx=np.hstack([qc[cols].fillna(0),p])
        reg=HistGradientBoostingRegressor(max_iter=80,max_leaf_nodes=7,l2_regularization=50,early_stopping=False,random_state=seed)
        reg.fit(tx,(y==1)-inner[:,1]);r=reg.predict(qx)
        old=p[:,1].copy();p[:,1]=np.clip(p[:,1]+.25*r,1e-5,1-1e-5)
        p[:,[0,2]]*=((1-p[:,1])/np.maximum(1-old,1e-8))[:,None];p/=p.sum(1,keepdims=True)
    return p,factory
