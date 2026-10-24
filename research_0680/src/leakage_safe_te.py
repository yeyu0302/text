import numpy as np
import pandas as pd
from src.features import keys,encode,TE_SPECS
from .common import cf_splits

class NestedTE:
    def __init__(self,mode='plain',seed=913):self.mode,self.seed=mode,seed
    def one(self,fit,y,query):
        # Same scalar source definitions as previous code, but priors always fit-local.
        result=encode(fit,y,query,1.,True)
        if self.mode!='plain':
            suffix=self.mode.replace('balanced_log','') if self.mode.startswith('balanced_log') else ''
            m=float(suffix) if suffix else None
            # Replace finest group posterior with explicit requested pseudo-count.
            # Exactly m pseudo-observations for each finest-group representation;
            # keep other group/interactions' supplied smoothing fixed.
            parent=pd.DataFrame(np.eye(3)[y]).groupby(fit['f_109'].to_numpy()).agg(['sum','count'])
            prior=np.bincount(y,minlength=3)/len(y)
            counts=parent[(0,'count')].to_numpy()[:,None]
            par=(np.column_stack([parent[(k,'sum')] for k in range(3)])+8*prior)/(counts+8)
            app=pd.DataFrame(par,index=parent.index).reindex(query['f_109']).fillna(pd.Series(prior)).to_numpy()
            child=pd.DataFrame(np.eye(3)[y]).groupby(fit['f_141'].to_numpy()).agg(['sum','count'])
            count=child[(0,'count')].reindex(query['f_141']).fillna(0).to_numpy()[:,None]
            total=np.column_stack([child[(k,'sum')].reindex(query['f_141']).fillna(0) for k in range(3)])
            if m is not None:
                smooth=(total+m*app)/(count+m)
                for prefix in ['f141_m3','f141_m10','f141_m25']:
                    for k in range(3):result[f'{prefix}_te_c{k}']=smooth[:,k]
            prior=np.bincount(y,minlength=3)/len(y)
            for _,_,_,prefix in TE_SPECS:
                cols=[f'{prefix}_te_c{k}' for k in range(3)]
                p=result[cols].to_numpy()/prior;p/=p.sum(1,keepdims=True)
                result[cols]=np.log(np.clip(p,1e-6,1)) if self.mode.startswith('balanced_log') else p
                result[f'{prefix}_te_expected']=p[:,1]+2*p[:,2]
        return result
    def fit_transform(self,fit,y,query):
        self.fit_keys=fit.reset_index(drop=True);self.y=np.array(y).copy()
        parts=[]
        for t,v in cf_splits(len(y),self.seed):
            p=self.one(self.fit_keys.iloc[t].reset_index(drop=True),self.y[t],self.fit_keys.iloc[v].reset_index(drop=True));p.index=v;parts.append(p)
        train=pd.concat(parts).sort_index();valid=self.one(self.fit_keys,self.y,query.reset_index(drop=True))
        self.distribution=pd.DataFrame(dict(feature=train.columns,train_mean=train.mean().values,train_variance=train.var().values,
                                            query_mean=valid.mean().values,query_variance=valid.var().values,
                                            query_nan=valid.isna().sum().values))
        return train,valid
