"""Implementation repairs; no claims of BLUP, label purity, or Kaggle gains."""
import numpy as np
import pandas as pd

TE_PREFIXES=('te_f90','te_f109','te_f141_m10')

class SandwichFeatures:
    """Use producer names and fit normalization on training rows only."""
    @staticmethod
    def raw(base,te):
        required=['clean_age','clean_storey','f_182','f_214']
        if not set(required)<=set(base):raise ValueError('Missing named physical features')
        columns=[f'{g}_bal_p{k}' for g in TE_PREFIXES for k in [1,3]]
        if not set(columns)<=set(te):raise ValueError('Missing expected balanced geographic TE columns')
        a=base.clean_age.to_numpy();s=base.clean_storey.to_numpy()
        v=.45*np.log1p(np.clip(a,0,200))+.35*np.log1p(np.clip(s,1,20))-.40*base.f_182.to_numpy()-.30*base.f_214.to_numpy()
        h=np.mean([te[f'{g}_bal_p3'].to_numpy()-te[f'{g}_bal_p1'].to_numpy() for g in TE_PREFIXES],axis=0)
        return np.column_stack([v,h])
    def fit(self,base,te):
        a=self.raw(base,te)
        if not np.isfinite(a).all():raise ValueError('Non-finite fit inputs')
        self.mean_=a.mean(0);self.scale_=np.maximum(a.std(0),1e-6)
        return self
    def transform(self,base,te):
        a=self.raw(base,te)
        if not np.isfinite(a).all():raise ValueError('Non-finite query inputs')
        v,h=((a-self.mean_)/self.scale_).T
        return np.column_stack([np.maximum(h,0)*np.maximum(-v,0),np.maximum(-h,0)*np.maximum(v,0),abs(h-v),h-v,np.exp(-.5*(h+v)**2)])


def combine_plans(oof_a,oof_b,test_a,test_b,weight_b=.70):
    """Arithmetic guard only: actual B-OOF must come from outer-local refits."""
    if oof_b is None:raise ValueError('Plan B requires independently generated outer-local OOF predictions')
    arrays=[np.asarray(p,float) for p in [oof_a,oof_b,test_a,test_b]]
    if not 0<=weight_b<=1:raise ValueError('Invalid weight')
    for a in arrays:
        if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all() or np.any(a<0) or not np.allclose(a.sum(1),1):raise ValueError('Invalid probabilities')
    a,b,t,u=arrays
    if a.shape!=b.shape or t.shape!=u.shape:raise ValueError('Plan alignment mismatch')
    return (1-weight_b)*a+weight_b*b,(1-weight_b)*t+weight_b*u
