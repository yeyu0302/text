"""Outer-held-out silver-reference reconstruction, using proxies only as predictors."""
import numpy as np
import pandas as pd
from .common import *
from .build_proxy_features import ProxyBuilder,GROUPS,trusted

def run():
    x,y,_,_=read_data();folds=outer_folds(y,42);rows=[]
    names={'structure_age':'age_proxy_mean','storey_count':'storey_proxy_mean','footprint_area':'area_proxy_mean','f_165':'height_proxy_mean'}
    for f in range(5):
        fit=np.flatnonzero(folds!=f);val=np.flatnonzero(folds==f);q=x.iloc[val].reset_index(drop=True)
        for variant in ['V0','V1','V2','V4']:
            obj=ProxyBuilder(variant).fit(x.iloc[fit]);a,_=obj.transform(q)
            for source,col in names.items():
                good=trusted(q,source);residual=q.loc[good,source].to_numpy()-a.loc[good,col].to_numpy()
                rows.append(dict(fold=f,variant=variant,source=source,n=int(good.sum()),mae=float(abs(residual).mean()),
                    rmse=float(np.sqrt(np.mean(residual**2))),evaluation='Query main value only scores silver-reference proxy-only prediction; never fitted on query main values'))
    pd.DataFrame(rows).to_csv(RESULTS/'01_data_forensics/proxy_reconstruction_outer.csv',index=False)
    print('Silver reference reconstruction evaluated on outer-held-out feature values',flush=True)

if __name__=='__main__':run()
