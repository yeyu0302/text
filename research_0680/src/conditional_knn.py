"""Mixed-feature distances, categorical group IDs, complete label cross-fitting."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler,OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from src.features import REAL_CAT_COLS,REAL_BIN_COLS
from .build_proxy_features import ProxyBuilder
from .common import cf_splits

class ConditionalKNN:
    def __init__(self,k=20,scope='region',bandwidth=1.,seed=913):
        self.k,self.scope,self.bandwidth,self.seed=k,scope,bandwidth,seed
    def fit(self,x,y):
        self.x=x.reset_index(drop=True).copy();self.y=np.asarray(y).copy();self.ids=self.x.id.to_numpy()
        self.proxy=ProxyBuilder('V0').fit(self.x);a,_=self.proxy.transform(self.x)
        self.numeric=['clean_age','clean_storey','clean_area','clean_height']
        self.scaler=RobustScaler().fit(a[self.numeric]);self.num_median=a[self.numeric].median().to_numpy()
        self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(self.x[REAL_CAT_COLS].fillna('MISSING').astype(str))
        self.matrix=self.matrix_for(self.x)
        self.prior=np.bincount(y,minlength=3)/len(y)
        self.id_to_label=dict(zip(self.ids,self.y))
        self.indices={g:{str(value):np.asarray(idx) for value,idx in self.x.groupby(g).indices.items()} for g in ['f_141','f_109','f_90']}
        self.cache={}
        return self
    def matrix_for(self,x):
        a,_=self.proxy.transform(x.reset_index(drop=True))
        num=self.scaler.transform(a[self.numeric].fillna(pd.Series(self.num_median,index=self.numeric)))
        cat=self.encoder.transform(x[REAL_CAT_COLS].fillna('MISSING').astype(str))*.7
        binary=x[REAL_BIN_COLS].fillna(.5).to_numpy(float)*.35
        return np.hstack([np.clip(num,-10,10),cat,binary]).astype(np.float64)
    def transform(self,x):
        x=x.reset_index(drop=True);q=self.matrix_for(x)
        buckets={}
        for i,row in x.iterrows():
            if self.scope=='global':key=('global','all')
            else:
                levels=['f_141','f_109','f_90'] if self.scope=='region' else ([self.scope,'f_90'] if self.scope!='f_90' else ['f_90'])
                key=('global','all')
                for g in levels:
                    value=str(row[g]);pool=self.indices[g].get(value,np.array([],int))
                    if len(pool)>=self.k:key=(g,value);break
            buckets.setdefault(key,[]).append(i)
        rows=np.zeros((len(x),8));neighbor_ids=np.full((len(x),self.k),-1,dtype=np.int64)
        fallbacks=np.zeros(len(x),int);nearest=np.zeros(len(x))
        for key,where in buckets.items():
            pool=np.arange(len(self.x)) if key[0]=='global' else self.indices[key[0]][key[1]]
            n=min(self.k+1,len(pool))
            model=NearestNeighbors(n_neighbors=n,algorithm='brute',n_jobs=1).fit(self.matrix[pool])
            distance,index=model.kneighbors(q[where]);index=pool[index]
            for loc,i in enumerate(where):
                keep=self.ids[index[loc]]!=x.id.iloc[i]
                nb=index[loc][keep][:self.k];dist=distance[loc][keep][:self.k]
                assert not np.any(self.ids[nb]==x.id.iloc[i])
                w=np.exp(-np.minimum(dist/self.bandwidth,30));total=w.sum();effective=total**2/(np.sum(w*w)+1e-12)
                prior=self.prior
                own=self.id_to_label.get(x.id.iloc[i])
                if own is not None:
                    counts=np.bincount(self.y,minlength=3).astype(float);counts[own]-=1
                    prior=counts/max(1,len(self.y)-1)
                prob=(np.bincount(self.y[nb],weights=w,minlength=3)+3*prior)/(total+3)
                entropy=-np.sum(prob*np.log(np.clip(prob,1e-12,1)))
                rows[i]=[*prob,entropy,effective,float(dist.mean()) if len(dist) else 0.,total,float(len(nb))]
                neighbor_ids[i,:len(nb)]=self.ids[nb]
                nearest[i]=float(dist[0]) if len(dist) else np.nan
                fallbacks[i]=int(key[0]!='f_141')
        self.neighbor_ids=neighbor_ids;self.fallbacks=fallbacks;self.nearest_distances=nearest
        return pd.DataFrame(rows,columns=['knn_p1','knn_p2','knn_p3','neighbor_entropy','neighbor_effective_n','knn_distance','knn_weight_sum','neighbor_n'])
    def fit_transform(self,x,y,query):
        parts=[];self.crossfit_audit=[]
        for t,v in cf_splits(len(x),self.seed):
            model=ConditionalKNN(self.k,self.scope,self.bandwidth,self.seed).fit(x.iloc[t],np.asarray(y)[t])
            p=model.transform(x.iloc[v]);p.index=v;parts.append(p)
            allowed=set(x.id.iloc[t]);assert all(int(i) in allowed for i in model.neighbor_ids.ravel() if i!=-1)
            assert not set(x.id.iloc[v])&allowed
            self.crossfit_audit.append(dict(fit_ids=x.id.iloc[t].tolist(),query_ids=x.id.iloc[v].tolist()))
        self.fit(x,y)
        return pd.concat(parts).sort_index(),self.transform(query)
