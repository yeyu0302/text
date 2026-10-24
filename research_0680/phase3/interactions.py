"""Forty-eight physics/region interactions, sharing the nested runner."""
import hashlib,json,os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from . import run as engine

def predictions(x,y,q):
    a,b=engine.FeatureFactory(engine.BASE).fit_transform(x,y,q)
    prior=np.bincount(y,minlength=3)/len(y)
    result={'reference':engine.prior_correct(engine.model_predict(a,y,b),prior)}
    aa=a.fillna(0).to_numpy(float);bb=b.fillna(0).to_numpy(float)
    med=np.median(aa,axis=0);iqr=np.quantile(aa,.75,axis=0)-np.quantile(aa,.25,axis=0)
    scale=np.where(iqr>1e-5,iqr,np.std(aa,axis=0));scale=np.where(scale>1e-5,scale,1)
    aa=np.clip((aa-med)/scale,-5,5);bb=np.clip((bb-med)/scale,-5,5)
    def geography(table):
        columns=[]
        for prefix in ['f90','f109','f141_m10']:
            p=table[[prefix+'_te_c'+str(k) for k in range(3)]].to_numpy()
            columns.extend([np.log((p[:,2]+.01)/(p[:,0]+.01)),np.log((p[:,1]+.01)/(p[:,0]+p[:,2]+.01))])
        return np.array(columns).T
    scaler=StandardScaler().fit(geography(a));ga=np.clip(scaler.transform(geography(a)),-5,5);gb=np.clip(scaler.transform(geography(b)),-5,5)
    phys=engine.PROTOCOL['physics'];scaler=StandardScaler().fit(a[phys].fillna(0))
    pa=np.clip(scaler.transform(a[phys].fillna(0)),-5,5);pb=np.clip(scaler.transform(b[phys].fillna(0)),-5,5)
    ia=(ga[:,:,None]*pa[:,None,:]).reshape(len(a),-1);ib=(gb[:,:,None]*pb[:,None,:]).reshape(len(b),-1)
    aa=np.hstack([aa,.15*aa**2,.25*ia]);bb=np.hstack([bb,.15*bb**2,.25*ib])
    for name,weight in [('interaction',None),('balanced_interaction','balanced')]:
        model=LogisticRegression(C=1/35,class_weight=weight,max_iter=700,tol=1e-4).fit(aa,y)
        assert max(model.n_iter_)<700
        p=model.predict_proba(bb);result[name]=engine.prior_correct(p,prior) if weight is None else p
    for name in engine.CHOICES:
        if '+' in name:
            left,right=name.split('+');result[name]=.5*(result[left]+result[right])
    for p in result.values():assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
    return result

if __name__=='__main__':
    engine.PROTOCOL=json.loads((engine.HERE/'interaction_protocol.json').read_text())
    engine.PROTOCOL['backend_source_sha256']=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
    engine.OUT=engine.Path(os.environ.get('RESEARCH0680_PHASE3_INTERACTION_RESULTS',str(engine.HERE/'interaction_results')))
    engine.CHOICES=engine.PROTOCOL['choices'];engine.predictions=predictions;engine.run()
