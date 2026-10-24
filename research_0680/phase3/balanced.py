"""Cheap alternative loss; reuse the exact same nested-validation runner."""
import hashlib,json,os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler,SplineTransformer
from . import run as engine

def predictions(x,y,q):
    a,b=engine.FeatureFactory(engine.BASE).fit_transform(x,y,q)
    result={'reference':engine.prior_correct(engine.model_predict(a,y,b),np.bincount(y,minlength=3)/len(y))}
    aa=a.fillna(0).to_numpy(float);bb=b.fillna(0).to_numpy(float)
    med=np.median(aa,axis=0);iqr=np.quantile(aa,.75,axis=0)-np.quantile(aa,.25,axis=0)
    scale=np.where(iqr>1e-5,iqr,np.std(aa,axis=0));scale=np.where(scale>1e-5,scale,1)
    za=np.clip((aa-med)/scale,-5,5);zb=np.clip((bb-med)/scale,-5,5)
    model=LogisticRegression(C=1/35,class_weight='balanced',max_iter=700,tol=1e-4)
    model.fit(np.hstack([za,.15*za**2]),y)
    assert max(model.n_iter_)<700
    # Fitted on balanced loss already: do not divide class priors a second time.
    result['balanced_quad']=model.predict_proba(np.hstack([zb,.15*zb**2]))
    columns=engine.CORE+[c for c in a if c.startswith('onehot_')]
    aa=a[columns].fillna(0).to_numpy(float);bb=b[columns].fillna(0).to_numpy(float)
    scale=StandardScaler().fit(aa);aa=np.clip(scale.transform(aa),-8,8);bb=np.clip(scale.transform(bb),-8,8)
    continuous=[j for j in range(aa.shape[1]) if len(np.unique(aa[:,j]))>=12]
    spline=SplineTransformer(n_knots=4,degree=2,knots='quantile',include_bias=False,extrapolation='linear')
    sa=np.hstack([aa,spline.fit_transform(aa[:,continuous])]);sb=np.hstack([bb,spline.transform(bb[:,continuous])])
    model=LogisticRegression(C=.1,class_weight='balanced',max_iter=700,tol=1e-4).fit(sa,y)
    assert max(model.n_iter_)<700
    result['balanced_gam']=model.predict_proba(sb)
    for name in engine.CHOICES:
        if '+' in name:
            left,right=name.split('+');result[name]=.5*(result[left]+result[right])
    for p in result.values():assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
    return result

def run():
    engine.PROTOCOL=json.loads((engine.HERE/'balanced_protocol.json').read_text())
    engine.PROTOCOL['backend_source_sha256']=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
    engine.OUT=engine.Path(os.environ.get('RESEARCH0680_PHASE3_BALANCED_RESULTS',str(engine.HERE/'balanced_results')))
    engine.CHOICES=engine.PROTOCOL['choices'];engine.predictions=predictions
    engine.run()

if __name__=='__main__':run()
