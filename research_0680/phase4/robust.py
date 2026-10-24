"""Bounded losses on the frozen design, anchored to reference coefficients."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from research_0680.phase3 import run as engine

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE4_ROBUST_RESULTS',str(HERE/'robust_results')))
PROTOCOL=json.loads((HERE/'robust_protocol.json').read_text())

def objective(flat,x,y,w,anchor,strength,kind):
    theta=flat.reshape(anchor.shape);p=softmax(x@theta,axis=1);one=np.eye(3)[y]
    if kind=='brier':
        residual=p-one;loss=.5*(residual**2).sum(1)
        grad=p*(residual-(p*residual).sum(1,keepdims=True))
    else:
        true=np.clip(p[np.arange(len(y)),y],1e-12,1)
        loss=np.where(true>=.1,-np.log(true),-np.log(.1)+1-true/.1)
        grad=(p-one)*np.minimum(1,true/.1)[:,None]
    delta=theta-anchor;delta[-1]=0
    return float(w@loss+.5*strength*(delta**2).sum()),(x.T@(w[:,None]*grad)+strength*delta).ravel()

def predictions(x,y,q):
    a,b=engine.FeatureFactory(engine.BASE).fit_transform(x,y,q)
    aa=a.fillna(0).to_numpy(float);bb=b.fillna(0).to_numpy(float)
    med=np.median(aa,axis=0);iqr=np.quantile(aa,.75,axis=0)-np.quantile(aa,.25,axis=0)
    scale=np.where(iqr>1e-5,iqr,np.std(aa,axis=0));scale=np.where(scale>1e-5,scale,1)
    za=np.clip((aa-med)/scale,-5,5);zb=np.clip((bb-med)/scale,-5,5)
    aa=np.hstack([za,.15*za**2]);bb=np.hstack([zb,.15*zb**2])
    clf=LogisticRegression(C=1/35,max_iter=700,tol=1e-4).fit(aa,y)
    assert max(clf.n_iter_)<700
    prior=np.bincount(y,minlength=3)/len(y)
    result={'reference':engine.prior_correct(clf.predict_proba(bb),prior)}
    aa=np.column_stack([aa,np.ones(len(aa))]);bb=np.column_stack([bb,np.ones(len(bb))])
    anchor=np.vstack([clf.coef_.T,clf.intercept_-np.log(prior)])
    weights=compute_sample_weight('balanced',y)/len(y)
    for kind in ['brier','capped_ce']:
        fit=minimize(objective,anchor.ravel(),args=(aa,y,weights,anchor,35/len(y),kind),jac=True,
                     method='L-BFGS-B',options=dict(maxiter=60,ftol=1e-8,gtol=1e-5,maxcor=8))
        assert np.isfinite(fit.fun) and np.isfinite(fit.x).all()
        result[kind]=softmax(bb@fit.x.reshape(anchor.shape),axis=1)
    for name in PROTOCOL['choices']:
        if '+' in name:
            left,right=name.split('+');result[name]=.5*(result[left]+result[right])
    return result

def setup():
    engine.OUT=OUT;engine.PROTOCOL=PROTOCOL.copy();engine.PROTOCOL['backend_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine.CHOICES=PROTOCOL['choices'];engine.predictions=predictions

if __name__=='__main__':
    import pandas as pd
    OUT.mkdir(parents=True,exist_ok=True);setup();engine.screen()
    rows=[engine.outer(42),engine.outer(2026)]
    pd.DataFrame(rows).to_csv(OUT/'nested_comparison.csv',index=False)
    engine.dump(OUT/'gate.json',dict(passed=all(r['delta']>.001 and r['ci_low']>0 for r in rows),protocol=PROTOCOL))
