"""Repair only strong proxy/main contradictions; never change labels."""
import hashlib,json,os
import numpy as np
from scipy.special import softmax
from research_0680.phase2.src.methods import PosteriorProxy
from research_0680.src.build_proxy_features import GROUPS,trusted
from . import run as engine

def repair(builder,x):
    out=x.copy()
    for source,cols in GROUPS.items():
        grid,prior=builder.priors[source];bias,slope,var=builder.params[source]['params'].T
        observed=x[cols].to_numpy(float);sigma=np.sqrt(var)*abs(slope)+1e-6
        r=(observed[:,None,:]-(bias[None,:]+grid[:,None]*slope[None,:])[None,:,:])/sigma[None,None,:]
        ll=-2.5*np.log1p(r*r/4)-np.log(sigma[None,None,:])
        post=softmax(np.log(prior)[None,:]+np.where(np.isfinite(observed[:,None,:]),ll,0).sum(2),axis=1)
        raw=x[source].to_numpy(float);mean=post@grid;sd=np.sqrt(np.maximum(post@(grid**2)-mean**2,1e-6))
        closest=np.abs(raw[:,None]-grid).argmin(1);support=post[np.arange(len(x)),closest]
        correct=.95*support/(.95*support+.05/len(grid))
        flag=trusted(x,source).to_numpy()&(correct<.1)&(abs(raw-mean)>3*sd)
        out.loc[flag,source]=grid[post[flag].argmax(1)]
    return out

def predictions(x,y,q):
    from research_0680.src.train_baseline import predict_pipeline
    p=predict_pipeline(x,y,q,engine.BASE)[0]
    builder=PosteriorProxy().fit(x);xr=repair(builder,x);qr=repair(builder,q)
    pr=predict_pipeline(xr,y,qr,engine.BASE)[0]
    return {'reference':p,'repair':pr,'reference+repair':.5*(p+pr)}

if __name__=='__main__':
    engine.PROTOCOL=json.loads((engine.HERE/'repair_protocol.json').read_text())
    engine.PROTOCOL['backend_source_sha256']=hashlib.sha256(open(__file__,'rb').read()).hexdigest()
    engine.OUT=engine.Path(os.environ.get('RESEARCH0680_PHASE3_REPAIR_RESULTS',str(engine.HERE/'repair_results')))
    engine.CHOICES=engine.PROTOCOL['choices'];engine.predictions=predictions;engine.run()
