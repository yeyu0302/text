"""One latent severity and positive cutpoint gap: not two independent logits."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit,logsumexp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight
from . import run as joint
from research_0680.src.common import read_data,dump,manifest,full_metrics,bootstrap

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE5_ORDINAL_RESULTS',str(HERE/'ordinal_results')))
PROTOCOL=json.loads((HERE/'ordinal_protocol.json').read_text())

def logprob(eta,width):
    d=.05+np.logaddexp(0,width);lo=-d/2-eta;hi=d/2-eta
    return np.column_stack([-np.logaddexp(0,-lo),
        -np.logaddexp(0,-hi)-np.logaddexp(0,lo)+np.log(-np.expm1(-d)),
        -np.logaddexp(0,hi)])

def objective(theta,x,z,y,w,penalty):
    n=x.shape[1];beta=theta[:n];gamma=theta[n:];eta=x@beta;width=z@gamma
    d=.05+np.logaddexp(0,width);a=expit(-d/2-eta);b=expit(d/2-eta)
    ge=np.where(y==0,1-a,np.where(y==1,1-a-b,-b))
    gd=np.where(y==0,.5*(1-a),np.where(y==1,-.5*(1-b+a)-1/np.expm1(np.minimum(d,50)),.5*b))*expit(width)
    pb=beta.copy();pb[-1]=0;pg=gamma.copy();pg[0]=0
    loss=-w@logprob(eta,width)[np.arange(len(y)),y]+.5*penalty*(pb@pb+pg@pg)
    grad=np.r_[x.T@(w*ge)+penalty*pb,z.T@(w*gd)+penalty*pg]
    return float(loss),grad

def matrices(x,q):
    a,b=joint.designs(x,q)['joint_intercepts']
    a=sparse.hstack([a,np.ones((len(x),1))],format='csr');b=sparse.hstack([b,np.ones((len(q),1))],format='csr')
    encoder=OneHotEncoder(handle_unknown='ignore').fit(x[['f_90']].astype(str))
    za=sparse.csr_matrix(np.ones((len(x),1)));zb=sparse.csr_matrix(np.ones((len(q),1)))
    return a,b,dict(ordinal_shared_width=(za,zb),ordinal_macro_width=(sparse.hstack([za,encoder.transform(x[['f_90']].astype(str))],format='csr'),
                                                                 sparse.hstack([zb,encoder.transform(q[['f_90']].astype(str))],format='csr')))

def predictions(x,y,q):
    a,b,widths=matrices(x,q);result={};diagnostics=[]
    for name,(za,zb) in widths.items():
        initial=np.zeros(a.shape[1]+za.shape[1]);initial[a.shape[1]]=np.log(np.expm1(2*np.log(2)-.05))
        fit=minimize(objective,initial,args=(a,za,y,compute_sample_weight('balanced',y)/len(y),20/len(y)),
                     jac=True,method='L-BFGS-B',options=PROTOCOL['optimizer'])
        assert np.isfinite(fit.fun) and np.isfinite(fit.x).all()
        lp=logprob(b@fit.x[:a.shape[1]],zb@fit.x[a.shape[1]:]);p=np.exp(lp-logsumexp(lp,axis=1,keepdims=True))
        assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
        result[name]=p;diagnostics.append(dict(model=name,success=bool(fit.success),iterations=int(fit.nit),loss=float(fit.fun),max_gradient=float(np.max(abs(fit.jac)))))
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/f'optimization_{joint.digest(x.id)[:12]}.json',diagnostics)
    return result

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);x,y,_,_=read_data();t,v=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=847)
    old=np.load(joint.INNER_REFERENCE/'screen/probabilities.npz');assert np.array_equal(old['ids'],x.id.iloc[v]);base=old['reference']
    start=time.perf_counter();pred=predictions(x.iloc[t],y[t],x.iloc[v]);rows=[]
    np.savez_compressed(OUT/'screen_probabilities.npz',ids=x.id.iloc[v],**pred)
    for name,p in pred.items():rows.append(dict(model=name,scope='screen_3750',**full_metrics(y[v],p),**bootstrap(y[v],p,base)))
    pd.DataFrame(rows).to_csv(OUT/'screen.csv',index=False)
    dump(OUT/'manifest.json',manifest(dict(protocol=PROTOCOL,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_seconds=time.perf_counter()-start)))
    promoted=max(r['delta'] for r in rows)>.002
    dump(OUT/'gate.json',dict(promoted=promoted,reason='Proceed only if new screening delta > .002; no failed old models rerun'))
    print('ORDINAL SCREEN',[(r['model'],r['balanced_accuracy'],r['delta']) for r in rows],flush=True)
