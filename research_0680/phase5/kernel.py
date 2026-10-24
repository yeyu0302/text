"""Two predeclared low-rank RBF classifiers, not another old-model sweep."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from research_0680.src.train_baseline import FeatureFactory,BASE,CORE
from research_0680.src.common import read_data,dump,manifest,full_metrics,bootstrap
from .run import INNER_REFERENCE

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE5_KERNEL_RESULTS',str(HERE/'kernel_results')))
PROTOCOL=json.loads((HERE/'kernel_protocol.json').read_text())

def matrices(x,y,q):
    a,b=FeatureFactory(BASE).fit_transform(x,y,q)
    columns=CORE+[c for c in a if c.startswith('onehot_')]
    aa=a[columns].fillna(0).to_numpy(float);bb=b[columns].fillna(0).to_numpy(float)
    med=np.median(aa,axis=0);scale=np.quantile(aa,.75,axis=0)-np.quantile(aa,.25,axis=0)
    scale=np.where(scale>1e-5,scale,np.std(aa,axis=0));scale=np.where(scale>1e-5,scale,1)
    return np.clip((aa-med)/scale,-5,5),np.clip((bb-med)/scale,-5,5)

def predictions(x,y,q):
    a,b=matrices(x,y,q);result={};diagnostics=[]
    for name,width in PROTOCOL['models'].items():
        tick=time.perf_counter()
        kernel=Nystroem(kernel='rbf',gamma=1/(width*a.shape[1]),n_components=min(384,len(x)),random_state=913)
        ka=kernel.fit_transform(a);kb=kernel.transform(b)
        aa=np.hstack([.1*a,ka]);bb=np.hstack([.1*b,kb])
        model=LogisticRegression(C=1.,class_weight='balanced',max_iter=600,tol=1e-4).fit(aa,y)
        assert max(model.n_iter_)<600
        result[name]=model.predict_proba(bb)
        diagnostics.append(dict(model=name,iterations=model.n_iter_.tolist(),features=aa.shape[1],runtime_seconds=time.perf_counter()-tick))
    OUT.mkdir(parents=True,exist_ok=True);dump(OUT/'fit_diagnostics.json',diagnostics)
    return result

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);x,y,_,_=read_data();t,v=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=847)
    old=np.load(INNER_REFERENCE/'screen/probabilities.npz');assert np.array_equal(old['ids'],x.id.iloc[v])
    tick=time.perf_counter();pred=predictions(x.iloc[t],y[t],x.iloc[v]);rows=[]
    np.savez_compressed(OUT/'screen_probabilities.npz',ids=x.id.iloc[v],**pred)
    for name,p in pred.items():rows.append(dict(model=name,scope='screen_3750',**full_metrics(y[v],p),**bootstrap(y[v],p,old['reference'])))
    pd.DataFrame(rows).to_csv(OUT/'screen.csv',index=False)
    dump(OUT/'manifest.json',manifest(dict(protocol=PROTOCOL,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_seconds=time.perf_counter()-tick)))
    promoted=max(r['delta'] for r in rows)>.002
    dump(OUT/'gate.json',dict(promoted=promoted,reason='Stop unless new screening delta > .002; no old-model fitting'))
    print('KERNEL SCREEN',[(r['model'],r['balanced_accuracy'],r['delta']) for r in rows],flush=True)
