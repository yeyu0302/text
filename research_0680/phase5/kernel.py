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
from . import run as runner
from sklearn.model_selection import StratifiedKFold

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

def predictions(x,y,q,models=None):
    a,b=matrices(x,y,q);result={};diagnostics=[]
    for name,width in PROTOCOL['models'].items():
        if models is not None and name not in models:continue
        tick=time.perf_counter()
        kernel=Nystroem(kernel='rbf',gamma=1/(width*a.shape[1]),n_components=min(384,len(x)),random_state=913)
        ka=kernel.fit_transform(a);kb=kernel.transform(b)
        aa=np.hstack([.1*a,ka]);bb=np.hstack([.1*b,kb])
        model=LogisticRegression(C=1.,class_weight='balanced',max_iter=600,tol=1e-4).fit(aa,y)
        assert max(model.n_iter_)<600
        result[name]=model.predict_proba(bb)
        diagnostics.append(dict(model=name,iterations=model.n_iter_.tolist(),features=aa.shape[1],runtime_seconds=time.perf_counter()-tick))
    folder=OUT/'cv_fit_diagnostics';folder.mkdir(parents=True,exist_ok=True)
    suffix='_'.join(models) if models is not None else 'both'
    path=folder/f'{runner.digest(x.id)[:12]}_{suffix}.json'
    if not path.exists():dump(path,diagnostics)
    return result

def validate():
    runner.OUT=OUT
    runner.PROTOCOL=dict(PROTOCOL,models=list(PROTOCOL['models']),bandwidths=PROTOCOL['models'],
        backend_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),screen_snapshot_commit='6621ae5',
        cv_rule='Stratified inner2 seed391; choose bandwidth only within outer train, retain cached reference unless delta > .001')
    runner.predict_pair=predictions
    runner.predict_selected=lambda x,y,q,name:predictions(x,y,q,[name])[name]
    rows=[runner.outer(42)]
    if rows[0]['delta']>.001 and rows[0]['ci_low']>0:rows.append(runner.outer(2026))
    pd.DataFrame(rows).to_csv(OUT/'comparison.csv',index=False)
    passed=len(rows)==2 and all(r['delta']>.001 and r['ci_low']>0 for r in rows)
    dump(OUT/'validation_gate.json',dict(passed=passed,reason='Confirmation only after positive primary CI; no weak-route reruns'))
    if not passed:return
    x,y,test,sample=read_data();values={m:[] for m in PROTOCOL['models']}
    for t,v in StratifiedKFold(2,shuffle=True,random_state=391).split(x,y):
        p=predictions(x.iloc[t],y[t],x.iloc[v])
        for m in values:values[m].append(full_metrics(y[v],p[m])['balanced_accuracy'])
    old_path=HERE.parent/'phase4/results/candidate_status.json'
    old=json.loads(old_path.read_text());baseline=old['trace']['means']['reference']
    means={m:float(np.mean(s)) for m,s in values.items()};choice=max(means,key=means.get)
    retained=means[choice]>baseline+.001
    dump(OUT/'full_selection.json',dict(choice=choice if retained else 'reference',means=means,fold_scores=values,
        reference=baseline,cached_reference_source=str(old_path),cached_reference_sha256=hashlib.sha256(old_path.read_bytes()).hexdigest()))
    if not retained:return
    p=predictions(x,y,test,[choice])[choice]
    path=OUT/'submission_candidate_phase5_kernel.csv';assert not path.exists(),'Preserve existing candidate'
    sample['label']=p.argmax(1)+1;sample.to_csv(path,index=False)
    pd.DataFrame(dict(id=test.id,p1=p[:,0],p2=p[:,1],p3=p[:,2])).to_csv(OUT/'candidate_probabilities.csv',index=False)
    print('VALIDATED KERNEL CANDIDATE',choice,path,flush=True)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);x,y,_,_=read_data();t,v=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=847)
    old=np.load(INNER_REFERENCE/'screen/probabilities.npz');assert np.array_equal(old['ids'],x.id.iloc[v])
    if (OUT/'screen.csv').exists():
        assert json.loads((OUT/'manifest.json').read_text())['protocol']==PROTOCOL
        if bool(json.loads((OUT/'gate.json').read_text())['promoted']):validate()
        raise SystemExit(0)
    tick=time.perf_counter();pred=predictions(x.iloc[t],y[t],x.iloc[v]);rows=[]
    np.savez_compressed(OUT/'screen_probabilities.npz',ids=x.id.iloc[v],**pred)
    for name,p in pred.items():rows.append(dict(model=name,scope='screen_3750',**full_metrics(y[v],p),**bootstrap(y[v],p,old['reference'])))
    pd.DataFrame(rows).to_csv(OUT/'screen.csv',index=False)
    dump(OUT/'manifest.json',manifest(dict(protocol=PROTOCOL,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_seconds=time.perf_counter()-tick)))
    promoted=max(r['delta'] for r in rows)>.002
    dump(OUT/'gate.json',dict(promoted=promoted,reason='Stop unless new screening delta > .002; no old-model fitting'))
    print('KERNEL SCREEN',[(r['model'],r['balanced_accuracy'],r['delta']) for r in rows],flush=True)
