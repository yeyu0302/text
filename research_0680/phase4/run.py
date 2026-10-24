"""Expanded feature control, piecewise linear model and small weighted MLP."""
import hashlib,json,os,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.utils.class_weight import compute_sample_weight
from research_0680.phase3 import run as engine
from research_0680.src.train_baseline import FeatureFactory,BASE,model_predict
from research_0680.src.common import read_data,dump,manifest,outer_folds,full_metrics,bootstrap,save_run
from src.features import keys
from src.evaluation import prior_correct

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE4_RESULTS',str(HERE/'results')))
PROTOCOL=json.loads((HERE/'protocol.json').read_text())

def features(x,y,q):
    x=x.reset_index(drop=True);q=q.reset_index(drop=True)
    builder=FeatureFactory(BASE);ra,rb=builder.fit_transform(x,y,q)
    a=builder.transform_base(x);b=builder.transform_base(q)
    clean,_=builder.proxy_builder.transform(q)
    ta,tb=builder.te_builder.fit_transform(builder.te_builder.fit_keys,y,keys(q,clean))
    return ra,rb,pd.concat([a,ta],axis=1),pd.concat([b,tb],axis=1)

def predictions(x,y,q):
    ra,rb,a,b=features(x,y,q);prior=np.bincount(y,minlength=3)/len(y)
    result={'reference':prior_correct(model_predict(ra,y,rb),prior),
            'expanded_quad':prior_correct(model_predict(a,y,b),prior)}
    aa=a.fillna(0).to_numpy(float);bb=b.fillna(0).to_numpy(float)
    med=np.median(aa,axis=0);iqr=np.quantile(aa,.75,axis=0)-np.quantile(aa,.25,axis=0)
    scale=np.where(iqr>1e-5,iqr,np.std(aa,axis=0));scale=np.where(scale>1e-5,scale,1)
    aa=np.clip((aa-med)/scale,-5,5);bb=np.clip((bb-med)/scale,-5,5)
    continuous=[j for j in range(a.shape[1]) if a.iloc[:,j].nunique()>10]
    # Compact hinge representation: only nonbinary fields gain nonlinear terms.
    ha=np.hstack([aa]+[np.maximum(aa[:,continuous]-k,0) for k in [-1,-.5,0,1]])
    hb=np.hstack([bb]+[np.maximum(bb[:,continuous]-k,0) for k in [-1,-.5,0,1]])
    clf=LogisticRegression(C=1/35,max_iter=800,tol=1e-4).fit(ha,y)
    assert max(clf.n_iter_)<800
    result['hinge']=prior_correct(clf.predict_proba(hb),prior)
    clf=MLPClassifier(**PROTOCOL['mlp'],random_state=913,early_stopping=False,tol=1e-4,n_iter_no_change=12)
    # Already optimizes the balanced target distribution; no second prior correction.
    clf.fit(aa,y,sample_weight=compute_sample_weight('balanced',y))
    result['mlp']=clf.predict_proba(bb)
    for choice in PROTOCOL['choices']:
        if '+' in choice:
            left,right=choice.split('+');result[choice]=.5*(result[left]+result[right])
    for p in result.values():assert np.isfinite(p).all() and np.allclose(p.sum(1),1)
    return result

def setup():
    engine.OUT=OUT;engine.PROTOCOL=PROTOCOL.copy();engine.PROTOCOL['backend_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine.CHOICES=PROTOCOL['choices'];engine.predictions=predictions

def final_export(rows):
    """Keep the statistical recommendation separate from explicit experimental delivery."""
    path=OUT/'submission_candidate_phase4_experimental.csv'
    if path.exists():return
    x,y,test,sample=read_data();choice,trace=engine.choose(x,y)
    # Full retrain: features, cross-fit training TE and all test transforms share train scope.
    p=predictions(x,y,test)[choice];sample['label']=p.argmax(1)+1
    assert sample.id.equals(test.id) and sample.label.isin([1,2,3]).all()
    sample.to_csv(path,index=False)
    pd.DataFrame(dict(id=test.id,p1=p[:,0],p2=p[:,1],p3=p[:,2])).to_csv(OUT/'candidate_test_probabilities.csv',index=False)
    gate=all(r['delta']>.001 and r['ci_low']>0 for r in rows)
    dump(OUT/'candidate_status.json',dict(choice=choice,trace=trace,statistical_gate_passed=gate,
        status='validated local candidate, Kaggle unmeasured' if gate else 'experimental, improvement statistically unconfirmed',
        measured_kaggle_score=None,submission_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        note='Explicit latest user request for a submit-ready CSV; existing .679 submission unchanged',protocol=PROTOCOL))
    print('EXPORTED',path,'choice',choice,'statistical gate',gate,flush=True)

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);setup();engine.screen()
    # Evaluate the frozen selection rule even if exploratory gain is weak: a
    # bounded final research round with an explicitly requested CSV deliverable.
    rows=[engine.outer(42),engine.outer(2026)]
    pd.DataFrame(rows).to_csv(OUT/'nested_comparison.csv',index=False)
    final_export(rows)
