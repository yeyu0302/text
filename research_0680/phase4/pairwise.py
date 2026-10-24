"""Two adjacent damage-boundary specialists; shared nested validation."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.special import softmax,logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from research_0680.phase3 import run as engine

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE4_PAIR_RESULTS',str(HERE/'pairwise_results')))
PROTOCOL=json.loads((HERE/'pairwise_protocol.json').read_text())

def predictions(x,y,q):
    a,b=engine.FeatureFactory(engine.BASE).fit_transform(x,y,q)
    prior=np.bincount(y,minlength=3)/len(y)
    result={'reference':engine.prior_correct(engine.model_predict(a,y,b),prior)}
    columns=engine.CORE+[c for c in a if c.startswith('onehot_')]
    aa=a[columns].fillna(0).to_numpy(float);bb=b[columns].fillna(0).to_numpy(float)
    med=np.median(aa,axis=0);iqr=np.quantile(aa,.75,axis=0)-np.quantile(aa,.25,axis=0)
    scale=np.where(iqr>1e-5,iqr,np.std(aa,axis=0));scale=np.where(scale>1e-5,scale,1)
    za=np.clip((aa-med)/scale,-5,5);zb=np.clip((bb-med)/scale,-5,5)
    qa=np.hstack([za,.15*za**2]);qb=np.hstack([zb,.15*zb**2])
    for name in ['pair_hist','pair_quad']:
        odds=[]
        for left,right in [(0,1),(1,2)]:
            mask=(y==left)|(y==right);target=(y[mask]==right).astype(int)
            if name=='pair_hist':
                model=HistGradientBoostingClassifier(**PROTOCOL['hist'],early_stopping=False,random_state=913)
                model.fit(aa[mask],target,sample_weight=compute_sample_weight('balanced',target));p=model.predict_proba(bb)[:,1]
            else:
                model=LogisticRegression(C=1/35,class_weight='balanced',max_iter=700,tol=1e-4).fit(qa[mask],target)
                assert max(model.n_iter_)<700;p=model.predict_proba(qb)[:,1]
            odds.append(logit(np.clip(p,1e-5,1-1e-5)))
        result[name]=softmax(np.column_stack([-odds[0],np.zeros(len(q)),odds[1]]),axis=1)
    for choice in PROTOCOL['choices']:
        if '+' in choice:
            left,right=choice.split('+');result[choice]=.5*(result[left]+result[right])
    return result

def setup():
    engine.OUT=OUT;engine.PROTOCOL=PROTOCOL.copy();engine.PROTOCOL['backend_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine.CHOICES=PROTOCOL['choices'];engine.predictions=predictions

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);setup();engine.screen()
    rows=[engine.outer(42),engine.outer(2026)]
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT/'nested_comparison.csv',index=False)
    dump=engine.dump;dump(OUT/'gate.json',dict(passed=all(r['delta']>.001 and r['ci_low']>0 for r in rows),protocol=PROTOCOL))
