"""Training-local FDR control for omitted predictors; not a global feature list."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from src.features import SELECTED
from research_0680.phase3 import run as engine

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE4_DISCOVERY_RESULTS',str(HERE/'discovery_results')))
PROTOCOL=json.loads((HERE/'discovery_protocol.json').read_text())

def select(x,y,q):
    parts={};rows=[]
    for c in [c for c in x if c not in SELECTED+['id','label']]:
        if x[c].dtype==object:
            levels=sorted(x[c].fillna('MISSING').astype(str).unique());mapping={v:i for i,v in enumerate(levels)}
            a=x[c].fillna('MISSING').astype(str).map(mapping).to_numpy();b=q[c].fillna('MISSING').astype(str).map(mapping).fillna(len(levels)).to_numpy(int)
        else:
            edges=np.unique(x[c].dropna().quantile(np.linspace(.1,.9,9)).to_numpy())
            a=np.searchsorted(edges,x[c].to_numpy(float));b=np.searchsorted(edges,q[c].to_numpy(float))
            a=np.where(x[c].isna(),len(edges)+1,a);b=np.where(q[c].isna(),len(edges)+1,b)
        n=int(a.max())+1;counts=np.bincount(a*3+y,minlength=n*3).reshape(n,3);counts=counts[counts.sum(1)>0]
        p=chi2_contingency(counts,correction=False)[1] if len(counts)>1 else 1.
        parts[c]=(a,b);rows.append(dict(feature=c,p=float(p)))
    table=pd.DataFrame(rows).sort_values('p');rank=np.arange(1,len(table)+1)
    table['q']=np.minimum.accumulate((table.p.to_numpy()*len(table)/rank)[::-1])[::-1].clip(0,1)
    selected=table.loc[table.q<.05,'feature'].head(8).tolist()
    table['selected']=table.feature.isin(selected)
    OUT.mkdir(parents=True,exist_ok=True);key=hashlib.sha256(x.id.to_numpy().tobytes()).hexdigest()[:12]
    table.to_csv(OUT/f'feature_audit_{key}.csv',index=False)
    return selected,parts

def predictions(x,y,q):
    a,b=engine.FeatureFactory(engine.BASE).fit_transform(x,y,q)
    prior=np.bincount(y,minlength=3)/len(y);p=engine.prior_correct(engine.model_predict(a,y,b),prior)
    selected,parts=select(x,y,q)
    if selected:
        for c in selected:
            ta,tb=parts[c]
            for v in np.unique(ta):a['discover_'+c+'_'+str(v)]=(ta==v).astype(float);b['discover_'+c+'_'+str(v)]=(tb==v).astype(float)
        new=engine.prior_correct(engine.model_predict(a,y,b),prior)
    else:new=p.copy()
    return {'reference':p,'discovery':new,'reference+discovery':.5*(p+new)}

if __name__=='__main__':
    engine.OUT=OUT;engine.PROTOCOL=PROTOCOL.copy();engine.PROTOCOL['backend_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    engine.CHOICES=PROTOCOL['choices'];engine.predictions=predictions;engine.run()
