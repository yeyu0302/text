"""Two genuinely new EBM designs; cached old controls, no hyperparameter sweep."""
import hashlib,json,os,time,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.model_selection import train_test_split
from research_0680.src.common import read_data,dump,manifest,full_metrics,bootstrap
from research_0680.src.train_baseline import FeatureFactory,BASE,CORE
from src.features import REAL_CAT_COLS
from src.evaluation import prior_correct
from .run import INNER_REFERENCE

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE5_EBM_RESULTS',str(HERE/'ebm_results')))
PROTOCOL=json.loads((HERE/'ebm_protocol.json').read_text())

def matrices(x,y,q):
    a,b=FeatureFactory(BASE).fit_transform(x,y,q)
    a=a[CORE].copy();b=b[CORE].copy()
    for c in REAL_CAT_COLS:
        a['cat_'+c]=x[c].fillna('MISSING').astype(str).to_numpy()
        b['cat_'+c]=q[c].fillna('MISSING').astype(str).to_numpy()
    return a,b

def predictions(x,y,q):
    a,b=matrices(x,y,q);prior=np.bincount(y,minlength=3)/len(y);result={}
    for name,pairs in PROTOCOL['models'].items():
        start=time.perf_counter()
        model=ExplainableBoostingClassifier(**PROTOCOL['params'],interactions=pairs,
            feature_types=['continuous']*len(CORE)+['nominal']*len(REAL_CAT_COLS))
        model.fit(a,y);result[name]=prior_correct(model.predict_proba(b),prior)
        OUT.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(dict(term=model.term_names_,importance=model.term_importances())).to_csv(OUT/f'{name}_terms.csv',index=False)
        dump(OUT/f'{name}_fit.json',dict(runtime_seconds=time.perf_counter()-start,terms=model.term_names_,
            best_iteration=np.asarray(model.best_iteration_).tolist(),feature_count=a.shape[1]))
        print('EBM FIT',name,'seconds',round(time.perf_counter()-start,2),flush=True)
    return result

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True);x,y,_,_=read_data();t,v=train_test_split(np.arange(len(y)),test_size=.25,stratify=y,random_state=847)
    old=np.load(INNER_REFERENCE/'screen/probabilities.npz');assert np.array_equal(old['ids'],x.id.iloc[v])
    pred=predictions(x.iloc[t],y[t],x.iloc[v]);rows=[]
    np.savez_compressed(OUT/'screen_probabilities.npz',ids=x.id.iloc[v],**pred)
    for name,p in pred.items():rows.append(dict(model=name,scope='screen_3750',**full_metrics(y[v],p),
        **bootstrap(y[v],p,old['reference']),delta_vs_same_input_cat=float(full_metrics(y[v],p)['balanced_accuracy']-full_metrics(y[v],old['cat'])['balanced_accuracy'])))
    pd.DataFrame(rows).to_csv(OUT/'screen.csv',index=False)
    dump(OUT/'manifest.json',manifest(dict(protocol=PROTOCOL,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         interpret_core_version=importlib.metadata.version('interpret-core'))))
    promoted=max(r['delta'] for r in rows)>.002
    dump(OUT/'gate.json',dict(promoted=promoted,reason='Advance only if new screening gain > .002, otherwise stop without retraining'))
    print('EBM SCREEN',[(r['model'],r['balanced_accuracy'],r['delta']) for r in rows],flush=True)
