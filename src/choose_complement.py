"""Development-only bounded model-family complement comparison."""
import json
import shutil
import numpy as np
import pandas as pd
from src.train import ROOT
from src.evaluation import metrics

def main():
    frozen=ROOT/'configs/confirmation_plan.json'
    if frozen.exists():
        prior=json.loads(frozen.read_text())
        if prior.get('selection','').startswith('Frozen seed-42 leader plus'):
            print('Using existing frozen complement plan',flush=True)
            return
    rows=[json.loads(p.read_text()) for p in (ROOT/'results/predictions').glob('*/summary.json')]
    rows=[r for r in rows if r['seed']==42 and r['family']!='legacy_nested']
    best_by_model={}
    for r in sorted(rows,key=lambda r:r['balanced_accuracy'],reverse=True):
        best_by_model.setdefault(r['model'],r)
    leader=max(rows,key=lambda r:r['balanced_accuracy'])
    z=np.load(ROOT/'results/predictions'/leader['name']/'predictions.npz')
    comparison=[]
    for kind,r in best_by_model.items():
        if kind==leader['model']:
            continue
        p=np.load(ROOT/'results/predictions'/r['name']/'predictions.npz')
        comparison.append(dict(leader=leader['name'],complement=r['name'],**metrics(z['y'],(z['p']+p['p'])/2)))
    frame=pd.DataFrame(comparison).sort_values('balanced_accuracy',ascending=False)
    frame.to_csv(ROOT/'results/development_complements.csv',index=False)
    partner=next(r for r in rows if r['name']==frame.iloc[0].complement)
    path=ROOT/'configs/confirmation_plan.json'
    if path.exists() and not (ROOT/'configs/confirmation_initial_score_rank_plan.json').exists():
        shutil.copy2(path,ROOT/'configs/confirmation_initial_score_rank_plan.json')
    plan=dict(seed=2026,selected=[leader,partner],
              selection='Frozen seed-42 leader plus its best fixed equal-weight complement among the best configuration of each other model family; all pair selection uses seed 42 only',
              baseline='B03_hierarchy',development_pair_score=float(frame.iloc[0].balanced_accuracy))
    path.write_text(json.dumps(plan,indent=2))
    print(frame.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
