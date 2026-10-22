"""Freeze seed-42 choices, then evaluate them on an independent split."""
import json
from src.train import ROOT, run

if __name__=='__main__':
    path=ROOT/'configs/confirmation_plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
    else:
        rows=[json.loads(p.read_text()) for p in (ROOT/'results/predictions').glob('*/summary.json')]
        rows=[r for r in rows if r['seed']==42 and r['family']!='legacy_nested']
        # Select two distinct model families before inspecting any seed-2026 CV.
        selected=[]
        for r in sorted(rows,key=lambda r:r['balanced_accuracy'],reverse=True):
            if r['model'] not in [s['model'] for s in selected]:
                selected.append(r)
            if len(selected)==2:
                break
        plan=dict(seed=2026,selected=selected,selection='Top two distinct model families on seed 42, frozen before seed 2026',baseline='B03_hierarchy')
        path.write_text(json.dumps(plan,indent=2))
    for cfg in plan['selected']:
        params=dict(cfg['params'])
        if cfg['model'] in ('xgb','lgb'):
            params['n_jobs']=4
        if cfg['model']=='cat':
            params['thread_count']=4
        run('R2026_'+cfg['name'],cfg['family'],cfg['model'],2026,params,cfg['smoothing'],cfg['aggregate'],cfg['raw_mode'],cfg['native'])
    run('R2026_baseline_hierarchy','hierarchy','xgb',2026,{'n_jobs':4})
