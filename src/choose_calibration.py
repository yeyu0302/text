"""Freeze a decision recipe using development seed 42 only."""
import json
import pandas as pd
from src.train import ROOT

if __name__=='__main__':
    comparison=pd.read_csv(ROOT/'results/F00/comparison.csv').set_index('method')
    confirmation=json.loads((ROOT/'configs/confirmation_plan.json').read_text())
    leading=confirmation['selected'][0]['balanced_accuracy']
    permitted=comparison.loc[['prior','multipliers','weighted','stack']]
    best=permitted.balanced_accuracy.idxmax()
    threshold=max(comparison.loc['prior','balanced_accuracy'],leading)+.002
    selected=best if comparison.loc[best,'balanced_accuracy']>=threshold else 'prior'
    plan=dict(method=selected,selection_seed=42,independent_confirmation_seed=2026,
              selected_score=float(comparison.loc[selected,'balanced_accuracy']),
              rule='Use a fitted decision/ensemble method only if it beats both fixed averaging and the frozen single-model leader by >=0.002 BA on development seed. Otherwise use the fixed-model retention rule.',
              needs_independent_calibration_run=selected!='prior')
    (ROOT/'configs/decision_plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps(plan),flush=True)
