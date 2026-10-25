"""Sensitivity analysis for observed sparse-village exposure, not a new predictor."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit,logit
from .audit import load

HERE=Path(__file__).resolve().parent

def transport(y,correct,small,target_small):
    s=np.array([small[y==k].mean() for k in range(3)])
    rates=np.array([[correct[(y==k)&(small==g)].mean() for g in [False,True]] for k in range(3)])
    tau=brentq(lambda t:expit(logit(s)+t).mean()-target_small,-20,20)
    q=expit(logit(s)+tau)
    baseline=(rates[:,1]*s+rates[:,0]*(1-s)).mean()
    score=(rates[:,1]*q+rates[:,0]*(1-q)).mean()
    return dict(source_small_by_class=s.tolist(),assumed_target_small_by_class=q.tolist(),recall_by_class_large_small=rates.tolist(),odds_shift=float(tau),balanced_source_score=float(baseline),transported_score=float(score),delta=float(score-baseline))

def main():
    output=HERE/'risk_sensitivity.json'
    if output.exists():raise SystemExit('Preserve existing sensitivity result')
    x,y,test,a,b=load();counts=x.f_141.value_counts();small=(x.f_141.map(counts).to_numpy()<=15);target_small=float((test.f_141.map(counts).fillna(0)<=15).mean());correct=(a['bagged'].argmax(1)==y).astype(float)
    result=transport(y,correct,small,target_small)
    result.update(target_small_fraction=target_small,assumptions=['Hypothetical target class prior fixed at 1/3','Sparse-group odds shift identical across classes','Class-and-group-conditional prediction recall is invariant'],scope='Sensitivity calculation only: group defined by full source X counts for both populations; original base fold training counts and full-fit test scope unknown',limitations='These assumptions are not established; value is not an independently validated estimate or Kaggle forecast. It explains why covariate prevalence must accompany a single OOF score. No candidate selected or exported.')
    output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
