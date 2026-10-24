"""Paper-grounded soft BBSE diagnostic using cached, scope-matched predictions."""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from research_0680.src.common import read_data,dump,PROJECT

HERE=Path(__file__).resolve().parent
OUT=Path(os.environ.get('RESEARCH0680_PHASE6_RESULTS',str(HERE/'results')))

def solve_prior(matrix,marginal):
    raw=np.linalg.solve(matrix,marginal)
    if np.all(raw>=0) and np.all(raw<=1):return raw/raw.sum(),raw
    fit=minimize(lambda v:np.sum((matrix@v-marginal)**2),np.ones(3)/3,method='SLSQP',
        bounds=[(0,1)]*3,constraints=dict(type='eq',fun=lambda v:v.sum()-1))
    assert fit.success
    return fit.x,raw

def run():
    OUT.mkdir(parents=True,exist_ok=True);x,y,test,_=read_data();rows=[];rng=np.random.default_rng(923)
    for seed in [42,2026]:
        folder=PROJECT/f'phase5/kernel_results/seed{seed}'
        oof=pd.read_csv(folder/'oof_predictions.csv');tp=pd.read_csv(folder/'test_probabilities.csv')
        assert oof.id.equals(x.id) and tp.id.equals(test.id)
        p=oof[['p1','p2','p3']].to_numpy();target=tp[['p1','p2','p3']].to_numpy()
        parts=[[p[(oof.fold.to_numpy()==f)&(y==k)] for k in range(3)] for f in range(5)]
        c=np.mean([np.column_stack([a.mean(0) for a in part]) for part in parts],axis=0)
        estimate,raw=solve_prior(c,target.mean(0));draws=[]
        for _ in range(1000):
            cb=np.mean([np.column_stack([a[rng.integers(0,len(a),len(a))].mean(0) for a in part]) for part in parts],axis=0)
            tb=target[rng.integers(0,len(target),len(target))].mean(0)
            draws.append(solve_prior(cb,tb)[0])
        draws=np.asarray(draws)
        for k in range(3):rows.append(dict(seed=seed,level=k+1,estimated_target_prior=estimate[k],raw_estimate=raw[k],
            ci_low=np.quantile(draws[:,k],.025),ci_high=np.quantile(draws[:,k],.975),uniform_one_third=1/3,
            condition_number=np.linalg.cond(c)))
        pd.DataFrame(c,index=['pred_p1','pred_p2','pred_p3'],columns=['true1','true2','true3']).to_csv(OUT/f'bbse_soft_matrix_seed{seed}.csv')
    pd.DataFrame(rows).to_csv(OUT/'bbse_prior_diagnostic.csv',index=False)
    dump(OUT/'bbse_protocol.json',dict(source='Lipton, Wang & Smola 2018, Sections 3-5',
        assumptions=['Class-conditional raw feature distributions are invariant between train and test','Every target class exists in source','Soft confusion matrix invertible'],
        scope='Each fold source OOF and test prediction uses the same trained classifier and feature/TE map; class-conditional matrices averaged equally across folds',
        bootstrap='1000 conditional source-within-class-within-fold and unlabeled-test row resamples; does not include retraining uncertainty',
        not_available_for_original_submission3='Matching original OOF/probabilities missing; cannot transplant another model confusion matrix',
        limitation='This does not prove label shift assumptions, exact 1700 counts, any individual test label, or accuracy improvement. No correction applied to a submission.'))
    print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':run()
