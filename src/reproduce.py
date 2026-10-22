"""Run unchanged legacy models, intercepting and retaining their probabilities.

Run in the minimal onboarding venv so optional packages cannot silently change
the historical two-model ensemble. Does not modify the original implementation.
"""
import ast
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import optimize_pipeline as old


def main():
    dest = ROOT / 'results' / 'legacy_reproduction'
    dest.mkdir(exist_ok=True)
    if (dest/'metadata.json').exists() and (dest/'predictions.npz').exists():
        print('CACHED legacy reproduction; preserving existing predictions and submission',flush=True)
        return
    tr, te = pd.read_csv(ROOT/'train.csv'), pd.read_csv(ROOT/'test.csv')
    y = tr.label.to_numpy()-1
    rng = np.random.RandomState(42)
    folds = np.zeros(len(y), dtype=int)
    for k in range(3):
        idx = np.flatnonzero(y == k)
        rng.shuffle(idx)
        for f in range(5):
            folds[idx[f::5]] = f
    priors = np.bincount(y)/len(y)
    t = time.perf_counter()
    a, b = old.build_clean_and_engineered_features(tr, te, y, folds)
    tree = ast.parse((ROOT/'optimize_pipeline.py').read_text())
    cols = next(ast.literal_eval(node.value) for node in ast.walk(tree)
                if isinstance(node, ast.Assign) and any(isinstance(x, ast.Name) and x.id == 'gbdt_cols' for x in node.targets))
    ps, ts = old.fit_predict_spline_softmax(a, y, folds, b, priors)
    pg, tg = old.fit_predict_fast_hist_gbdt(a[cols], y, folds, b[cols], priors)
    qs, qg = old.apply_prior_shift(ps, priors), old.apply_prior_shift(pg, priors)
    qt = .4*old.apply_prior_shift(ts, priors)+.6*old.apply_prior_shift(tg, priors)
    q = .4*qs+.6*qg
    np.savez_compressed(dest/'predictions.npz', spline=qs, hist=qg, ensemble=q,
                        raw_spline=ps, raw_hist=pg, test=qt, folds=folds, y=y)
    rows = []
    for name, p in [('spline', qs), ('hist', qg), ('ensemble', q)]:
        for f in [-1, 0, 1, 2, 3, 4]:
            mask = np.ones(len(y), dtype=bool) if f == -1 else folds == f
            score, recalls = old.compute_balanced_accuracy(y[mask], p[mask].argmax(1))
            rows.append(dict(model=name, fold=f, balanced_accuracy=score, recalls=recalls))
    pd.DataFrame(rows).to_csv(dest/'metrics.csv', index=False)
    pred, _ = old.enforce_exact_cardinality(qt, [1700]*3)
    pd.DataFrame({'id':te.id, 'label':pred+1}).to_csv(dest/'submission.csv', index=False)
    meta = dict(seed=42, features=len(a.columns), hist_features=len(cols),
                models={'spline':{'l2_reg':35, 'n_iter':45},
                        'hist':{'n_estimators':120,'lr':.06,'max_depth':5,'max_leaves':20,'colsample':.65,'B':32}},
                runtime_seconds=time.perf_counter()-t,
                versions={x:importlib.metadata.version(x) for x in ['numpy','pandas','scipy']},
                submission_counts=pd.Series(pred+1).value_counts().to_dict())
    (dest/'metadata.json').write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame(rows).query('fold == -1').to_string(index=False), flush=True)
    print(json.dumps(meta), flush=True)


if __name__ == '__main__':
    main()
