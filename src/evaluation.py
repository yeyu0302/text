"""Metrics and exact cardinality decisions, without label-based tuning."""
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss, recall_score


def metrics(y, p):
    pred = p.argmax(axis=1)
    return dict(balanced_accuracy=balanced_accuracy_score(y, pred),
                accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average='macro'),
                log_loss=log_loss(y, p, labels=[0, 1, 2]),
                **{f'recall_{k+1}': v for k, v in enumerate(recall_score(y, pred, labels=[0, 1, 2], average=None))})


def prior_correct(p, priors):
    q = np.clip(p, 1e-12, 1) / np.asarray(priors)
    return q / q.sum(axis=1, keepdims=True)


def exact_assignment(p, counts=None, objective='accuracy'):
    """Transportation LP: totally unimodular constraints imply an integral optimum.

    Maximizes sum of calibrated probabilities (expected correct predictions), or
    sum log probabilities. HiGHS dual simplex returns a basic optimum. Check
    integrality, row constraints and cardinalities before accepting its solution.
    Complexity is linear in variables (3*N), unlike a duplicated N*N assignment.
    """
    p = np.asarray(p, dtype=float)
    n, k = p.shape
    counts = np.asarray(counts if counts is not None else [n//k]*(k-1)+[n-(k-1)*(n//k)])
    if len(counts) != k or counts.sum() != n or np.any(counts < 0) or np.any(counts != counts.astype(int)):
        raise ValueError('Invalid class counts')
    if not np.isfinite(p).all() or np.any(p < 0) or objective not in ('accuracy', 'log'):
        raise ValueError('Invalid probabilities or objective')
    score = p if objective == 'accuracy' else np.log(np.clip(p, 1e-12, 1))
    cols = np.arange(n*k)
    a = sparse.coo_matrix((np.ones(2*n*k),
                          (np.r_[np.repeat(np.arange(n), k), n+np.tile(np.arange(k), n)],
                           np.r_[cols, cols])), shape=(n+k, n*k)).tocsr()
    result = linprog(-score.ravel(), A_eq=a, b_eq=np.r_[np.ones(n), counts],
                     bounds=(0, None), method='highs-ds')
    if not result.success:
        raise RuntimeError(result.message)
    x = result.x.reshape(n, k)
    if not np.allclose(x, np.round(x), atol=1e-6):
        raise RuntimeError('Nonintegral LP solution')
    pred = x.argmax(axis=1)
    assert np.array_equal(np.bincount(pred, minlength=k), counts)
    assert np.allclose(x.sum(axis=1), 1)
    return pred


def folds_for(y, seed=42, n=5):
    """Exactly reproduce the legacy class-wise shuffled round-robin folds."""
    rng = np.random.RandomState(seed)
    folds = np.zeros(len(y), dtype=int)
    for k in range(3):
        idx = np.flatnonzero(y == k)
        rng.shuffle(idx)
        for f in range(n):
            folds[idx[f::n]] = f
    return folds
