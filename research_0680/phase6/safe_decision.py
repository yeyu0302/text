"""Explicit expected-accuracy decoding; never infer quotas from label files."""
import numpy as np
from src.evaluation import exact_assignment

def decode(probabilities,known_counts=None):
    p=np.asarray(probabilities,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or np.min(p)<0:
        raise ValueError('Require actual finite three-class probabilities, not a hard-label submission')
    if not np.allclose(p.sum(1),1):raise ValueError('Probabilities must sum to one')
    if known_counts is None:return p.argmax(1)+1
    # Caller must substantiate known_counts externally. Sum(p) is expected
    # correct count; sum(log(p)) optimizes a different utility.
    return exact_assignment(p,known_counts,objective='accuracy')+1
