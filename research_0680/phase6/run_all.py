"""Forensic audit only; no damage classifier retraining; require fresh outputs."""
import os
from pathlib import Path
from .wiener_audit import run as measurement
from .submission_audit import run as submissions
from .shift_audit import run as shift

if __name__=='__main__':
    destination=os.environ.get('RESEARCH0680_PHASE6_RESULTS')
    if not destination:
        raise SystemExit('Set RESEARCH0680_PHASE6_RESULTS to a fresh output directory; old results are preserved.')
    path=Path(destination)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise SystemExit('Output path already contains files; choose a fresh directory.')
    submissions();measurement();shift()
