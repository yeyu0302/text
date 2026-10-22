"""Run frozen confirmation, calibration evaluation and final artifact delivery.

Run only after priority, comparator, refinement and native/core jobs finish.
"""
import json
import runpy
from src.train import ROOT
from src.postprocess import main as postprocess
from src.deliver import main as deliver
from src.report import main as report

if __name__=='__main__':
    expected=['B07_full','D04_smooth_strong','C01_full_cat','C03_native_cat','C07_core_spline']
    for name in expected:
        assert (ROOT/'results/predictions'/name/'summary.json').exists(),name
    runpy.run_module('src.choose_complement',run_name='__main__')
    runpy.run_module('src.confirm',run_name='__main__')
    plan=json.loads((ROOT/'configs/confirmation_plan.json').read_text())
    names=[c['name'] for c in plan['selected']]
    postprocess(names,'F00')
    runpy.run_module('src.choose_calibration',run_name='__main__')
    decision=json.loads((ROOT/'configs/decision_plan.json').read_text())
    if decision['needs_independent_calibration_run']:
        postprocess(['R2026_'+n for n in names],'F01')
    deliver()
    report()
