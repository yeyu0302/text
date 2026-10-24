"""Portable one-command reproducibility entry point; outputs can be isolated by env."""
import subprocess,sys,os,shutil
from .common import ROOT,PROJECT,RESULTS,REPORTS

def main():
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
    os.environ.setdefault('MPLCONFIGDIR',str(RESULTS/'matplotlib_cache'))
    RESULTS.mkdir(parents=True,exist_ok=True);REPORTS.mkdir(parents=True,exist_ok=True)
    from .audit_data import run as audit
    from .run_experiments import evaluate_outer,screening,final
    from .train_baseline import BASE
    audit();evaluate_outer('fixed_baseline',BASE,42)
    from .error_attribution import run as forensic
    forensic();screening()
    from .additional_diagnostics import run as additional
    additional()
    from .extra_checks import run as checks
    checks()
    from .review_pairs import run as review
    review()
    from .proxy_reconstruction import run as reconstruction
    reconstruction()
    from .controlled_outer_checks import evaluate_outer as outer
    from .run_experiments import cfg
    for name,c in [('S2_outer_precision_uncertainty',cfg(proxy='V3')),('S3_outer_balanced_log',cfg(te='balanced_log')),
                   ('S3_outer_consistent_fold_ensemble',cfg(strategy='fold_ensemble')),('S4_outer_knn20',cfg(knn='region_k20')),('S5_outer_hgb',cfg(model='hgb'))]:outer(name,c,42)
    final()
    from .finalize_candidates import run as export
    export()
    from .report_results import run as report
    report()
    subprocess.check_call([sys.executable,'-B','-m','pytest',str(PROJECT/'tests'),'-q'],cwd=ROOT)

if __name__=='__main__':main()
