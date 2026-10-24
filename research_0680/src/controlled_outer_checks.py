"""Fixed hypothesis controls, not selected candidates and never used for export gates."""
from .run_experiments import evaluate_outer,cfg

if __name__=='__main__':
    for name,config in [
        ('S2_outer_precision_uncertainty',cfg(proxy='V3')),
        ('S3_outer_balanced_log',cfg(te='balanced_log')),
        ('S3_outer_consistent_fold_ensemble',cfg(strategy='fold_ensemble')),
        ('S4_outer_knn20',cfg(knn='region_k20')),
        ('S5_outer_hgb',cfg(model='hgb'))]:
        evaluate_outer(name,config,42)
