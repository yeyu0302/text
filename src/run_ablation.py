"""Priority A/B and audit controls; existing experiment IDs are never overwritten."""
from src.train import run

if __name__=='__main__':
    for name,family in [('B00_all_raw','all_raw'),('B01_selected','selected'),('B02_proxy','proxy'),
                        ('B03_hierarchy','hierarchy'),('B04_combined','combined'),
                        ('B05_structural','structural'),('B06_groups','groups'),('B07_full','full')]:
        run(name,family)
    run('A01_nested_legacy_hist','legacy_nested','hist')
    run('A02_nested_legacy_spline','legacy_nested','spline')
