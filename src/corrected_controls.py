"""Final strict-prior legacy controls, matching original histogram RNG seeds."""
from src.train import run

if __name__=='__main__':
    run('A03_nested_legacy_hist_matching_rng','legacy_nested','hist')
    run('A04_nested_legacy_spline_strict_prior','legacy_nested','spline')
