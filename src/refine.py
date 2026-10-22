"""Small hypothesis-driven follow-ups after priority ablations complete."""
import json
from pathlib import Path
from src.train import run

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    runs=[json.loads(p.read_text()) for p in (ROOT/'results/predictions').glob('B*/summary.json')]
    strongest=max(runs,key=lambda r:r['balanced_accuracy'])
    family=strongest['family']
    (ROOT/'configs/refinement_plan.json').write_text(json.dumps(dict(
        family_selected_on_seed42=family,
        rationale='Hierarchy/repair ablations determine feature space before a two-point XGBoost regularization search. Native CatBoost tests direct high-cardinality interactions without treating group codes as ordered numbers.',
        search=[dict(max_depth=5,min_child_weight=8,gamma=0.,reg_alpha=.1,reg_lambda=3.),
                dict(max_depth=7,min_child_weight=12,gamma=.2,reg_alpha=.5,reg_lambda=8.)],
        outer_seed=42,confirmation_seed=2026),indent=2))
    run('C10_xgb_less_shrinkage',family,'xgb',overrides=dict(max_depth=5,min_child_weight=8,gamma=0.,reg_alpha=.1,reg_lambda=3.,n_jobs=3))
    run('C11_xgb_deeper',family,'xgb',overrides=dict(max_depth=7,min_child_weight=12,gamma=.2,reg_alpha=.5,reg_lambda=8.,n_jobs=3))
    # These test aggregation and raw preservation, not a repeat of a Top-50 run.
    run('D00_median','full','xgb',aggregate='median',overrides={'n_jobs':3})
    run('D01_trimmed','full','xgb',aggregate='trimmed',overrides={'n_jobs':3})
    run('D02_repaired_only','full','xgb',raw_mode='repaired',overrides={'n_jobs':3})
    run('D03_smooth_weak',family,'xgb',smoothing=.3,overrides={'n_jobs':3})
    run('D04_smooth_strong',family,'xgb',smoothing=3.,overrides={'n_jobs':3})
