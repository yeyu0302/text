"""Generate evidence-based reports after all selected runs have completed."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.train import ROOT, refresh_log
from src.evaluation import metrics
from src.deliver import paired_bootstrap, validate_submission


def table(frame, columns=None):
    d=frame if columns is None else frame[columns]
    def value(x):
        if isinstance(x,(float,np.floating)):
            return '' if np.isnan(x) else f'{x:.6f}'
        return str(x).replace('|','/')
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+
                     ['| '+' | '.join(value(x) for x in row)+' |' for row in d.itertuples(index=False,name=None)])


def main():
    refresh_log()
    dest=ROOT/'results'
    selected=json.loads((ROOT/'configs/selected_solution.json').read_text())
    validation=pd.read_csv(dest/'finalist_validation.csv')
    final_oof=pd.read_csv(dest/'oof_predictions.csv')
    final_fold_rows=[]
    for seed in [42,2026]:
        p=final_oof[[f'final_seed{seed}_p{k}' for k in [1,2,3]]].to_numpy()
        labels=final_oof.label.to_numpy()-1
        folds=final_oof[f'fold_seed{seed}'].to_numpy()
        for f in range(5):
            mask=folds==f
            final_fold_rows.append(dict(seed=seed,fold=f,**metrics(labels[mask],p[mask])))
    final_folds=pd.DataFrame(final_fold_rows)
    final_folds.to_csv(dest/'finalist_fold_metrics.csv',index=False)
    std=final_folds.groupby('seed').balanced_accuracy.std()
    validation['fold_std']=validation.seed.map(std)
    validation.to_csv(dest/'finalist_validation.csv',index=False)
    allruns=pd.read_csv(dest/'experiment_log.csv')
    original=np.load(dest/'legacy_reproduction/predictions.npz')
    hist=np.load(dest/'predictions/A03_nested_legacy_hist_matching_rng/predictions.npz')
    spline=np.load(dest/'predictions/A04_nested_legacy_spline_strict_prior/predictions.npz')
    corrected=.6*hist['p']+.4*spline['p']
    corrected_metrics=metrics(hist['y'],corrected)
    uncertainty=pd.read_csv(dest/'paired_bootstrap.csv')
    label='seed42 frozen ensemble minus corrected legacy ensemble'
    if label not in uncertainty.comparison.values:
        p=final_oof[[f'final_seed42_p{k}' for k in [1,2,3]]].to_numpy()
        extra=dict(comparison=label,**paired_bootstrap(hist['y'],p,corrected))
        uncertainty=pd.concat([uncertainty,pd.DataFrame([extra])],ignore_index=True)
        uncertainty.to_csv(dest/'paired_bootstrap.csv',index=False)
    np.savez_compressed(dest/'corrected_legacy_ensemble.npz',p=corrected,y=hist['y'],folds=hist['folds'],test=.6*hist['test']+.4*spline['test'])
    original_metrics=metrics(original['y'],original['ensemble'])
    rows=[]
    for label,p in [('Original two-model ensemble (LEAKED)',original['ensemble']),('Corrected nested two-model ensemble',corrected)]:
        row=dict(method=label,**metrics(hist['y'],p))
        rows.append(row)
    pd.DataFrame(rows).to_csv(dest/'original_vs_corrected.csv',index=False)
    ablation=allruns[allruns.name.str.match(r'B0[0-7]_')].sort_values('name').copy()
    ablation['increment_vs_previous']=ablation.balanced_accuracy.diff()
    extra=[]
    for a,b in [('B02_proxy','B01_selected'),('B03_hierarchy','B01_selected'),('B05_structural','B04_combined'),('B06_groups','B05_structural'),('B07_full','B06_groups')]:
        aa=np.load(dest/'predictions'/a/'predictions.npz');bb=np.load(dest/'predictions'/b/'predictions.npz')
        extra.append(dict(comparison=a+' minus '+b,**paired_bootstrap(aa['y'],aa['p'],bb['p'])))
    pd.DataFrame(extra).to_csv(dest/'ablation_bootstrap.csv',index=False)
    screening=pd.read_csv(dest/'exploratory_feature_screen.csv')
    omitted=screening[~screening.selected_hypothesis]
    proxy=pd.read_csv(dest/'proxy_reconstruction.csv')
    corrupt=pd.read_csv(dest/'controlled_corruption.csv')
    smooth=allruns[allruns.name.str.match(r'D0[0-4]_')]
    ablation_report=f'''# Feature-family ablation

All eight comparisons use the exact same class-wise shuffled seed-42 outer
folds, the same fixed 800-tree XGBoost budget, prior correction and fold-local
preprocessing. Raw values remain alongside repaired values except the explicit
repaired-only variant. `all_raw` is an available-parameter reference; the missing
historical `best_params.json` prevents an exact reproduction of the old 238-feature
hyperparameter search. All requested project/data files are present.

{table(ablation,['name','features_min','features_max','balanced_accuracy','fold_std','increment_vs_previous'])}

The largest observed gain is from hierarchical encoding, not proxy repair alone.
Group statistics help the structural+hierarchical recipe on this split; adding
all material/group target interactions hurts this XGBoost configuration. These
are conditional development comparisons, not proof of physical causation.
Paired conditional bootstrap intervals are in `ablation_bootstrap.csv`.

## Aggregation, repair preservation and shrinkage

{table(smooth,['name','family','aggregate','raw_mode','smoothing','balanced_accuracy','fold_std'])}

Mean/weighted/median/trimmed reconstruction uses apparently valid source values,
with a 70/30 fit/holdout split for any learned weights or affine calibration.
It does not reveal the true values of naturally missing/anomalous observations.
Complete individual proxy correlations are in `proxy_correlations.csv`.
The reconstruction diagnostic's trimmed mean removes one value from each end.
The CV trimmed variant requires at least four proxies before trimming; its
three-proxy area group retains the mean rather than a single middle sensor.

{table(proxy.query('calibrated == False'),['source','method','holdout_rows','masked_mae','masked_rmse','rounded_exact_rate'])}

Controlled corruption of valid records through the actual repair code (not
through an oracle that knows which value was corrupted):

{table(corrupt.query('aggregate == "mean"'),['source','corruption','n','mae','exact_rate'])}

Age/storey/missing-height reconstruction is imperfect. The sign-flip and tenfold
rules have substantially stronger controlled evidence in the tested records.
Unusual-value proxy residuals are in `anomaly_evidence.csv`; none proves that
every unusual value is erroneous. Preserve raw values as a source of evidence.
The fitted affine reconstruction diagnostics were not silently installed as a
new feature transformation or validated predictive improvement.

## Hierarchy and omitted features

`hierarchy_audit.csv` verifies no conflicting parent mappings in train, test or
their union. Train has 945 fine groups, test 924; all test fine groups occur in
train. Their geographic interpretation remains a hypothesis.

All 238 predictors also received descriptive univariate association screening.
Among {len(omitted)} omitted features, {int((omitted.exploratory_fdr_q<.05).sum())}
pass exploratory FDR 0.05. This supports the pruning hypothesis but cannot prove
that all omitted columns are independent noise or that interactions contain no
signal. The screen uses all training labels, is explicitly exploratory, and was
**not** used to select features or claim independent validation. The historical
52-feature list itself may retain earlier full-data selection optimism.

The manually chosen vulnerability coefficients are retained hypotheses. No
physical validation or coefficient optimality is claimed.
'''
    (dest/'feature_ablation.md').write_text(ablation_report)
    attempts=[]
    history=(dest/'ablation_run.log').read_text()
    if "Target is multiclass but average='binary'" in history:
        attempts.append(dict(attempt='initial_B00_instrumentation',status='resolved',
                             issue='Multiclass recall metric needed average=None',
                             resolution='Metric corrected, regression test added, full B00 rerun completed',
                             runtime_seconds=None,scientific_score=None))
    if 'Existing experiment C00_full_lgb has a different configuration' in history:
        attempts.append(dict(attempt='redundant_comparator_scheduler_call',status='resolved',
                             issue='An older running scheduler attempted an already completed comparator with a different thread setting',
                             resolution='Cache refused overwrite; completed comparator retained; comparator scheduling separated from ablations',
                             runtime_seconds=None,scientific_score=None))
    pd.DataFrame(attempts).to_csv(dest/'attempt_log.csv',index=False)
    model_names=[c['name'] for c in selected['models']]
    f00=pd.read_csv(dest/'F00/comparison.csv')
    balanced=pd.read_csv(dest/'final_balanced_summary.csv')
    uncertainty=pd.read_csv(dest/'paired_bootstrap.csv')
    all_safe=allruns[(allruns.seed==42)&allruns.eligible_for_final_selection.fillna(False)].sort_values('balanced_accuracy',ascending=False)
    meta=json.loads((dest/'reproducibility.json').read_text())
    counts={p.name:pd.read_csv(p).label.value_counts().sort_index().to_dict() for p in [dest/'best_submission.csv',dest/'alternative_submission.csv']}
    for p in [dest/'best_submission.csv',dest/'alternative_submission.csv']:
        validate_submission(pd.read_csv(p),pd.read_csv(ROOT/'test.csv'))
    max_local=validation.balanced_accuracy.max()
    target_status='No independently measured 0.700+ or 0.800 score was demonstrated.' if max_local<.7 else 'A local 0.700+ validation score is not a confirmed Kaggle score; 0.800 remains unverified.'
    final=f'''# Earthquake damage optimization: final research report

## Outcome

The pipeline was audited, the target-encoding leakage was reproduced and fixed,
controlled ablations and targeted follow-ups were executed, and two validated
5,100-row submissions were generated. The primary uses **{', '.join(model_names)}**,
probability method **{selected['probability_method']}**, and an exact
**{selected['decision_objective']}** transportation objective with 1,700 rows/class.

**The public Kaggle benchmark remains the user-confirmed 0.676. No new submission
was uploaded or scored on Kaggle.** Local CV cannot confirm an improvement over
that score. {target_status} Competitor scores near 0.800 cannot be reproduced or
explained from leaderboard claims alone; no external labels were used to pursue them.

{table(validation)}

Seed 42 was used for development/model selection. Candidate model families and
decision rules depend on development data only; seed 2026 tests the frozen recipe.
All rows are out of fold, but repeated CV reuses the same dataset and is not a
new independent population sample. Full benchmark metrics include Accuracy,
Macro F1, three Recalls, log loss and fold variation. `log_loss` is on the original
imbalanced validation population after the reported probability transformation;
`raw_metrics` retains uncorrected-probability log loss for base models.

## Reproduction and leakage correction

{table(pd.DataFrame(rows))}

The reported historical 0.678741285 was nearly reproduced at
{original_metrics['balanced_accuracy']:.9f}, difference
{original_metrics['balanced_accuracy']-.6787412854030501:+.9f}. FastHistGBDT's
0.678468508 reproduced exactly; the small discrepancy is in SplineSoftmax.
The historical report has no original OOF file or package versions, so the exact
cause of that tiny numerical/result discrepancy cannot be established. The
reported simulated 0.679725490 also lacks its original sampling recipe; it is
not relabeled as independently reproduced. New balanced simulations below use
explicit recorded folds/seeds.

The final nested legacy control matches the original histogram RNG seeds,
fits priors separately within each inner fit, fixes prior-normalized encoding,
and fits custom scaling/bins on outer train only. Its unlabeled group aggregation
remains deliberately transductive. It is not the main inductive feature builder.
The reduction relative to the original therefore measures a corrected validation
protocol, not a fair deployment accuracy loss attributable to only one change.

Earlier A01/A02 diagnostic adapters are retained as experiment history. A01 used
the same histogram seed 52 in each outer fold; A03 supersedes it with matching
fold RNGs. A04 uses strict fit-local prior normalization. No interim control is
used for final model selection. Detailed proof and counterexamples are in
`leakage_audit.md`, `leakage_probe.json` and `cardinality_audit.json`.

## Controlled experiments

{table(ablation,['name','balanced_accuracy','fold_std'])}

{table(all_safe[['name','family','model','balanced_accuracy','fold_std']].head(18))}

All base-model runs have raw/adjusted OOF and test probabilities, per-fold metrics,
full parameters, seeds, source hash and wall runtime under `predictions/`. See
`experiment_log.csv` for complete history and `model_comparison.csv` for rankings
within each seed. Legacy leaked rows are explicitly ineligible. Wall times were
measured while sharing a 4-CPU quota, so they are not isolated speed benchmarks.
`attempt_log.csv` distinguishes two resolved instrumentation/scheduling failures
from completed scientific experiments; their original tracebacks are preserved.

The two-point XGBoost search was driven by the ablations; aggregation and smoothing
variants tested specific hypotheses. Native CatBoost tested categorical group
codes directly with a bounded 500-tree budget and pairwise categorical CTRs;
its change from 800-tree one-hot CatBoost is a model-design experiment, not an
isolated native-category ablation. The conditional no-TE native branch was not
run because this design did not beat ordinary CatBoost by the required margin.
A same-55-feature XGBoost,
custom histogram and custom spline comparison controls for model input space.
The stronger full-space standard logistic model is also evaluated. This does not
repeat the earlier Top-50 or embedding/MLP searches.

An optimizer audit found that the original spline curvature factor 0.24 is not
a general multiclass Hessian upper bound. C08 therefore keeps its quadratic
feature map but fits a converged standard multinomial optimizer (C=1/35,
unpenalized intercept). This additional control and its seed-2026 check are in
the experiment log. It was added as a numerical audit after the principal pair
was frozen; its development equal blend with CatBoost is about 0.67921 versus
0.67882 for the frozen pair, a difference below the predeclared 0.001 retention
margin. It does not silently replace either the historical model or the frozen
confirmation recipe. No convergence certificate is claimed for the old solver.

## Decision and ensemble experiments

{table(f00)}

Multipliers, small-grid weights and regularized held-out stacking are trained on
a calibration subset strictly inside each outer training partition. Other outer
OOF rows are not used for this tuning. Rank averaging is exploratory; its scores
are not assumed to be calibrated probabilities. Fixed averaging and calibration
retention rules are recorded in `confirmation_plan.json`, `decision_plan.json`
and `selected_solution.json`. A fitted method must improve development BA by
at least 0.002 over both fixed averaging and the frozen leading base model;
otherwise the simpler predeclared model-retention rule applies. Independent
confirmation does not retune that choice.

Balanced held-out subsets simulate the documented test proportions (5 repeats
per fold per seed); repetitions overlap and are not independent observations.
The exact decision objective is selected using seed 42 only; seed 2026 evaluates
the frozen decision. The class-capacity LP uses no individual validation labels.

{table(balanced)}

The sparse LP is globally optimal for its supplied scores and quotas; this says
nothing about the correctness/calibration of its probabilities. Probability-sum
and log objectives differ, and the old greedy method is neither reliably exact
nor globally optimal. No public-leaderboard feedback was used to choose them.

## Uncertainty

{table(uncertainty)}

These stratified paired bootstrap intervals condition on fitted OOF predictions;
they do not account for training-set overlap, adaptive model selection or every
source of training variability. Treat them alongside the independent-seed
comparison, not as proof of a leaderboard gain. Seed-42 selected maxima have
selection optimism. Modest gains smaller than the uncertainty are inconclusive.

## Submissions and reproducibility

- `best_submission.csv`: cross-seed fold predictions with the frozen probability
  recipe, exact quotas; no test accuracy available locally.
- `alternative_submission.csv`: leakage-safe full-data refits of the selected
  base models with equal probability weights and prior correction, followed by
  the same exact objective. This tests the training-size/variance tradeoff and
  is not claimed superior. The two files differ on {selected['disagreement_rows']} rows.
- Counts: `{json.dumps(counts)}`. Both preserve all IDs and their original order,
  have only labels 1/2/3, columns id,label, no missing values and 5,100 data rows.
- `oof_predictions.csv` includes target labels, both fold assignments, finalist
  probabilities and the final recipe's OOF probabilities for both seeds.
- `configs/requirements-research.lock` pins the installed environment. Python
  {meta['python']}; package versions and input/source SHA256 are in
  `reproducibility.json`. No GPU, credentials or external data are required.
- `RESEARCH.md` documents setup, cache behavior and original reproduction. Each
  completed experiment is immutable by name; use a new name for a changed recipe.
- Original source, parameter JSON, methodology report, datasets and prior
  onboarding submission remain unchanged. There are no `/content/` assumptions.

Re-run the selected experiments from their complete `summary.json` settings, or
run `python -m src.confirm`, `python -m src.postprocess --models <frozen names>`
and `python -m src.deliver` in a new output directory. Delivery deliberately
refuses to overwrite an existing primary submission. The exact frozen model
names and all settings are stored in `configs/confirmation_plan.json` and
`configs/selected_solution.json`.

## Limits and next experiments

The requested main audits, five-fold feature ablations, three standard tree
families, custom-model controls, targeted search, smoothing, proxy aggregation,
decision constraints, nested calibration/ensembles and independent-split
confirmation were executed. Large random searches, neural embeddings, external
label recovery and leaderboard probing were not performed. The original
historical `best_params.json` and original simulated-test sampling code were
unavailable; the three requested current project files and all datasets were
available. No required dependency or hardware blocker remains.

Further improvement should first require consistent gains on new held-out
experiments: test adaptive/recursive shrinkage beyond the bounded smoothing
variants, inspect class-2 errors and within-group scarcity, and test proxy bias
calibration as a fully fitted inner-fold transformation. Continuous early-stopping
or hyperparameter optimization must reserve additional inner validation and
rebuild its encodings there. Full recursive empirical-Bayes and exhaustive
conditional-group modeling were not implemented in this bounded run. None is
evidence that 0.800 is reachable. Kaggle evaluation of a validated candidate is
the next external measurement, while preserving the existing 0.676 best.
'''
    (dest/'final_report.md').write_text(final)
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'src').glob('*.py'))+list((ROOT/'configs').glob('*'))+list((ROOT/'tests').glob('*.py')) if p.is_file()}
    (dest/'final_code_manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Reports generated; validated submissions and original-vs-corrected metrics',flush=True)


if __name__=='__main__':
    main()
