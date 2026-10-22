# Earthquake damage optimization: final research report

## Outcome

The pipeline was audited, the target-encoding leakage was reproduced and fixed,
controlled ablations and targeted follow-ups were executed, and two validated
5,100-row submissions were generated. The primary uses **C01_full_cat, C07_core_spline**,
probability method **prior**, and an exact
**accuracy** transportation objective with 1,700 rows/class.

**The public Kaggle benchmark remains the user-confirmed 0.676. No new submission
was uploaded or scored on Kaggle.** Local CV cannot confirm an improvement over
that score. No independently measured 0.700+ or 0.800 score was demonstrated. Competitor scores near 0.800 cannot be reproduced or
explained from leaderboard claims alone; no external labels were used to pursue them.

| seed | balanced_accuracy | accuracy | macro_f1 | log_loss | recall_1 | recall_2 | recall_3 | fold_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | 0.678820 | 0.668067 | 0.646877 | 0.739234 | 0.730159 | 0.639440 | 0.666861 | 0.005694 |
| 2026 | 0.676642 | 0.666800 | 0.645407 | 0.737548 | 0.729225 | 0.632205 | 0.668494 | 0.006026 |

Seed 42 was used for development/model selection. Candidate model families and
decision rules depend on development data only; seed 2026 tests the frozen recipe.
All rows are out of fold, but repeated CV reuses the same dataset and is not a
new independent population sample. Full benchmark metrics include Accuracy,
Macro F1, three Recalls, log loss and fold variation. `log_loss` is on the original
imbalanced validation population after the reported probability transformation;
`raw_metrics` retains uncorrected-probability log loss for base models.

## Reproduction and leakage correction

| method | balanced_accuracy | accuracy | macro_f1 | log_loss | recall_1 | recall_2 | recall_3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original two-model ensemble (LEAKED) | 0.678705 | 0.663600 | 0.645584 | 0.734129 | 0.729225 | 0.655776 | 0.651114 |
| Corrected nested two-model ensemble | 0.675592 | 0.664267 | 0.643415 | 0.736736 | 0.724556 | 0.641774 | 0.660446 |

The reported historical 0.678741285 was nearly reproduced at
0.678704983, difference
-0.000036303. FastHistGBDT's
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

| name | balanced_accuracy | fold_std |
| --- | --- | --- |
| B00_all_raw | 0.640423 | 0.009140 |
| B01_selected | 0.649293 | 0.007816 |
| B02_proxy | 0.649799 | 0.010449 |
| B03_hierarchy | 0.672126 | 0.005542 |
| B04_combined | 0.672592 | 0.009432 |
| B05_structural | 0.673720 | 0.006140 |
| B06_groups | 0.676872 | 0.007667 |
| B07_full | 0.671426 | 0.005316 |

| name | family | model | balanced_accuracy | fold_std |
| --- | --- | --- | --- | --- |
| F00_prior | nested_calibration | prior | 0.678820 | 0.005694 |
| F00_multipliers | nested_calibration | multipliers | 0.678702 | 0.005922 |
| F00_stack | nested_calibration | stack | 0.678312 | 0.006257 |
| F00_weighted | nested_calibration | weighted | 0.678041 | 0.005233 |
| C01_full_cat | full | cat | 0.677651 | 0.004843 |
| C08_core_standard_spline | core | spline_standard | 0.677027 | 0.004120 |
| B06_groups | groups | xgb | 0.676872 | 0.007667 |
| C03_native_cat | full | cat | 0.675550 | 0.008882 |
| D03_smooth_weak | groups | xgb | 0.675433 | 0.007184 |
| C10_xgb_less_shrinkage | groups | xgb | 0.675121 | 0.007412 |
| D04_smooth_strong | groups | xgb | 0.675005 | 0.007618 |
| C07_core_spline | core | spline | 0.674893 | 0.005811 |
| C02_full_logistic | full | logistic | 0.674691 | 0.008132 |
| C11_xgb_deeper | groups | xgb | 0.674652 | 0.006006 |
| C05_core_xgb | core | xgb | 0.674072 | 0.005805 |
| D02_repaired_only | full | xgb | 0.673799 | 0.006251 |
| D00_median | full | xgb | 0.673721 | 0.005611 |
| B05_structural | structural | xgb | 0.673720 | 0.006140 |

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

| method | balanced_accuracy | accuracy | macro_f1 | log_loss | recall_1 | recall_2 | recall_3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| plain | 0.656585 | 0.705600 | 0.662535 | 0.654036 | 0.607376 | 0.558926 | 0.803453 |
| prior | 0.678820 | 0.668067 | 0.646877 | 0.739234 | 0.730159 | 0.639440 | 0.666861 |
| multipliers | 0.678702 | 0.671533 | 0.645059 | 0.734433 | 0.750233 | 0.596499 | 0.689374 |
| weighted | 0.678041 | 0.670200 | 0.646475 | 0.735442 | 0.732960 | 0.623104 | 0.678059 |
| rank | 0.668838 | 0.630333 | 0.606801 | 0.854815 | 0.827731 | 0.565228 | 0.613554 |
| stack | 0.678312 | 0.673733 | 0.646785 | 0.727041 | 0.742764 | 0.597666 | 0.694506 |

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

| seed | objective | mean | std |
| --- | --- | --- | --- |
| 42 | accuracy | 0.678993 | 0.006105 |
| 42 | argmax | 0.678588 | 0.007479 |
| 42 | log | 0.679739 | 0.006142 |
| 2026 | accuracy | 0.678779 | 0.010656 |
| 2026 | argmax | 0.676504 | 0.009471 |
| 2026 | log | 0.679308 | 0.010687 |

The sparse LP is globally optimal for its supplied scores and quotas; this says
nothing about the correctness/calibration of its probabilities. Probability-sum
and log objectives differ, and the old greedy method is neither reliably exact
nor globally optimal. No public-leaderboard feedback was used to choose them.

## Uncertainty

| comparison | delta | ci95_low | ci95_high | draws |
| --- | --- | --- | --- | --- |
| seed42 fixed ensemble minus B00_all_raw | 0.038397 | 0.030654 | 0.046065 | 1000 |
| seed42 fixed ensemble minus B03_hierarchy | 0.006694 | 0.000818 | 0.012607 | 1000 |
| seed2026 fixed ensemble minus baseline hierarchy | 0.007900 | 0.002103 | 0.013892 | 1000 |
| seed42 ensemble minus C01_full_cat | 0.001169 | -0.003109 | 0.005024 | 1000 |
| seed2026 ensemble minus C01_full_cat | 0.000469 | -0.003731 | 0.004438 | 1000 |
| seed42 ensemble minus C07_core_spline | 0.003927 | 0.000190 | 0.007469 | 1000 |
| seed2026 ensemble minus C07_core_spline | 0.004667 | 0.001009 | 0.008558 | 1000 |
| seed42 frozen ensemble minus corrected legacy ensemble | 0.003228 | -0.000974 | 0.007429 | 1000 |

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
  is not claimed superior. The two files differ on 122 rows.
- Counts: `{"best_submission.csv": {"1": 1700, "2": 1700, "3": 1700}, "alternative_submission.csv": {"1": 1700, "2": 1700, "3": 1700}}`. Both preserve all IDs and their original order,
  have only labels 1/2/3, columns id,label, no missing values and 5,100 data rows.
- `oof_predictions.csv` includes target labels, both fold assignments, finalist
  probabilities and the final recipe's OOF probabilities for both seeds.
- `configs/requirements-research.lock` pins the installed environment. Python
  3.12.14; package versions and input/source SHA256 are in
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
