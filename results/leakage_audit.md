# Leakage and correctness audit

The historical public Kaggle score **0.676 is user-confirmed**. Nothing in this
local audit establishes a new Kaggle score. The historical method's validation
score cannot be treated as leakage-free.

## Confirmed cross-fold label leakage

`optimize_pipeline.py::build_clean_and_engineered_features` computes a full
training feature matrix before the outer model CV. Training row B's encoding
excludes B's encoding fold but includes labels from fold A. When the downstream
model holds out A, its training features therefore depend on A's labels.
Excluding a validation row's own label is insufficient.

The global prior used in `compute_te` is also computed from all training labels,
including outer validation and the inner held-out row. Global priors and parent
shrinkage must be fitted on the same permitted label subset as the child counts.

An actual intervention (`src/leakage_probe.py`, seed 42, fixed outer fold 0)
rotated only the 3,001 outer validation labels. This changed **all 11,999 outer
training rows**, in 112 feature columns, with maximum absolute change 6.9985.
The corrected builder produced identical training and validation features under
the same intervention. Evidence: `leakage_probe.json`.

## Corrected implementation

`FeatureBuilder.fit_transform` accepts training labels only. Inside each outer
fold it fits four inner stratified folds. Each inner training encoding is fitted
from that inner fit partition's own prior, parent and child counts. Outer
validation/test encodings use only the outer training partition. Missing groups
fall back to fitted parent probabilities, then the fitted class prior.

All conditional imputation maps, categorical vocabularies, frequencies, and
group statistics fit outer training X only. No test X is used for the main
experiments' feature fitting. Row-local algebra needs no fitted statistics.
The supplied 52-feature selection is a fixed historical hypothesis, not a newly
validated causal feature list; it may have prior full-data selection optimism.
The all-feature and selected-raw controls quantify its combined effect rather
than performing new supervised selection on outer validation data.

The `legacy_nested` audit control instead keeps the original **transductive
unlabeled X aggregation**, to isolate the encoding correction from a broader
feature redesign. Its labels are nested; its group context is transductive.
Its final implementation reuses the original feature function with explicit
fit-local priors for raw and prior-normalized encodings, invoked separately within each outer fold. That
control is never confused with the fully inductive main feature builder.

## Scaling, stopping, model selection, calibration

- Original SplineSoftmax scales using all CV rows; original histogram bins use
  train+test X. The corrected custom-model adapter fits scale/quantile estimates
  on outer training rows only. Their original math is reused and source hashes
  are recorded; both adapters execute only the designated prediction fold and
  never receive holdout labels.
- Tree budgets are fixed from the supplied configuration before evaluation.
  No outer validation labels are passed into fit or early stopping.
- Seed-42 ablations/hyperparameter comparisons are development estimates; choosing
  their best score produces selection optimism. Additional fixed independent
  seed comparisons test stability but are not a Kaggle guarantee.
- Tuning multipliers, weights or stacking on other folds' OOF predictions can
  produce a second-level leak: those other models trained on the current fold's
  labels. `src/postprocess.py` instead builds a separate calibration holdout
  **inside each outer training partition**, trains auxiliary models without its
  labels, chooses the decision rule there, and evaluates only on untouched outer
  validation predictions. This also makes calibration's training-size shift
  explicit. Fixed averaging requires no fitted labels.
- Balanced evaluation subsets use labels only to simulate a known balanced test
  population; their individual labels never enter decision optimization.

## Cardinality solver

The original function's 400 dual updates and 50 greedy moves neither prove global
optimality nor guarantee counts. `cardinality_audit.json` records a 12-row example
with correct counts but suboptimal log objective (-6.297258 vs -5.856513), and a
300-row tied example returning 50/250/0 instead of 100/100/100.

The replacement solves a sparse transportation LP with HiGHS dual simplex. Each
row sends one unit to one of three classes, with fixed column capacities. Its
constraint matrix is totally unimodular, so an optimal basic solution is
integral. Solver success, integrality, row constraints and class counts are
checked. Tests compare both probability-sum and log-probability objectives with
exhaustive enumeration, and test ties and invalid counts. Expected accuracy
maximizes **sum of calibrated probabilities**; log likelihood optimizes a
different objective. Exact optimality of an assignment does not imply accurate
or calibrated input probabilities, nor a better Kaggle score.

## Scope

The original spline optimizer's claimed global convergence is also unproven.
Its scalar curvature factor 0.24 is not an upper bound on the full multiclass
softmax covariance: at uniform three-class probabilities its nonzero eigenvalues
are 1/3. Finite probability outputs and CV are empirical checks, not an optimizer
certificate. An additional standard multinomial-optimizer control preserves its
clipped quadratic feature map with C=1/35, but uses an unpenalized intercept
instead of the original tiny intercept penalty. It is separately named and not
silently substituted into historical results.

The supplied PDF explicitly shows 1,700 test observations per class. No external
labels or datasets were downloaded; id is excluded. No Kaggle submission or
leaderboard probing is performed. Hyperparameter results, reconstruction trials
and uncertainty estimates are local evidence only.
