# Feature-family ablation

All eight comparisons use the exact same class-wise shuffled seed-42 outer
folds, the same fixed 800-tree XGBoost budget, prior correction and fold-local
preprocessing. Raw values remain alongside repaired values except the explicit
repaired-only variant. `all_raw` is an available-parameter reference; the missing
historical `best_params.json` prevents an exact reproduction of the old 238-feature
hyperparameter search. All requested project/data files are present.

| name | features_min | features_max | balanced_accuracy | fold_std | increment_vs_previous |
| --- | --- | --- | --- | --- | --- |
| B00_all_raw | 471.000000 | 471.000000 | 0.640423 | 0.009140 |  |
| B01_selected | 93.000000 | 93.000000 | 0.649293 | 0.007816 | 0.008870 |
| B02_proxy | 110.000000 | 110.000000 | 0.649799 | 0.010449 | 0.000506 |
| B03_hierarchy | 113.000000 | 113.000000 | 0.672126 | 0.005542 | 0.022327 |
| B04_combined | 130.000000 | 130.000000 | 0.672592 | 0.009432 | 0.000466 |
| B05_structural | 138.000000 | 138.000000 | 0.673720 | 0.006140 | 0.001128 |
| B06_groups | 177.000000 | 177.000000 | 0.676872 | 0.007667 | 0.003152 |
| B07_full | 221.000000 | 221.000000 | 0.671426 | 0.005316 | -0.005446 |

The largest observed gain is from hierarchical encoding, not proxy repair alone.
Group statistics help the structural+hierarchical recipe on this split; adding
all material/group target interactions hurts this XGBoost configuration. These
are conditional development comparisons, not proof of physical causation.
Paired conditional bootstrap intervals are in `ablation_bootstrap.csv`.

## Aggregation, repair preservation and shrinkage

| name | family | aggregate | raw_mode | smoothing | balanced_accuracy | fold_std |
| --- | --- | --- | --- | --- | --- | --- |
| D00_median | full | median | both | 1.000000 | 0.673721 | 0.005611 |
| D01_trimmed | full | trimmed | both | 1.000000 | 0.672554 | 0.003662 |
| D02_repaired_only | full | mean | repaired | 1.000000 | 0.673799 | 0.006251 |
| D03_smooth_weak | groups | mean | both | 0.300000 | 0.675433 | 0.007184 |
| D04_smooth_strong | groups | mean | both | 3.000000 | 0.675005 | 0.007618 |

Mean/weighted/median/trimmed reconstruction uses apparently valid source values,
with a 70/30 fit/holdout split for any learned weights or affine calibration.
It does not reveal the true values of naturally missing/anomalous observations.
Complete individual proxy correlations are in `proxy_correlations.csv`.
The reconstruction diagnostic's trimmed mean removes one value from each end.
The CV trimmed variant requires at least four proxies before trimming; its
three-proxy area group retains the mean rather than a single middle sensor.

| source | method | holdout_rows | masked_mae | masked_rmse | rounded_exact_rate |
| --- | --- | --- | --- | --- | --- |
| structure_age | mean | 4235 | 15.014605 | 18.889626 | 0.208973 |
| structure_age | median | 4235 | 17.609448 | 22.027742 | 0.167887 |
| structure_age | trimmed | 4235 | 15.571826 | 19.584350 | 0.199528 |
| structure_age | weighted | 4235 | 15.015019 | 18.890089 | 0.209917 |
| storey_count | mean | 4258 | 0.704481 | 0.883209 | 0.429544 |
| storey_count | median | 4258 | 0.804483 | 1.009083 | 0.382339 |
| storey_count | trimmed | 4258 | 0.743320 | 0.933077 | 0.408877 |
| storey_count | weighted | 4258 | 0.704604 | 0.883430 | 0.429075 |
| footprint_area | mean | 4308 | 270.386038 | 338.627125 | 0.059424 |
| footprint_area | median | 4308 | 311.766250 | 391.953855 | 0.050836 |
| footprint_area | trimmed | 4308 | 311.766250 | 391.953855 | 0.050836 |
| footprint_area | weighted | 4308 | 270.387180 | 338.652554 | 0.060585 |
| f_165 | mean | 4246 | 6.443444 | 8.273530 | 0.196420 |
| f_165 | median | 4246 | 6.910888 | 8.619895 | 0.177108 |
| f_165 | trimmed | 4246 | 6.910888 | 8.619895 | 0.177108 |
| f_165 | weighted | 4246 | 6.657286 | 8.372634 | 0.182996 |
| f_165 | legacy_weighted_clipped | 4246 | 6.566195 | 8.210639 | 0.182760 |

Controlled corruption of valid records through the actual repair code (not
through an oracle that knows which value was corrupted):

| source | corruption | n | mae | exact_rate |
| --- | --- | --- | --- | --- |
| structure_age | missing | 1000 | 12.450000 | 0.281000 |
| storey_count | zero | 1000 | 0.553000 | 0.493000 |
| storey_count | sentinel | 1000 | 0.553000 | 0.493000 |
| footprint_area | times10 | 1000 | 12.150000 | 0.987000 |
| f_165 | sign_flip | 1000 | 0.000000 | 1.000000 |
| f_165 | missing | 1000 | 6.048000 | 0.185000 |

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
Among 186 omitted features, 0
pass exploratory FDR 0.05. This supports the pruning hypothesis but cannot prove
that all omitted columns are independent noise or that interactions contain no
signal. The screen uses all training labels, is explicitly exploratory, and was
**not** used to select features or claim independent validation. The historical
52-feature list itself may retain earlier full-data selection optimism.

The manually chosen vulnerability coefficients are retained hypotheses. No
physical validation or coefficient optimality is claimed.
