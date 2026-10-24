# S2–S6 消融與選模

## 固定控制與外層流程

| name | seed | balanced_accuracy | recall_1 | recall_2 | recall_3 | delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fixed_baseline | 2026.000000 | 0.678116 | 0.747899 | 0.571995 | 0.714452 |  |  |  |
| fixed_baseline | 42.000000 | 0.674304 | 0.740896 | 0.572695 | 0.709320 |  |  |  |
| S2_outer_precision_uncertainty | 42.000000 | 0.672787 | 0.742764 | 0.566628 | 0.708970 | -0.001517 | -0.004100 | 0.001296 |
| S3_outer_balanced_log | 42.000000 | 0.672866 | 0.741830 | 0.575729 | 0.701038 | -0.001438 | -0.004061 | 0.001214 |
| S3_outer_consistent_fold_ensemble | 42.000000 | 0.675587 | 0.745098 | 0.568495 | 0.713169 | 0.001283 | -0.001850 | 0.004382 |
| S4_outer_knn20 | 42.000000 | 0.674187 | 0.736695 | 0.578063 | 0.707804 | -0.000117 | -0.003402 | 0.003104 |
| S5_outer_hgb | 42.000000 | 0.666408 | 0.730159 | 0.569662 | 0.699405 | -0.007895 | -0.013211 | -0.002520 |
| nested_adaptive | 2026.000000 | 0.676133 | 0.745565 | 0.576196 | 0.706637 | -0.001983 | -0.005826 | 0.001961 |
| nested_adaptive | 42.000000 | 0.675938 | 0.742297 | 0.577363 | 0.708154 | 0.001634 | -0.002249 | 0.005368 |
| nested_proxy | 2026.000000 | 0.677027 | 0.749767 | 0.570362 | 0.710953 | -0.001089 | -0.003861 | 0.001926 |
| nested_proxy | 42.000000 | 0.675238 | 0.745565 | 0.570595 | 0.709553 | 0.000934 | -0.001711 | 0.003568 |

兩個最終規則事先限定：proxy 規則只在外層 train 的 inner CV 從 V0–V5 選擇；
adaptive 規則依序在同一允許範圍內選 Proxy、TE、KNN、刪除特徵群與模型。
inner 每步要求超過 0.001 才取代簡單配置。Outer validation 完全不參與這些
選擇。所有 inner training TE/KNN 使用 label-independent KFold cross-fitting；
own-label intervention 測試涵蓋 TE 與 KNN。Inner 選擇用 StratifiedKFold(2)，
Outer 則按任務書 sklearn StratifiedKFold(5,shuffle=True,random_state=42/2026)。
為省算力沒有數百組 grid；S3 representation 與 smoothing 分成小組控制，
S5 的每一個刪除欄群分開跑。每項 scope 與負面結果見完整 CSV。

開發切分最高約 0.69，不等於完整 OOF；所有開發候選均標記 screening。
`S5_proxy_uncertainty` 在原 V0 沒有新不確定性欄，是明確 no-op；應使用
V3 uncertainty_control vs remove_uncertainty 這組配對，而非誤稱零效果已證明無用。

## TE 訓練／推論一致性

普通 TE、同平滑下 balanced TE、同平滑下 balanced-log TE 分開比較；
額外只有 finest group 的 pseudo-count 8/16/32。父群回退及 prior 全是 fit-local。
每折 `te_distribution_fold*.csv` 保存 cross-fitted train 與 query 的均值、變異及 NaN。
Full strategy 用該訓練集的 cross-fitted features 訓練單一模型，query TE 用該
訓練集全部標籤；fold ensemble 的每個組件連模型與 TE 都用同一子集。
沒有把 100% TE 偷換到既有 80% fold 模型。這不宣稱 cross-fitting 的樣本量
差造成的分布差異已完全消失，而是使完整方案可在外層公平模擬。

## 低成本開發切分（非最終 OOF）

| stage | name | balanced_accuracy | recall_1 | recall_2 | recall_3 | status |
| --- | --- | --- | --- | --- | --- | --- |
| S2 | proxy_V0 | 0.687803 | 0.733209 | 0.619048 | 0.711153 | complete |
| S2 | proxy_V1 | 0.689668 | 0.736940 | 0.612512 | 0.719552 | complete |
| S2 | proxy_V2 | 0.687804 | 0.727612 | 0.621849 | 0.713952 | complete |
| S2 | proxy_V3 | 0.689826 | 0.731343 | 0.624650 | 0.713486 | complete |
| S2 | proxy_V4 | 0.689047 | 0.736940 | 0.617180 | 0.713019 | complete |
| S2 | proxy_V5 | 0.689203 | 0.735075 | 0.620915 | 0.711619 | complete |
| S3 | fold_ensemble | 0.686871 | 0.725746 | 0.617180 | 0.717685 | complete |
| S3 | te_balanced | 0.689205 | 0.735075 | 0.633053 | 0.699487 | complete |
| S3 | te_balanced_log | 0.692473 | 0.731343 | 0.642390 | 0.703686 | complete |
| S3 | te_balanced_log16 | 0.692628 | 0.735075 | 0.639589 | 0.703220 | complete |
| S3 | te_balanced_log32 | 0.690605 | 0.735075 | 0.634921 | 0.701820 | complete |
| S3 | te_balanced_log8 | 0.692784 | 0.733209 | 0.641457 | 0.703686 | complete |
| S3 | te_plain | 0.687803 | 0.733209 | 0.619048 | 0.711153 | complete |
| S4 | knn_none | 0.687803 | 0.733209 | 0.619048 | 0.711153 | complete |
| S4 | knn_only | 0.668206 | 0.707090 | 0.602241 | 0.695287 | complete |
| S4 | knn_region_k10 | 0.691693 | 0.735075 | 0.626517 | 0.713486 | complete |
| S4 | knn_region_k20 | 0.690297 | 0.720149 | 0.636788 | 0.713952 | complete |
| S4 | knn_region_k40 | 0.688583 | 0.729478 | 0.626517 | 0.709753 | complete |
| S4 | no_te | 0.570366 | 0.626866 | 0.507937 | 0.576295 | complete |
| S5 | binary_crosses | 0.688270 | 0.731343 | 0.620915 | 0.712552 | complete |
| S5 | blend | 0.691535 | 0.742537 | 0.621849 | 0.710219 | complete |
| S5 | group_aggregate | 0.692936 | 0.736940 | 0.619981 | 0.721885 | complete |
| S5 | hgb | 0.679088 | 0.736940 | 0.596639 | 0.703686 | complete |
| S5 | none | 0.687803 | 0.733209 | 0.619048 | 0.711153 | complete |
| S5 | one_hot | 0.691383 | 0.729478 | 0.632120 | 0.712552 | complete |
| S5 | ordinal | 0.682827 | 0.718284 | 0.614379 | 0.715819 | complete |
| S5 | proxy_raw | 0.687645 | 0.736940 | 0.608777 | 0.717219 | complete |
| S5 | proxy_uncertainty | 0.687803 | 0.733209 | 0.619048 | 0.711153 | complete |
| S5 | remove_uncertainty | 0.687804 | 0.727612 | 0.621849 | 0.713952 | complete |
| S5 | residual | 0.687333 | 0.731343 | 0.595705 | 0.734951 | complete |
| S5 | soft_weights | 0.687492 | 0.731343 | 0.619048 | 0.712086 | complete |
| S5 | te_interactions | 0.690139 | 0.725746 | 0.627451 | 0.717219 | complete |
| S5 | uncertainty_control | 0.689826 | 0.731343 | 0.624650 | 0.713486 | complete |

## 未執行或停止

V6 原樣重現失敗於資料 schema；沒有偽造後續分數。Residual／soft weighting
在 development 無改善即停。KNN 高 fallback 且增益未證實，沒有再放大頻寬搜尋。
未做 propensity weight、全資料 label ranking 選欄、公開榜權重調整或外部標籤搜尋。
當前 Test 無觀測到新 f141，因此沒有執行額外 GroupKFold 壓力測試，未知群回退
已由單元測試覆蓋。沒有通過 gate 的方法不做 full-data candidate refit。
