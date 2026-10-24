# Level 2 錯誤歸因

固定新參考的標準 sklearn 五折 OOF：BA 0.674304；
Recalls 0.740896/0.572695/0.709320。
Level 2→1 **829** 筆，Level 2→3 **1002** 筆。
屋頂、屋齡、群組及 f182×storey 的確切錯判分解見
`01_data_forensics/subgroup_metrics.csv`；群組太小時不宣稱穩定因果關係。

## 近鄰混合並非單一方向

| scope | k | wrong_entropy | correct_entropy | delta_ci_low | delta_ci_high | mean_distance | mean_effective_n | fallback_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f_109 | 10 | 0.925800 | 0.956727 | -0.037175 | -0.024916 | 1.756481 | 9.555030 | 1.000000 |
| f_109 | 20 | 0.905951 | 0.953835 | -0.055370 | -0.040460 | 1.920463 | 18.719752 | 1.000000 |
| f_109 | 40 | 0.882244 | 0.949207 | -0.075755 | -0.058560 | 2.114308 | 36.305153 | 1.000000 |
| global | 10 | 0.905764 | 0.882245 | 0.017376 | 0.029288 | 1.128907 | 9.834990 | 1.000000 |
| global | 20 | 0.897678 | 0.869931 | 0.021243 | 0.033483 | 1.216118 | 19.599923 | 1.000000 |
| global | 40 | 0.890411 | 0.864381 | 0.019830 | 0.031854 | 1.309778 | 39.058116 | 1.000000 |
| region | 10 | 0.939032 | 0.961701 | -0.027585 | -0.018057 | 2.165375 | 8.916057 | 0.224267 |
| region | 20 | 0.912360 | 0.954203 | -0.049127 | -0.035071 | 2.069078 | 18.121735 | 0.750667 |
| region | 40 | 0.882104 | 0.949207 | -0.075867 | -0.058753 | 2.118359 | 36.263297 | 0.992800 |

全域近鄰中，Level 2 錯判列的 entropy 較高；同群組近鄰則相反。
所以不能概括說所有 Level 2 錯判都因高熵。k=20 的區域方法約 75% 回退，
k=40 幾乎全數回退，細群樣本稀少限制其增量。這是可觀測可分性有限的
部分證據，並非錯標證明。距離只含標準化物理欄位、類別 mismatch 與二元
指標，不把 945 個群組 ID 當連續距離；所有 query 的標籤來源均在外層 fit。

## Proxy 分歧與錯判的關聯

| feature | wrong_mean | correct_mean | fdr_q | controlled_cv_logloss_gain |
| --- | --- | --- | --- | --- |
| structure_age_proxy_std | 50.525490 | 49.797487 | 0.426382 | 0.000430 |
| structure_age_proxy_missing | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| structure_age_main_gap | 1.431777 | 5.674376 | 0.592534 | -0.000149 |
| storey_count_proxy_std | 1.880904 | 1.863607 | 0.796000 | 0.000269 |
| storey_count_proxy_missing | 0.060623 | 0.040750 | 0.035299 | 0.000177 |
| storey_count_main_gap | 0.060565 | 0.118643 | 0.426382 | -0.000027 |
| footprint_area_proxy_std | 436.489944 | 436.266742 | 1.000000 | -0.000038 |
| footprint_area_proxy_missing | 0.037138 | 0.035045 | 1.000000 | -0.000036 |
| footprint_area_main_gap | 18.166415 | 26.217582 | 0.592534 | -0.000621 |
| f_165_proxy_std | 13.129228 | 13.352491 | 1.000000 | 0.000102 |
| f_165_proxy_missing | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| f_165_main_gap | -0.005167 | -0.730068 | 0.102909 | -0.000388 |

部分缺失指標有描述性關聯，但控制建築、群組與屋頂後的額外診斷 log-loss
收益很小。這些關聯不是因果，也不是最終模型的無偏改善證據。
兩組 bootstrap mean-difference CI 在原 CSV；conditional 分析屬探索性。

## 殘差與樣本權重

| method | mse_baseline | mse_modified | balanced_mse_baseline | balanced_mse_modified | note |
| --- | --- | --- | --- | --- | --- |
| residual | 0.168142 | 0.165436 | 0.176090 | 0.176779 | Independent development holdout, not final OOF. Damage labels unchanged. |
| soft_weights | 0.168142 | 0.168545 | 0.176090 | 0.176215 | Independent development holdout, not final OOF. Damage labels unchanged. |

殘差模型確實在允許訓練集內產生 cross-fitted baseline p2，再擬合 residual，
沒有拿全域 OOF 的其他折來訓練當前驗證折校正器。它降低普通 MSE 卻未改善
balanced MSE／BA，故沒有保留；soft weighting 亦未採用。
沒有自行改標、移除驗證列或用 Test Label 評估。

匿名欄位 f51/f5/f125 的 corruption-flag conditional permutation 與隨機負控制
見 corrected CSV。這不是 damage-label 重要性，更不是將這三欄加入候選的理由。

## 新增判對與新增判錯

| name | seed | level | newly_correct | newly_wrong | net_correct |
| --- | --- | --- | --- | --- | --- |
| nested_adaptive | 2026 | 1 | 35 | 40 | -5 |
| nested_adaptive | 2026 | 2 | 149 | 131 | 18 |
| nested_adaptive | 2026 | 3 | 108 | 175 | -67 |
| nested_adaptive | 42 | 1 | 42 | 39 | 3 |
| nested_adaptive | 42 | 2 | 137 | 117 | 20 |
| nested_adaptive | 42 | 3 | 112 | 122 | -10 |
| nested_proxy | 2026 | 1 | 23 | 19 | 4 |
| nested_proxy | 2026 | 2 | 84 | 91 | -7 |
| nested_proxy | 2026 | 3 | 71 | 101 | -30 |
| nested_proxy | 42 | 1 | 24 | 14 | 10 |
| nested_proxy | 42 | 2 | 54 | 63 | -9 |
| nested_proxy | 42 | 3 | 53 | 51 | 2 |

個別屋頂、區域與建材的增損見研究根目錄 `results/subgroup_metrics.csv`。
不能只看 Level 2 Recall，而忽略 Level 1/3 或稀有群的新增錯誤。
