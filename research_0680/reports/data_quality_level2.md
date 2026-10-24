# S0–S1 資料品質、版本與 Level 2 取證

## 0.679 對應與 S0 狀態

使用者在本次 0.679 脈絡提供 `submission (3).csv`，已保留為
`results/00_baseline/provided/submission_user_0679.csv`。
SHA256：`61d3d73ee03aeb8bacf938fbdd2fa9673e38a7b7016d3aeaddf22c6413ebdc1f`。5,100 筆、ID 與 sample 順序、標籤格式均符合。
這是**使用者回報的 Kaggle 分數**，未連線核對競賽提交歷史，也沒有相應 OOF。

上傳 Python 標頭是 V6，JSON 是 V5。直接、未修改的執行在 base feature
階段失敗：`'c3'` 無法轉為浮點數；尚未開始模型訓練。宣告二元但實為文字的
欄位有 ['f_19', 'f_78', 'f_35']；宣告二元但資料不僅 0/1 的欄位共有 7 個。
沒有用強制轉型、移除欄位或重新編碼把這個失敗隱藏起來。
因此 **0.679 生成流程未重現，S0 此部分是 No-Go**，不是成功重現 V5/V6。

現有可執行程式提供本研究固定參考模型：Gemini core、原始 proxy、低基數
one-hot，fold-local 前處理及標準 regularized quadratic multinomial。
它不是 V4 Standalone，也不是所附 V6。所有後續 ΔBA 都是相對此固定模型，
**不是相對 Kaggle 0.679 的 ΔAccuracy**。V4 Standalone、V7、原始 V5 與
配套 OOF 均未找到；歷史報告分數沒有被當成已重現結果。

## 資料與可觀測分類衝突

Train 15,000、Test 5,100、預測欄位 238，train 計數 2142/4285/8573。
所附競賽 PDF 明示 Test 各 1,700 與 Accuracy；此證據被沿用，Test 標籤未知。
當前 Test 的 f_141 都在 train 出現；群組名稱只作代碼，不宣稱地理單位。
完全相同原始 X 的重複列有 0；清洗後物理特徵
存在不同標籤的簽章群有 96。
這些特徵忽略未觀測因子，**不等於已證實錯標，更不能宣稱資料造假**。

`near_conflicting_pairs_for_review.csv` 列出跨外層折、距離很近但標籤不同的配對。
`label_review_candidates.csv` 結合主欄位、結構與群組資料，僅供人工審查。
真實 Level 2 且固定參考 OOF P(true)<0.1 的列有
146 筆。
舊 CatBoost 是另一個獨立訓練的演算法；其歷史 OOF 折不同，僅作逐列共識診斷，
沒有把這種跨模型共識或全資料排名當作新模型訓練權重。
所有 train.csv 標籤保持不變。

## Proxy 測量品質

下表描述各 proxy，主欄位只作 silver reference，沒有將欄位角色假說當成金標準。

| source | proxy | silver_correlation | peer_correlation | skew | excess_kurtosis | missing |
| --- | --- | --- | --- | --- | --- | --- |
| structure_age | f_168 | 0.317016 | 0.101687 | 0.071987 | 0.008684 | 0.000000 |
| structure_age | f_210 | 0.302479 | 0.091822 | 0.053442 | 0.039398 | 0.000000 |
| structure_age | f_65 | 0.303532 | 0.095375 | 0.028584 | -0.046649 | 0.000000 |
| structure_age | f_118 | 0.308684 | 0.101598 | 0.068241 | 0.025179 | 0.000000 |
| structure_age | f_108 | 0.307668 | 0.100587 | 0.049360 | -0.019535 | 0.000000 |
| structure_age | f_76 | 0.304197 | 0.100911 | 0.060698 | 0.041006 | 0.000000 |
| structure_age | f_64 | 0.302408 | 0.102298 | 0.056201 | 0.132303 | 0.000000 |
| structure_age | f_231 | 0.292998 | 0.095121 | 0.062221 | 0.042612 | 0.000000 |
| storey_count | f_167 | 0.319210 | 0.099782 | -0.006720 | -0.003178 | 0.000000 |
| storey_count | f_115 | 0.313715 | 0.099959 | -0.001559 | -0.004422 | 0.050333 |
| storey_count | f_58 | 0.320907 | 0.103840 | 0.038410 | 0.052047 | 0.000000 |
| storey_count | f_163 | 0.325102 | 0.105407 | 0.007539 | -0.044905 | 0.000000 |
| storey_count | f_96 | 0.315971 | 0.104054 | 0.023765 | 0.053833 | 0.000000 |
| storey_count | f_170 | 0.318152 | 0.104677 | 0.035811 | 0.055026 | 0.000000 |
| footprint_area | f_41 | 0.299367 | 0.093704 | 0.052491 | 0.109939 | 0.034267 |
| footprint_area | f_106 | 0.291272 | 0.090415 | 0.082731 | 0.019433 | 0.000000 |
| footprint_area | f_194 | 0.293069 | 0.089422 | 0.025946 | -0.011438 | 0.000000 |
| f_165 | f_186 | 0.314933 | 0.087300 | 0.002856 | 0.060129 | 0.000000 |
| f_165 | f_151 | 0.299407 | 0.091136 | -0.000246 | -0.018447 | 0.000000 |
| f_165 | f_20 | 0.319095 | 0.091501 | 0.022054 | -0.017490 | 0.000000 |
| f_165 | f_80 | 0.244083 | 0.074666 | 5.574887 | 88.054995 | 0.000000 |

f_80 有顯著重尾；不是所有誤差都可視為同質 Gaussian。silver-reference
條件殘差的 pairwise-complete 相關見 `proxy_residual_correlations_corrected.csv`；
接近零不證明統計獨立。早期未處理缺失的相關與早期錯序的 conditional
permutation 表格保留為歷史中間產物，應使用檔名含 corrected 的修正版。

折內 calibration bias/scale/variance 在各正式模型目錄保存；修復行數與例子在
`repair_counts_and_examples.json`。V5 的可信主欄位完全不變已通過測試。
Proxy-only reconstruction 在外層未見的 silver main 值上評分，不用修復後
仍保留主值的零誤差來假裝去噪成功：

| variant | source | mae | rmse |
| --- | --- | --- | --- |
| V0 | f_165 | 6.447890 | 8.069667 |
| V0 | footprint_area | 274.006622 | 343.305109 |
| V0 | storey_count | 0.714308 | 0.897319 |
| V0 | structure_age | 14.796546 | 18.677220 |
| V1 | f_165 | 3.647684 | 5.269694 |
| V1 | footprint_area | 129.527140 | 192.523702 |
| V1 | storey_count | 0.487916 | 0.737599 |
| V1 | structure_age | 11.988799 | 17.856159 |
| V2 | f_165 | 6.580926 | 8.761044 |
| V2 | footprint_area | 277.371801 | 347.653043 |
| V2 | storey_count | 0.702223 | 0.881926 |
| V2 | structure_age | 15.635340 | 19.704705 |
| V4 | f_165 | 3.686323 | 4.970665 |
| V4 | footprint_area | 120.375865 | 168.055432 |
| V4 | storey_count | 0.447277 | 0.563903 |
| V4 | structure_age | 9.804457 | 13.339762 |

## Train/Test 分布

| feature_set | prior_standardized | roc_auc |
| --- | --- | --- |
| supplied_52_feature_hypothesis | False | 0.581787 |
| supplied_52_feature_hypothesis | True | 0.508299 |
| all_238_no_id_no_label | False | 0.568000 |
| all_238_no_id_no_label | True | 0.498334 |

在兩種特徵集合都接近 chance，支持多數 domain 可辨識性與已知 label shift
有關；不能證明完全沒有 covariate/conditional shift，亦不能證明 Test Level 2
有問題。沒有基於這些結果採用 propensity reweighting。KS、PSI、SMD 與 FDR
在 `distribution_shift.csv`；未看到證據就不加入高方差權重。
