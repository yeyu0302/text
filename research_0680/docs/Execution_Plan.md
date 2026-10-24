# Codex 執行任務書｜地震建築受損分類：從 Kaggle 0.679 挑戰 0.680+

> 日期：2026-10-24｜語言：繁體中文｜用途：**可直接貼給 Codex 的完整執行規格**。
>
> **目標**：在不洩漏標籤、不利用 Kaggle 公榜過擬合的前提下，找出可重現、實際有機會讓測試 Accuracy 超過 **0.680** 的新方案。**目前已知實際 Kaggle 0.679**（使用者回報；請以當時提交記錄核對）。不保證一定突破；禁止捏造改善幅度或聲稱未實測的 Kaggle 成績。

---

## 0. Codex 角色與立即行動

你是負責設計、程式實作、執行實驗、診斷錯誤、產出可重現提交檔的資深 ML Research Engineer。

**不要只回傳建議；請實際執行並保存可重現產物。**

1. 自動掃描專案與檔案，定位 `train.csv`、`test.csv`、`sample_submission.csv`、現有最佳 Kaggle submission、各版 OOF、原始 `.py` / `.ipynb`、Gemini V5 程式及報告。不要假設當前環境有先前記載的絕對路徑；路徑若存在才使用。
2. 先建立 `results/00_baseline/` 與 `reports/`，重現固定基準，核對 metric、標籤、ID、模型輸出、5 折種子。**舊版最佳提交不得覆寫**。
3. 依優先級執行低成本診斷與受控單項消融。通過獨立驗證的少數候選才執行完整 5-fold 訓練。
4. 統一輸出 CSV、圖表、Markdown 報告、可在 Colab 重現的 Notebook、最終 `submission_candidate_*.csv`。
5. **每項實驗都回報成功或失敗**；不要只挑最佳分數、不要宣稱用 Test Label 進行驗證。

## 1. 資料、競賽與必須核實的事實

- 三分類：Level 1、2、3。訓練 15,000 筆、測試 5,100 筆。約 238 個預測欄位；`id` 不得作為未經驗證的預測特徵。
- Train 類別計數：`[2142, 4285, 8573]`；據既有競賽材料，Test 三類各 1,700 筆。如果競賽評分與 Test 分布的證據不足，先核實，並註明假設。
- **若 Test 真正是 1:1:1 且評分為 Accuracy，Test Accuracy 數值等於 Balanced Accuracy；Train 上的 Balanced Accuracy 是合理代理指標，但並非 Kaggle 隱藏分數。**
- 歷史本地基準：單一 Spline Ord V4 約 `0.67531` OOF Balanced Accuracy；V7 約 `0.67566`（特徵篩選不完全外層隔離，屬探索）；Gemini V5 報告聲稱多模型 OOF `0.68368`；**不可直接拿不同方法、不同切分、不同選模流程數字判斷進步**。
- 現有背景參考檔（如果存在）：`SplineOrd_V4_Standalone.py`、`Single_SplineOrd_V4_NestedTE_Colab.ipynb`、`conditional_feature_selection_v7.py`、`methodology_report (2).md`、`Kaggle_0680_Improvement_Roadmap.md`、`spline_ord_v4_results/oof_predictions.csv`。
- 以競賽真實規則核對 Sample Submission 的欄順序與 `id` 集合；只輸出 `id,label`，標籤合法且不可有 NaN。

**優先測試「實際已提交到 0.679 的檔案」的完整產生流程**。若原程式不可得，報告該缺口，不要把別的 OOF 冒充它的 OOF。

## 2. 驗證防洩漏與防過擬合規格（硬性要求）

### 2.1 資料切分

- 固定 `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` 為主要比較；另用 `random_state=2026` 作穩定性複驗（僅最有希望的 1–2 項）。
- **最終模型選擇使用 nested protocol**：外層 5 折只做評估；特徵篩選、參數選擇、模型選擇、殘差模型或可學習的修正規則，必須只在外層訓練集的內層 CV 完成。不要先看全資料 OOF 選特徵後再以同一份 OOF 宣稱無偏提升。
- 省算力可用：階段 A 固定一個 **development / screening split** 做篩選；階段 B 將選擇規則（不是選出來的全資料特徵清單）放入外層 5-fold 重新評估。若成本不足，只能報告「探索性」。
- 若 `f_141` 可能在真實盲測中有新區域，額外做 `GroupKFold` 或留出部份區域的壓力測試；但不能拿它取代原 Kaggle 隨機切分情境。註明分組名稱只是資料推測，未證實實際地理含義。

### 2.2 Target Encoding、KNN 與代理修復

- 對驗證列計算任何 Label-based features，只能由該折訓練列擬合。
- 對訓練列必須使用**內層 OOF target encoding / KNN**；不可把該列自身的 `label` 計入自己的特徵。
- 針對最終 Test 可在完成所有模型選擇後以完整 15,000 筆訓練標籤估算 TE，但**必須整套重訓與推論方案一致**，不得只把 fold 模型的 80%-train TE 換成 100%-train Test TE 然後宣稱一定改善。
- 無監督的統計量（Proxy 偏差／變異估計、量尺、缺值插補、類別聚合等），評估時也要在該外層訓練折擬合，避免 transductive leakage；若使用 train+test 的無監督資料，另外標示為 transductive 實驗。
- `id` 必須全程 join 校驗；所有 OOF 特徵／標籤／機率須保持同一列對齊。

### 2.3 評估與停止條件

對每個候選記錄：

- OOF `balanced_accuracy_score`、普通 Accuracy、各類 Recall、Level 2→1 / Level 2→3 的筆數、confusion matrix、log loss、模型預測 Level1/2/3 計數。
- **使用 paired bootstrap**（以觀測列重抽，≥1,000 次；固定 seed）估計相對相同基準 OOF 的 ΔBalanced Accuracy 區間，並以重複 seed／外層切分確認方向；不可將 post-selection 的區間當成最終無偏統計證據。
- 比較 Level 2 新增判對數與 Level 1/3 新增判錯數，不可只追求 Level 2 Recall。
- 實驗預算：先挑選每階段 3–5 個合理組合，絕不做數百組盲調；每次僅變動一個方法或一組相關變數。
- 若 OOF 微幅提升 ≤0.001 或 CI 明顯跨 0，列為「未證實」，不要自動替換既有最佳 Kaggle 提交。
- 評分網站查詢有限；只有通過篩選的 **最多 2–3 個候選** 才建議使用者上傳，絕不透過反覆公榜提交擬合測試答案。

---

## 3. P0：先找出卡住 Level 2 的錯誤來源（**先診斷，不訓練大量模型**）

### H1 — 相似建築卻不同受損等級：可觀測特徵不足？

**問題**：Level 2 Recall 偏低是分類器能力不足，還是相似建築原本就有不同標籤（不可辨別或缺少重要變數）？

執行：

1. 以清洗後的屋齡、樓層、面積、高度、屋頂、結構二元指標、區域代碼建立適合混合資料的距離；數值以 fold-train robust scaler 標準化，類別用 mismatch / one-hot distance，避免 945 個區域代碼被當連續量。
2. KNN `k=[10,20,40]`、**排除自己**、並比較全域 KNN、同 `f_109`、同 `f_141`（群組太稀少時回退至父群）。鄰居標籤統計要由外層訓練折取得。
3. 輸出每筆：`neighbor_entropy`, `neighbor_level2_rate`, `neighbor_label_disagreement`, `neighbor_effective_n`, `knn_distance`, `oof_pred`, `oof_p2`, `true_label`。
4. 對真實 Level 2 的正確／錯判組，比較近鄰熵分布、可信距離、Proxy 雜訊；加入 bootstrap CI 或非參數檢定，但不要把相關性視為因果。
5. 對 Level 2→1、Level 2→3 各自拆解；觀察不同屋齡與結構下高熵比例。

**判讀**：若低距離近鄰標籤仍高度混雜，支持「可觀測特徵可分性有限」的假說，但**不能直接宣稱資料造假或標籤錯誤**；須做 Proxy 修復敏感度分析。

### H2 — 是否還存在模型未學到的非線性訊號？

1. 使用既有真正的 OOF 機率 `p1,p2,p3` 建立 Level 2 殘差 `r2=1(y=2)-p2`；區分正負殘差與邊界樣本。
2. 用 **nested / cross-fitted** 小型 HistGradientBoostingRegressor、HistGradientBoostingClassifier 或 shallow XGBoost 分別預測殘差或 `Y=2`，輸入結構、區域、Proxy 不確定性，及原模型 OOF 機率。
3. 評估新模型對**未見驗證折**的 residual MSE、Log Loss 改善，最後看原三分類的 Balanced Accuracy 是否真的改善。
4. 對 `roofing_kind × age`、`f_182 × floors`、`f_109 × age` 做局部誤差圖；沒有獨立改善，就停止建第二階段校正器。

**判讀**：如果能穩定預測 residual 而非只在訓練集有效，表示現有表示或模型形式可能不足。

### H3 — Train/Test 分布偏移是否不只 1:2:4 vs 1:1:1？

1. 訓練輕量 **adversarial classifier** 分辨 Train 或 Test，報告 stratified CV ROC-AUC；排除 `id` 與任何直接洩漏來源的欄位。
2. 用 Train 類別重加權至 1:1:1（訓練集權重 `w_k=(1/3)/pi_k`），再對抗式驗證，區分**已知 Label Shift** 與可能額外的 Covariate Shift。注意 Test 真實類別未知，無法真正以 Test Label 分層。
3. 比較各 `f_90/f_109/f_141`、建材、樓層、屋齡、Proxy 缺失指標的 Train/Test 分布；出 KS / PSI / standardized mean difference，並進行多重檢定控制。
4. 只針對可重現且與錯誤相關的分布差異研究 reweighting，將 propensity weight clip 防止高方差。

**限制**：Train/Test 分類器很準不代表一定有 test conditional drift，也不能證明 Test 上的 Label 2 有問題。

---

## 4. P1：優先排查「看似有用但可能是假的」資料

### H4 — 21 個 Proxy 的誤差結構不是同質 Gaussian noise

核心欄位與 Proxy 清單：

- `structure_age`：`f_168,f_210,f_65,f_118,f_108,f_76,f_64,f_231`（8）
- `storey_count`：`f_167,f_115,f_58,f_163,f_96,f_170`（6）
- `footprint_area`：`f_41,f_106,f_194`（3）
- 建築高度 `f_165`：`f_186,f_151,f_20,f_80`（4）

**不得直接把 Proxy 當作真實測量：欄位與物理量的對應是既有探索推論，需要驗證。**

A. **資料取證／污染診斷**（優先無監督、低成本）

1. 每個 Proxy 計算分位數、偏度、重尾、與同群其他 Proxy 的相關、與清潔主欄位的相關、缺失率、重複值、離散化／截斷現象。
2. 建立清楚的主欄位可信樣本集合：去除 `age=999`、`floors∈{0,99}`、已辨識的面積 10x 與高度符號反轉、及其他確定的不合理哨兵；主欄位不是金標準，僅作 **silver reference**。
3. 用折內 robust regression `proxy_j ~ true_like_main` 估計 `bias`, `scale`, `residual variance`，檢查是否為 **有偏感測、比例縮放、群組依賴偏誤、異質變異**；必要時用 Huber loss / RANSAC 作健壯回歸。
4. 計算 `proxy_std`, `median_absolute_deviation`, `IQR`, `max-min`, `missing_count`, `main_proxy_gap`、各欄偏差正規化殘差；檢查是否和 Level 2 錯判顯著相關（控制屋齡、樓層、區域、結構）。
5. 若有相互獨立的雜訊可試 **reliability-weighted mean** `z_hat=sum(w_j z_j)/sum(w_j)`，權重約 `1/(sigma_j²+epsilon)`；再比較折內線性因子模型／PCA(1)／穩健中位數／median of means／quantile weighted aggregate。
6. 建立 `z_hat`, `z_uncertainty`, `main_minus_z_hat`, `main_trust_flag`, `proxy_consistency`，但把「測量誤差變數」是否能提高分類能力交給外層 CV 判定。

B. **必做消融**（使用相同單一基準模型）

- V0：目前已有 `mean / median / std`、原始異常修復方式。
- V1：保留主欄位＋異常標記，但不用任何 Proxy。
- V2：折內估計各 Proxy 偏差／scale 後 **精度加權去噪**。
- V3：V2 ＋不確定性特徵（spread / disagreement / missingness）。
- V4：折內穩健因子／PLS（若用 label 的 PLS 須在折內進行；預設無標籤 PCA）。
- V5：強制不修復可信主欄位，只有異常或缺失才由可信代理補上。

**決策**：若 V2/V3 在內層 CV 及外層評估均改善，才保留。若不同 Proxy 彼此弱相關，先確認是否根本 **不是同一真實屬性的量測**，不可盲目精度加權。

### H5 — 有沒有標錯的 Level 2 或不可靠樣本？

1. 用真正外層 OOF 機率，列出 Level 2 高信心錯判 `P(true_label)<0.1` 或真實 Level 2 但 `p1/p3` 高的候選。
2. 查找**完全相同**或**近乎相同**的清洗後物理特徵、不同 label 的樣本；輸出 overlap 與跨折重複檢查。
3. 若有另一種**真正獨立訓練**的模型（小型 GBDT 即可），比較跨模型高信心錯判一致性；若無則別宣稱 cross-model consensus。
4. 以 confident learning 思路輸出 `suspicion_score` 排名供人工查核；**不可擅自將 label 改成模型預測**。
5. 僅在內層 CV 探索小比例 soft sample weighting / 少量剔除疑似異常，並在外層純淨未干預驗證折評估；不要根據同一份 OOF 排除樣本後再在原 OOF 宣稱提高。
6. 若輕微樣本權重調整沒有穩定改善，放棄 label 清洗，避免懲罰真正少見但正確的 Level 2。

### H6 — 被判作合成雜訊的匿名欄位，有沒有「污染指標」價值？

- 核對人工雜訊欄位的統計鑑別證據，不要把一次 `p-value` 當成真偽證明。
- 測試匿名欄位是否與缺失／哨兵／Proxy 間異常相關；如果只和 `label` 在單次開發集偶然相關，應剔除。
- 調查 `f_51`、`f_5`、`f_125` 過去在 holdout 有不穩定跡象；除非 nested CV 穩定，不能直接納入。
- 在明確定義候選欄位組合後做 **conditional permutation importance** 或 cross-fitted residual screening；控制多重比較，報告 negative controls（隨機置換的欄位）。

---

## 5. P2：最有機會提升分數的低成本模型改造（必須逐項消融）

### H7 — Target Encoding 的算式與 Train/Test 一致性

**目前重要但尚未可靠證實的提升點**。

1. 重現原本完整 Nested Target Encoding，檢查每個 `f_90`, `f_109`, `f_141`、交互組合的 train/validation/test feature distribution（平均、變異、缺失、平滑後父層回退）。
2. 比較普通 TE、**prior-balanced TE**、`log(TE+eps)` / class log-odds，與只保留核心 3 層區域。所有 label-conditioned stats 必須 fold-safe。
3. 對 `f_141` 小樣本區設定 empirical Bayes 平滑與 parent-level 回退，探索少量 `smoothing ∈ {8,16,32}`，不要大型 grid。
4. 測試兩種真正公平的最終訓練策略：
   - **Fold ensemble**：模型與 Test TE 均由各訓練折的 80% 資料擬合，再平均 Test 機率。
   - **Full retrain**：內層 cross-fitting 為全 15,000 筆建出 train TE，單一最終模型擬合全 train；Test TE 用完整 15,000 筆 train label。用外層模擬這個完整 pipeline 才能比較。
5. **不得**僅在現有 Fold 模型推論時直接替換為 100% TE 並當作「一致性已改善」；如執行，必須標示 train–infer feature distribution shift 的實驗。

### H8 — Within-region Kernel KNN 是否提供比 TE 額外的新訊號？

1. 距離只用物理欄位：去噪屋齡、樓層、面積、建材、屋頂等；數值與類別權重由內層訓練決定。
2. 同 `f_141` 優先，不足者回退 `f_109` / `f_90`；KNN kernel bandwidth 只測 2–3 組，重點檢查樣本少造成的不穩定。
3. 估計 KNN 類別比例加 Bayesian prior shrinkage、有效鄰居數、距離與密度指標；訓練列用 out-of-fold KNN，驗證列只用 training labels。
4. 比較 `baseline`、`+TE`、`+KNN`、`+TE+KNN`；檢查 KNN 與既有 TE 的增量資訊是否穩定。
5. 檢查 same-region KNN 是否因群組稀疏而頻繁 fallback；若 945 小群帶來高方差，停止該路線。

### H9 — 精簡特徵與較適合的模型歸納偏差

1. **Group ablation**：依次移除 `one-hot`, `te_interactions`, `proxy_raw`, `proxy_uncertainty`, `group_aggregate`, `binary_crosses`；同一種子與模型超參數，比 paired ΔBA。
2. 小規模比較單一 **Spline Ord** 與單一 **HistGradientBoosting / XGBoost**，原則是相同折、相近計算預算；不預設 GBDT 一定比較好。
3. 只在證明錯誤互補且有穩定增量後，測試最多 2 個模型**固定權重**的簡單機率平均；融合權重只能從內層 OOF 挑選，不可用 Kaggle 公榜調權重。
4. 如果真正能重現 Gemini V5，做單因子 ablation，區分 Full-TE、Balanced-TE、KNN、特徵篩選、多深度 GBDT 與融合各自貢獻，不能將報告 13 模型總分歸功於其中一項。

---

## 6. P3：如果前面仍未提高，再檢查這些可能性

- **原始主欄位與 Proxy 的缺失並非隨機**（MNAR）：用缺失旗標與條件分布分析，避免只做平均插補。
- **異常值修復誤傷正常值**：面積 10 倍、負高度、年齡 999、樓層 0/99 的規則逐條檢查，輸出可疑列並做消融。
- **分層模型**：對 `f_90` 大區域共享主模型與地區偏差截距；小組嚴重收縮，避免直接在 945 群建獨立模型。
- **Ordinal vs nominal multiclass**：如果 Level 2 與其他類別並非簡單單調邊界，測試一次同樣特徵下的 regularized multinomial logistic / GAM，評估模型歸納偏差是否更適切。
- **Fold variance**：若每折分數高度不一致，定位特定區域與結構群組的泛化失敗。
- **Test prediction prevalence constraint**：測試集據稱各類 1,700 並不代表必須強制各預測類別各 1,700；只有在內層模擬驗證明確改善 Accuracy 才可試受約束分配，且避免利用公榜反覆調整。

---

## 7. 實驗順序與算力配置（執行此表，不要同時全做）

| Stage | 方法 | 預期計算 | Go / No-Go |
|---|---|---|---|
| **S0** | 資料稽核、重現最佳已知基準、正確 OOF 對齊 | 低 | baseline 可完整重現，指標無矛盾 |
| **S1** | H1/H3/H4/H5：近鄰標籤、Train/Test shift、Proxy／Label 取證 | 低至中 | 找到具有一致性、統計支持的可修正訊號 |
| **S2** | H4：Proxy 加權／不確定性 3 組消融 | 中 | 至少改善驗證子集且方向可重現 |
| **S3** | H7：Balanced Log TE、Full retrain / Fold TE 一致性 | 中 | 外層 ΔBA 穩定為正 |
| **S4** | H8：區域 Kernel KNN（最多 3 設定） | 中 | 相對 TE 有增量，不只是補償洩漏 |
| **S5** | H9：特徵群組消融、單一 GBDT 對照 | 中 | 2 seeds 方向一致，OOF 誤差減少 |
| **S6** | 只重跑最佳 1–2 個方法的嚴格 nested CV、生成提交 | 中至高 | 通過預設效能與穩定性檢查 |

**節省 Colab 時間**：對只使用既有 OOF 的「診斷」不重訓模型；代理模型和 TE 的單因子消融先用 3-fold／固定 small split 篩選，再對最佳候選正式 5-fold（必須如實標示篩選偏差）；設定 thread limit、cache 無標籤特徵、保存每折預測。建議預設 `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`，避免超量平行。

## 8. 產出目錄與檔案合約

```text
project/
├── src/
│   ├── audit_data.py
│   ├── build_proxy_features.py
│   ├── leakage_safe_te.py
│   ├── conditional_knn.py
│   ├── error_attribution.py
│   ├── train_baseline.py
│   └── run_experiments.py
├── configs/
│   └── experiments.yaml
├── results/
│   ├── 00_baseline/
│   ├── 01_data_forensics/
│   ├── 02_proxy_ablation/
│   ├── 03_te_ablation/
│   ├── 04_knn_ablation/
│   └── 05_final_candidates/
├── reports/
│   ├── data_quality_level2.md
│   ├── error_attribution_level2.md
│   ├── ablation_summary.md
│   └── final_recommendation.md
├── notebooks/
│   └── Kaggle_0680_Colab_Reproducible.ipynb
├── tests/
│   ├── test_no_target_leakage.py
│   ├── test_feature_alignment.py
│   └── test_submission_schema.py
└── README.md
```

必要每次輸出：

- `experiment_manifest.json`（Git commit 若可得、dataset hash、seed、fold IDs、特徵清單、套件版本、執行指令）。
- `fold_metrics.csv`, `oof_predictions.csv`（`id,label,p1,p2,p3,pred`）, `test_probabilities.csv`（`id,p1,p2,p3`）, `confusion_matrix.csv`, `subgroup_metrics.csv`。
- `paired_delta_bootstrap.csv`、`ablation_summary.csv`（包含失敗與退出原因）。
- 最終可上傳 `submission_candidate_<experiment_id>.csv`；沿用 sample submission 的原順序；`id` 不得重複，僅 1、2、3 類別。
- 在 `reports/final_recommendation.md` 說明何者為「可信改善」、何者只是「探索」、哪種實驗還需要真正 Kaggle 提交才能判斷。

## 9. 必要的防呆測試（Codex 自動撰寫、執行）

1. **TE 洩漏測試**：在訓練折單筆交換其 `label`，該筆的 cross-fitted 訓練 TE 不應直接因其自身 label 改變；其餘點因訓練依賴可能改變，故測試範圍僅對該列自標籤依賴。
2. **KNN 洩漏測試**：每筆 train KNN neighbor 列表不能含自己，且所有鄰居標籤來源均在允許訓練折內；僅去除自己不等於 complete cross-fitting，應測試折間隔離。
3. **OOF 完整性**：每個 train `id` 剛好一筆 OOF，外層驗證標籤未參與 fitted transforms、hyperparameter selection、model training。
4. **先驗變換**：平衡機率正規化後各列總和為 1，沒有 NaN/Inf，Test 無未知群造成崩潰。
5. **測試集機率**：`0<=p<=1`、各列加總≈1、所有 `id` 與 sample submission 一一對應。
6. **真偽取證**：Proxy 的主欄位與 proxy 名單、缺值／污染假說不可視為已確證生成機制；任何自動修復都要記錄行數與範例。
7. **可重現**：固定種子同環境二次跑得相同或數值近似結果；報告訓練耗時與記憶體。

## 10. 最後報告必須回答的 10 個問題

1. 目前 Kaggle 0.679 對應哪份提交／哪份程式？有沒有找到該版 OOF？
2. Level 2 的 2→1、2→3 的確切 OOF 錯判數與主要結構子群？
3. Level 2 錯判樣本是否有較高的 Proxy 分歧、哨兵值或缺失率？控制建築／區域後還有差別嗎？
4. KNN 近鄰標籤混合程度是否足以支持「特徵可分性有限」？對距離／k 是否穩定？
5. 是否有可能的 label noise？有哪些候選，為何不能直接改標籤？
6. Adversarial CV AUC 原始 vs 先驗校正後是多少？有沒有超出已知 label shift 的證據？
7. 哪一種 Proxy 去噪方式最有效？是 BA 改善，還是僅改善 reconstruction proxy MSE？
8. Balanced Log TE、Full retrain、Kernel KNN 分別貢獻多少外層 ΔBA？
9. 最佳候選對 Level 2 Recall、Level 1/3 Recall、各區域與建材子群帶來什麼變化？
10. 目前**最強經驗證方法**是否真的比 0.679 對應方案可信？如果沒有，請直接說沒有，不要宣稱已達 Kaggle 0.680。

## 11. Codex 可以直接執行的精簡指令

> 請將本檔視為規格，先查找所有現有 CSV / code / OOF 與真正 Kaggle 0.679 對應的提交版本；先執行 S0、S1，將 Level 2 錯判與 Proxy 測量誤差／疑似 label noise／Train-Test 分布偏移的實證結果寫入 Markdown。然後依照 S2、S3、S4、S5 做**每次只改一項**的低成本消融，嚴格避免 Target Encoding 和近鄰 Label 洩漏。通過獨立驗證者才重跑正式 5-fold／nested evaluation，保留所有 OOF、各類 Recall、逐折 BA、測試機率與消融紀錄；最終產生 1–2 個格式正確的 Kaggle Submission、完整可重現 Colab、README 與最終推薦報告。禁止只根據公榜反覆調權重、禁止宣稱未實測的 Kaggle 分數、禁止自行改掉疑似錯誤標籤。若結果都不改善，請誠實說明瓶頸，並指出下一個資訊增益最大的實驗。

---

**實務建議**：最優先投入的是 `Proxy 品質稽核 + 屋齡／樓層／面積／高度的折內可靠度加權修復 + Balanced/Log Target Encoding + 訓練／測試編碼一致性`，並使用近鄰標籤混合、獨立殘差實驗決定後續模型方向。這些方法比繼續盲調 Level 2 權重更有機會提供新訊號，但只有真正外層驗證與 Kaggle 實測可以確認是否跨過 0.680。
