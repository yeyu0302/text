# 2026 ML&DL Midterm Project — 地震後建築受損等級預測：進階資料清洗、特徵工程與超參數最佳化技術報告

## 1. 專案背景與核心挑戰 (Problem Formulation & Core Challenges)

本專案目標為根據地震後的建築物結構屬性與地理區域特徵（共 238 個特徵變數），預測建築物的受損嚴重程度（三類別分類問題）：
- **Level 1**：輕微或中度受損 (Negligible to moderate damage)
- **Level 2**：嚴重受損 (Substantial to heavy damage)
- **Level 3**：完全倒塌 (Collapse)

### 1.1 關鍵挑戰一：嚴重的類別先驗偏移 (Severe Prior Shift)
根據競賽說明簡報與資料探索，`train.csv`（15,000 筆）與 `test.csv`（5,100 筆）之間存在極為顯著的類別分佈差異：

| 資料集 | 總筆數 | Level 1 筆數 (比例) | Level 2 筆數 (比例) | Level 3 筆數 (比例) | 類別比例關係 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **訓練集 (`train.csv`)** | 15,000 | 2,142 (`14.28%`) | 4,285 (`28.57%`) | 8,573 (`57.15%`) | **1 : 2 : 4** |
| **測試集 (`test.csv`)** | 5,100 | 1,700 (`33.33%`) | 1,700 (`33.33%`) | 1,700 (`33.33%`) | **1 : 1 : 1** |

由於測試集的評估指標為整體分類準確率（Overall Accuracy），且測試集三個類別嚴格各佔 $\frac{1}{3}$（1,700 筆），因此測試集上的 Overall Accuracy 在數學上完全等價於**三類別平均召回率（Macro-Recall / Balanced Accuracy）**：

$$\text{Accuracy}_{\text{test}} = \frac{1}{3} \sum_{k=1}^{3} \text{Recall}_k = \frac{1}{3} \sum_{k=1}^{3} P(\hat{y} = k \mid y = k)$$

若直接以訓練集原生機率 $P_{\text{train}}(y=k \mid x)$ 取 $\arg\max$，模型會嚴重偏向預測多數類 Level 3，導致在平衡測試集上的準確率僅約 `0.50 ~ 0.57`（對應課程 Public Leaderboard Simple Baseline `0.57372`）。

### 1.2 關鍵挑戰二：高維度合成雜訊、異常值污染與隱藏階層結構
除了 5 個具名特徵（`structure_age`, `storey_count`, `footprint_area`, `roofing_kind`, `terrain_slope`）外，其餘 233 個特徵（`f_1` 至 `f_233`）皆為匿名特徵。若直接將全部 238 個特徵餵入梯度提升樹（如 `XGBClassifier`），即使經過超參數搜尋（如 `max_depth=9, colsample_bytree=0.589`），Cross-Validation 分數仍會卡在 `0.65636` 附近，難以進一步突破。

---

## 2. 深度探索性資料分析 (Deep EDA)：破解 238 個特徵的真實構造

透過對全部 238 個欄位進行單變數分佈檢定、卡方獨立性檢定（$\chi^2$ test）、無母數 Kruskal-Wallis 檢定、變異數對均值比（Dispersion Index, $\text{Var}/\text{Mean}$）以及跨特徵相關性矩陣分析，我們完整還原了資料集的生成機制：**全部 238 個特徵中，僅有 52 個是真實訊號特徵，其餘 186 個（佔 78.2%）為人工合成注入的純隨機雜訊特徵。**

### 2.1 238 個特徵的完整解剖分類表

| 特徵類別 | 欄位數量 | 統計特性與鑑定依據 | 具體欄位清單 | 處理策略 |
| :--- | :---: | :--- | :--- | :--- |
| **真實類別特徵 (`object`)** | **8** | 分佈不均勻，與 `label` 之卡方檢定顯著（$p < 10^{-11}$，最高 $\chi^2 = 1972.6$） | `roofing_kind`, `terrain_slope`, `f_13`, `f_45`, `f_62`, `f_69`, `f_105`, `f_209` | 保留；進行條件插補、頻率編碼、One-Hot 與高階交互目標編碼 |
| **合成類別雜訊 (`object`)** | **33** | 每個類別計數呈完美均勻分佈（Uniform），與 `label` 及地理區完全獨立（$p > 0.1$） | `f_3, f_15, f_18, f_19, f_22, f_32, f_35, f_44, f_51, f_56, f_59, f_60, f_66, f_78, f_88, f_92, f_110, f_112, f_114, f_117, f_134, f_140, f_146, f_173, f_191, f_197, f_203, f_205, f_213, f_217, f_219, f_224, f_232` | **完全剔除**（避免稀釋決策樹特徵抽樣） |
| **三層地理階層代碼 (`int`)** | **3** | 嚴格 100% 巢狀階層：`f_90` (11區) $\supset$ `f_109` (110區) $\supset$ `f_141` (945區)；Cramér's $V$ 達 `0.5882` | `f_90` (Macro), `f_109` (Meso), `f_141` (Micro) | 視為**高基數階層類別變數**；建立多尺度階層式 OOF 目標編碼與群組聚合特徵 |
| **核心離散建築屬性** | **5** | 具明確物理意義，但遭注入哨兵異常值（`999`, `99`, `0`, `10x`, 負號反轉）與缺失值 | `structure_age` (屋齡), `storey_count` (樓層), `footprint_area` (面積), `f_165` (高度), `f_119` (住戶數) | 利用下方 21 個代理欄位進行**精準去噪與異常值修復** |
| **隱藏感測代理欄位 (`float`)** | **21** | 為上述 4 大核心數值屬性加上常態雜訊 $\mathcal{N}(0, \sigma^2)$ 之未污染副本（與原欄位相關係數 $r \approx 0.30$） | - `structure_age` 代理 (8欄): `f_168, f_210, f_65, f_118, f_108, f_76, f_64, f_231`<br>- `storey_count` 代理 (6欄): `f_167, f_115, f_58, f_163, f_96, f_170`<br>- `footprint_area` 代理 (3欄): `f_41, f_106, f_194`<br>- `f_165` 代理 (4欄): `f_186, f_151, f_20, f_80` | 計算群組去噪均值與標準差以修復主欄位，並**捨棄個別高雜訊原始欄位** |
| **合成連續亂數雜訊 (`float`)** | **104** | 服從 $\text{Uniform}(0, M)$ 或 $\text{Normal}(\mu, \sigma^2)$，與所有真實特徵及 `label` 無關 | 其餘 104 個連續浮點數欄位（如 `f_4, f_6, f_7, f_8, f_9...`） | **完全剔除** |
| **合成卜瓦松離散雜訊 (`int`)** | **19** | 變異數除以平均值 $\text{Var}/\text{Mean} \approx 1.00$，符合純 $\text{Poisson}(\lambda)$ 隨機過程 | `f_2, f_5, f_25, f_61, f_63, f_67, f_94, f_100, f_107, f_130, f_145, f_147, f_150, f_160, f_184, f_192, f_195, f_228` | **完全剔除** |
| **真實二元結構/用途指標** | **15** | 與地理區、屋頂材質及 `label` 呈極顯著關聯（$p < 10^{-10}$），代表建材與次要用途 | `f_182, f_214, f_199, f_103, f_113, f_124, f_175, f_23, f_104, f_226, f_1, f_75, f_26, f_127, f_48` | 保留；對含缺失值之 `f_103`, `f_175` 進行條件插補並建構建材脆弱度指標 |
| **合成伯努利二元雜訊** | **30** | 與所有真實建築屬性、地理區及 `label` 皆獨立（$p > 0.002$） | 其餘 30 個 0/1 二元欄位（如 `f_11, f_16, f_34, f_36, f_83, f_208...`） | **完全剔除** |

---

## 3. 資料清洗、代理去噪與異常值修復方法 (Data Cleaning & Proxy Denoising)

本研究最關鍵的發現之一，是資料集中隱藏了 **21 個連續代理欄位（Proxy Copies）**。這些代理欄位是在主欄位尚未被注入異常值（如 `999`、`99`、`10倍放大`、`正負號反轉`）與缺失值（`NaN`）之前，由真實數值加上高斯白雜訊生成的：

$$x_{\text{proxy}, j} = x_{\text{true}} + \varepsilon_j, \quad \varepsilon_j \sim \mathcal{N}(0, \sigma_j^2)$$

因此，透過對同一屬性的多個代理欄位取平均（或截斷重尾後的反變異數加權平均），雜訊標準差會以 $\frac{1}{\sqrt{K}}$ 倍率收斂，讓我們能精準識別並還原主欄位中的每一筆異常與缺失資料：

### 3.1 屋齡 (`structure_age`) 的哨兵值 `999.0` 與缺失值修復
- **問題診斷**：`structure_age` 的正常值為 `0, 10, 20, ..., 200`，但含有 71 筆極端哨兵值 `999.0`（代表歷史老舊建築）以及 1,105 筆 `NaN`（佔 `5.5%`）。若不處理，`999.0` 會嚴重扭曲線性與數值特徵尺度。
- **修復方法**：
  1. 計算 8 個代理欄位的去噪均值：
     $$\hat{\mu}_{\text{age}} = \frac{1}{8} \sum_{c \in \{\text{f\_168, f\_210, f\_65, f\_118, f\_108, f\_76, f\_64, f\_231}\}} x_c$$
  2. 新增二元指標特徵 `is_age_999 = 1(structure_age == 999.0)` 與 `is_age_nan = 1(isnan(structure_age))`。
  3. 將 `NaN` 與 `999.0` 的屋齡替換為最接近的 10 的倍數：
     $$\text{clean\_age} = \text{clip}\left(\text{round}\left(\frac{\hat{\mu}_{\text{age}}}{10}\right) \times 10, \; 0, \; 100\right)$$
  4. 建立貝氏收縮平滑屋齡 `age_blended = 0.8 * clean_age + 0.2 * clip(age_proxy_mean, 0, 120)`。

### 3.2 樓層數 (`storey_count`) 的異常編碼 `0` 與 `99` 修復
- **問題診斷**：正常樓層數為 `1 ~ 7` 樓，但資料中混入 34 筆 `0` 樓與 38 筆 `99` 樓。此外，交叉表分析顯示當 `storey_count == 1` 時，其他樓板材質 `f_209` 有極高比例為 `'c2'`（代表單層建築無其他樓板）。
- **修復方法**：
  1. 計算 6 個樓層代理欄位（`f_167, f_115, f_58, f_163, f_96, f_170`）之均值 $\hat{\mu}_{\text{storey}}$（與真實樓層相關係數達 `0.631`）。
  2. 當 `storey_count` 為 `0` 或 `>= 99` 時，若 `f_209 == 'c2'` 且 $\hat{\mu}_{\text{storey}} < 2.2$，直接還原為 `1` 樓；否則以 $\text{clip}(\text{round}(\hat{\mu}_{\text{storey}}), 1, 7)$ 還原。

### 3.3 基地面積 (`footprint_area`) 的「10 倍放大污染」偵測與還原
- **問題診斷**：正常基地面積多介於 `50 ~ 1400`（50 的倍數），但資料中存在 `1500, 2000, 2500, 3000, ..., 12000` 等極端值。比對 3 個面積代理欄位（`f_41, f_106, f_194`）之均值 $\hat{\mu}_{\text{area}}$ 後發現，例如 `footprint_area == 2500` 的樣本，其 $\hat{\mu}_{\text{area}}$ 中位數為 `181.2`；`footprint_area == 3000` 的樣本，其 $\hat{\mu}_{\text{area}}$ 中位數為 `213.7`——證明這些樣本遭到了 **$\times 10$ 倍的隨機乘法污染**！
- **修復方法**：
  當 `footprint_area >= 1500` 且為 `500` 的整數倍時，檢驗除以 10 是否更接近代理均值：
  $$\text{is\_10x} = (\text{raw\_area} \ge 1500) \land (\text{raw\_area} \bmod 500 == 0) \land \left(\left|\frac{\text{raw\_area}}{10} - \hat{\mu}_{\text{area}}\right| < |\text{raw\_area} - \hat{\mu}_{\text{area}}|\right)$$
  若符合條件，則令 $\text{clean\_area} = \frac{\text{raw\_area}}{10}$，完美還原真實基地面積。

### 3.4 建築高度 (`f_165`) 的「正負號反轉」與缺失值修復
- **問題診斷**：`f_165` 的正常值為 `4, 8, 12, ..., 64`（4 的倍數），但含有負值（`-64, -28, -24, -20, -16, -12, -8, -4`）、極端值（`96, 100`）與 `5.3%` 的 `NaN`。比對高度代理欄位（`f_186, f_151, f_20, f_80`）發現，`raw_f165 = -24.0` 的代理均值為 `+27.15`、`raw_f165 = -16.0` 的代理均值為 `+16.04`，證明負值完全是**正負號反轉污染（Sign-Flip Corruption）**。
- **修復方法**：
  1. 由於 `f_80` 含有少數重尾離群值（標準差 `20.56` 高於其餘三欄的 `15.7`），先將 `f_80` 截斷至 `[-40, 80]`，再以精確度加權計算高度代理均值：
     $$\hat{\mu}_{\text{height}} = 0.28 (x_{\text{f\_186}} + x_{\text{f\_151}} + x_{\text{f\_20}}) + 0.16 \, \text{clip}(x_{\text{f\_80}}, -40, 80)$$
  2. 當 `-48 <= f_165 < 0` 時，直接取絕對值 `clean_height = -f_165` 還原。
  3. 當 `f_165` 為 `NaN`、`-64` 或 `>= 96` 時，以 $\text{clip}(\text{round}(\hat{\mu}_{\text{height}} / 4) \times 4, \; 4, \; 64)$ 插補。

---

## 4. 領域特徵工程與階層式經驗貝氏目標編碼 (Feature Engineering & Hierarchical Target Encoding)

### 4.1 三層地理階層挖掘 (`f_90 -> f_109 -> f_141`)
透過條件唯一性檢定，我們證實三個整數欄位構成嚴格的**三層地理行政區階層**：
- **第一層大區 (`f_90`)**：共 11 個大區（$\chi^2 = 4,799.0, \; \text{Cramér's } V = 0.4000$）
- **第二層中區 (`f_109`)**：共 110 個中區，每個 `f_109` 100% 唯一隸屬於一個 `f_90`（$\chi^2 = 7,080.4, \; \text{Cramér's } V = 0.4858$）
- **第三層微區/村落 (`f_141`)**：共 945 個微區，每個 `f_141` 100% 唯一隸屬於一個 `f_109`（$\chi^2 = 10,379.8, \; \text{Cramér's } V = 0.5882$）

由於 `f_141` 的數值 `1..945` 只是名目代碼（Nominal ID），直接作為數值輸入決策樹無法有效表達各村落的震度與地質差異。我們設計了 **Out-of-Fold (OOF) 多尺度階層式經驗貝氏目標編碼（Hierarchical Empirical Bayes Target Encoding）**：

1. 在每個 Cross-Validation Fold 內部（僅使用該 Fold 的訓練樣本以杜絕 Target Leakage），先計算母層區域 `parent` 的平滑後驗機率 $\hat{P}(y=k \mid \text{parent})$。
2. 將子層區域 `child`（如 `f_141`）的樣本統計量向其所屬的母層區域 `parent`（如 `f_109`）進行貝氏收縮：
   $$\hat{P}_m(y=k \mid \text{child}) = \frac{n_{\text{child}} \cdot \bar{y}_{\text{child}, k} + m \cdot \hat{P}(y=k \mid \text{parent})}{n_{\text{child}} + m}$$
3. 同時針對 `f_141` 採用多尺度平滑參數 $m \in \{3, 10, 25\}$，讓模型能同時掌握大樣本村落的精確受損率與小樣本村落的穩健區域趨勢。

### 4.2 高階「地理區域 $\times$ 結構材質」交互目標編碼
在地震工程學中，同一震度區內「泥漿石造（`f_182=1`）」與「鋼筋混凝土/磚造（`f_214=1` 或 `roofing_kind='Reinforced concrete roof'`）」的倒塌率截然不同。因此我們建構了關鍵交互組合並進行階層式 OOF 目標編碼：
- `inter_141_182`：微區 `f_141` $\times$ 泥漿石造指標 `f_182`（向母層 `f_141` 收縮）
- `inter_141_roof`：微區 `f_141` $\times$ 屋頂材質 `roofing_kind`（向母層 `f_141` 收縮）
- `inter_141_f62`：微區 `f_141` $\times$ 地基類型 `f_62`（向母層 `f_141` 收縮）
- `inter_109_182_roof`：中區 `f_109` $\times$ `f_182` $\times$ `roofing_kind`
- `inter_struct_full`：完整結構系統簽章（`roofing_kind + f_62 + f_69 + f_209 + f_182 + f_214`）及其與區域 `f_90`、`f_109` 的交互項

### 4.3 零洩漏村落背景特徵 (Transductive Regional Context Aggregations)
因為群組統計特徵僅使用特徵 $X$（完全不使用標籤 $y$），我們將 `train` 與 `test` 合併計算每個區域（`f_90`, `f_109`, `f_141`）的整體建築環境統計量：
- 區域建築總數對數：$\log(1 + N_{\text{region}})$
- 區域平均屋齡、平均樓層、平均面積、平均高度、區域內 `f_182`（泥漿石造）與 `f_214`（水泥磚造）普及率
- **個體相對差異特徵**：單棟建築相對於所屬村落平均的屋齡差（`clean_age - f_141_mean_clean_age`）、高度差與面積差，用以捕捉「該建築是否比同村落其他建築更老舊或更高聳」。

### 4.4 結構力學物理比例與脆弱度指標
- **單層平均挑高 (`height_per_storey`)**：$\frac{\text{clean\_height}}{\max(\text{clean\_storey}, 1)}$
- **高寬細長比 (`slenderness`)**：$\frac{\text{clean\_height}}{\sqrt{\max(\text{clean\_area}, 50)}}$（細長比愈高，地震側向力矩愈大）
- **建築體積指標 (`volume_proxy`)**：$\log(1 + \text{clean\_area} \times \text{clean\_height})$
- **脆弱砌體綜合評分 (`weak_masonry_score`)**：$2.0 \cdot \text{f\_182} + 1.5 \cdot \text{f\_124} + 1.0 \cdot \text{f\_175} - 2.0 \cdot \text{f\_214} - 2.5 \cdot \text{f\_199}$

---

## 5. 模型架構、抗過擬合超參數調整與多模型融合 (Modeling & Hyperparameter Tuning)

### 5.1 為什麼去噪與目標編碼後，樹模型的超參數必須重新調整？
在原本未清洗的 238 特徵資料集上，你搜尋到的參數為 `max_depth=9, min_child_weight=8, reg_lambda=0.57`。但在剔除 186 個雜訊欄位並加入階層式 OOF 目標編碼後，超參數配置需做以下關鍵調整：

| 超參數名稱 | 原始設定 (`best_params.json`) | 最佳化設定 (`best_params_optimized.json`) | 調整原理說明 |
| :--- | :---: | :---: | :--- |
| **`max_depth`** | `9` | **`5 ~ 6`** | 原本需要深樹是因為被迫對數值化的 `f_141` (`1..945`) 做數十次切割；改用階層 OOF 目標編碼後，過深的樹會過度記憶單一訓練樣本的 OOF 數值，降至 `5~6` 泛化能力最強 |
| **`min_child_weight`** | `8` | **`16 ~ 25`** | 提高葉節點最小海森權重（Hessian weight），防止決策樹對少數異常建築產生細碎分裂 |
| **`reg_lambda` (L2)** | `0.571` | **`5.0`** | 大幅加強葉節點輸出權重的 L2 正規化，使預測機率更平滑，有利於後續先驗偏移校正 |
| **`reg_alpha` (L1)** | `0.186` | **`1.5`** | 促使樹分裂時自動稀疏化次要特徵 |
| **`learning_rate`** | `0.04` | **`0.03`** | 搭配 `n_estimators=800` 與 `early_stopping_rounds=50`，以更小步長穩定收斂 |
| **`colsample_bytree`** | `0.589` | **`0.65`** | 剔除 186 個雜訊欄位後，特徵集皆為高訊號特徵，可適度提高特徵抽樣比例並鼓勵不同階層編碼輪流參與分裂 |

### 5.2 異質多模型融合 (Heterogeneous Ensembling)
為了兼顧非線性邊界切割與平滑機率校準，管線結合了兩大類截然不同的模型家族：
1. **直方圖梯度提升決策樹家族 (`FastHistGBDT` / `XGBoost` / `LightGBM` / `CatBoost`)**：擅長捕捉高階結構物理特徵與地理區域的非線性交互作用。
2. **Böhning 二次上界預條件牛頓多類別邏輯斯模型 (`SplineSoftmax`)**：將連續與目標編碼特徵進行 Robust IQR 標準化與二階非線性展開 $[z, 0.15 z^2]$，透過固定海森上界預條件迭代求解全域最佳解，提供極為平滑且抗過擬合的對數機率估計。

---

## 6. 先驗偏移校正與 1:1:1 精確基數匹配 (Prior Shift & Exact Cardinality Matching)

### 6.1 第一階段：貝氏先驗偏移校正 (Bayes Prior Shift Calibration)
設模型在訓練集分佈 $\pi_{\text{train}} = (0.1428, 0.2857, 0.5715)$ 下預測的後驗機率為 $P_{\text{train}}(y=k \mid x)$。根據貝氏定理，在特徵條件分佈 $P(x \mid y=k)$ 不變的假設下，對應於 1:1:1 平衡測試集 $\pi_{\text{test}} = (\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$ 的校準機率為：

$$P_{\text{cal}}(y=k \mid x) = \frac{P_{\text{train}}(y=k \mid x) \cdot (\pi_{\text{train}, k})^{-\alpha} \cdot w_k}{\sum_{j=1}^{3} P_{\text{train}}(y=j \mid x) \cdot (\pi_{\text{train}, j})^{-\alpha} \cdot w_j}$$

其中 $\alpha = 1.0$ 為標準貝氏先驗反轉，而 $w = (1.0, 0.89, 1.15)$ 為透過 Out-of-Fold 巢狀搜尋所得到的決策邊界微調乘子。

### 6.2 第二階段：1:1:1 精確基數匹配 (Exact Cardinality Constrained Assignment)
由於已知 `test.csv`（5,100 筆）的三個類別筆數**嚴格等於 `1,700 : 1,700 : 1,700`**，單純對每筆樣本獨立取 $\arg\max_k P_{\text{cal}}(y_i = k \mid x_i)$ 往往會因為各模型機率校準誤差，導致某類別被預測了 1,850 筆而另一類別僅有 1,550 筆。

我們將測試集預測建模為一個**受基數約束的最大對數似然指派問題（Cardinality-Constrained Maximum Likelihood Assignment）**：

$$\max_{\hat{y}_1, \dots, \hat{y}_N \in \{1,2,3\}} \sum_{i=1}^{N} \log P_{\text{cal}}(y_i = \hat{y}_i \mid x_i) \quad \text{s.t.} \quad \sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = k) = 1700, \;\; \forall k \in \{1, 2, 3\}$$

由拉格朗日對偶理論（Lagrange Duality），此最佳指派等價於尋找一組對偶平移乘子 $\boldsymbol{\lambda} = (\lambda_1, \lambda_2, \lambda_3)$，使得：

$$\hat{y}_i = \arg\max_{k \in \{1,2,3\}} \left( \log P_{\text{cal}}(y_i = k \mid x_i) + \lambda_k \right)$$

剛好滿足三類別筆數皆為 1,700 筆。`optimize_pipeline.py` 中的 `enforce_exact_cardinality()` 透過座標上升法（Coordinate Ascent）快速收斂出對偶乘子 $\boldsymbol{\lambda}$，並對邊界上最後 1~3 筆樣本以最小邊際成本交換（Minimum-Margin Greedy Swap）完成精確對齊。

---

## 7. 5-Fold Cross-Validation 消融實驗結果 (Ablation Study)

以下為固定隨機種子（`seed=42`）之 5-Fold Stratified Cross-Validation 各階段實測表現：

| 實驗階段 | 方法與特徵配置 | 5-Fold Balanced Acc | Level 1 Recall | Level 2 Recall | Level 3 Recall | 相較原始最佳模型提升 |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline 0** | 課程 Public LB Simple Baseline | `0.57372` | — | — | — | `-8.26%` |
| **Baseline 1** | 原始 238 特徵 XGBoost Baseline (`best_params.json`) | `0.65493` | — | — | — | `-0.14%` |
| **Baseline 2** | 原始 238 特徵 XGBoost 調參最佳值 (`max_depth=9`) | `0.65636` | — | — | — | `基準 (0.00%)` |
| **Stage 1** | 剔除 186 個雜訊 + 代理去噪修復 + 階層式地理與結構交互 OOF 目標編碼 (`SplineSoftmax`) | `0.66867` | `0.7026` | `0.6917` | `0.6117` | **`+1.23%`** |
| **Stage 2** | Stage 1 + OOF 類別決策乘子微調 (`w = [1.0, 0.89, 1.15]`) | `0.67295` | `0.7194` | `0.6553` | `0.6441` | **`+1.66%`** |
| **Stage 3** | 精煉 55 個核心與目標編碼特徵之直方圖梯度提升樹 (`FastHistGBDT`, `max_depth=5`) | `0.67847` | `0.7488` | `0.5958` | `0.6908` | **`+2.21%`** |
| **Stage 4** | 異質多模型加權融合 (`0.60 FastHistGBDT + 0.40 SplineSoftmax`) | **`0.67874`** | `0.7292` | `0.6558` | `0.6512` | **`+2.24%`** |
| **Stage 5** | 1:1:1 平衡測試抽樣模擬 + 精確基數匹配 (`1700 : 1700 : 1700`) | **`0.67972`** | `0.729+` | `0.656+` | `0.654+` | **`+2.34%`** |

---

## 8. 檔案清單與執行方式 (Usage Guide)

1. **`optimize_pipeline.py`**：完整端到端資料清洗、代理去噪、階層目標編碼、多模型訓練與 `1700:1700:1700` 精確基數匹配腳本。
   - **執行方式**：將 `train.csv` 與 `test.csv` 置於同一目錄下，執行：
     ```bash
     python3 optimize_pipeline.py
     ```
   - **自動擴充支援**：若你的本機或 Colab 環境已安裝 `xgboost`、`lightgbm`、`catboost`，腳本內的 `try_fit_external_gbdt_models()` 會自動偵測並使用去噪後的最佳超參數訓練這三個模型，與內建的 `FastHistGBDT` 及 `SplineSoftmax` 進行五模型加權融合。
2. **`best_params_optimized.json`**：整理好的 52 個真實訊號特徵分類清單、去噪後推薦的 XGBoost 超參數以及後處理參數設定。
3. **`submission.csv`**：由上述最佳化融合管線所產出的 5,100 筆測試集預測結果（Level 1、Level 2、Level 3 各精確為 1,700 筆），可直接上傳至競賽平台評分。
