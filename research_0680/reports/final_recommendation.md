# 最終建議：未證實優於使用者的 Kaggle 0.679

## 結論

本次沒有達成「已驗證 Kaggle Accuracy 超過 0.680」；沒有上傳到 Kaggle，也沒有
看到隱藏 Test Label。通過提交 gate 的方法 **0** 個。
保留使用者的 `submission (3).csv`，**不替換目前回報 0.679 的最佳提交**。

完整 nested 規則 OOF BA：seed42 **0.675938**，
seed2026 **0.676133**。它相對同種子新參考模型的
ΔBA 分別為 **+0.001634** 與 **-0.001983**，
方向沒有重現。95% paired CI 分別是
[-0.002249,0.005368] 與
[-0.005826,0.001961]。
這些 conditional bootstrap 不重新訓練，不能當成涵蓋所有選模不確定性的證明。

| rule | passed | reason | delta42 | ci42_low | ci42_high | delta2026 |
| --- | --- | --- | --- | --- | --- | --- |
| proxy | False | unconfirmed: require delta > .001 and CI lower > 0 on BOTH seeds | 0.000934 | -0.001711 | 0.003568 | -0.001089 |
| adaptive | False | unconfirmed: require delta > .001 and CI lower > 0 on BOTH seeds | 0.001634 | -0.002249 | 0.005368 | -0.001983 |

## 必須回答的十個問題

1. **0.679 對應哪版？** 使用者提供的 `submission (3).csv` 已完整保存與雜湊。
   但新 Python 是 V6、JSON 是 V5，原樣執行因錯誤資料型別假設失敗；沒有它的
   OOF。不能把新參考或前次程式的 OOF 冒充該提交的 OOF。
2. **Level 2 具體錯判？** 新固定基準 seed42：2→1 829，
   2→3 1002。結構子群及屋齡分解在 error_attribution 報告與 CSV。
3. **Proxy 分歧是否較高？** 並非各群一致；部分 missingness 有關聯，控制後
   額外 log-loss 改善很小，不能宣稱它就是錯誤根因。
4. **近鄰是否支持可分性有限？** 全域支持部分混合，同區域方向相反；k20
   75% fallback、k40 約99% fallback，不能以單一 entropy 敘事概括。
5. **可能 label noise？** 高信心錯判與衝突配對已列供人工查核，但也可能是
   未觀測因子或罕見正確樣本；沒有改任何 label。
6. **Adversarial AUC？** 全238欄 raw 約0.5680，prior-standardized 約0.4983；
   52欄則約0.5818→0.5083。尚無一致的強額外 shift 證據，非不存在的證明。
7. **Proxy 最有效者？** reconstruction 與 BA 分開記錄。部分 calibration 可減少
   silver-reference 誤差，但 proxy 選擇規則在第二種子沒有改善，不能稱可部署增益。
8. **Balanced Log TE／重訓／KNN 貢獻？** 查看固定外層控制表的逐項 ΔBA。
   開發切分的正向結果沒有變成可信外層增益；full/fold 方案沒有非法 TE 替換。
9. **各類與子群影響？** `level_tradeoffs.csv` 列新增判對／新增判錯，
   `subgroup_metrics.csv` 列結構及群組差異；不能只用 Level 2 Recall 掩蓋其他類損失。
10. **是否真的比0.679方案可信且更強？** **沒有證據證明更強**。本研究的新
    驗證邊界可稽核，但它既沒有重現0.679的生成流程，也沒有通過新候選 gate。

## 保存與重現

- 程式、JSON-compatible YAML、所有成功／失敗紀錄與兩種種子的 outer OOF 已保存。
- 每個正式 run 的 `oof_predictions.csv` 有15,000筆且每個id一次；test probability
  CSV只有推論機率，**不是推薦上傳的 Submission**。小型 screening 檔則明示3,750筆。
- `experiment_manifest.json` 保存資料 hash、版本、seed、fold IDs、允許訓練ID、
  feature list、命令、runtime 和 memory。原始資料、前次結果、舊提交未覆寫。
- `Kaggle_0680_Colab_Reproducible.ipynb` 會在新的輸出目錄執行完整流程並以
  同樣 gate 控制候選產生。Notebook code cells 已做語法檢查；未宣稱已在 Google
  Colab 平台另跑一次。雲端本次已實際完成流程；兩次 fresh refit 的機率一致性
  另見 repeatability.json。
- unit tests 包含 own-label TE/KNN、fold隔離、OOF id、prior normalization、
  unknown-group、可信主值保留、schema 與原始資料 hash。

## 下一步資訊增益最高的工作

先取得**實際產生0.679檔案、可在目前資料schema執行的確切版本與其OOF**。
目前可重現性缺口比追逐0.001更重要。然後用固定、事先保留且未參與選模的
新 validation 設計，測試同平滑 Balanced Log TE 或 sensor-noise calibration。
現有開發切分並未給出改標籤／調公榜權重的正當理由。不要上傳多個未證實版本
探測公榜，也不要聲稱本地0.69開發分數就是Kaggle0.680。
