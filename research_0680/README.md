# Kaggle 0.680 研究：S0–S6 執行結果

本次沒有找到通過驗證門檻的新方法，**沒有產生新的推薦 Submission**。
使用者提供的 0.679 原提交保持不變。完整結論在
[reports/final_recommendation.md](reports/final_recommendation.md)。

[在 Colab 開啟](https://colab.research.google.com/github/yeyu0302/text/blob/research-0680-20261024/research_0680/notebooks/Kaggle_0680_Colab_Reproducible.ipynb)
／[直接查看 Notebook](notebooks/Kaggle_0680_Colab_Reproducible.ipynb)

## 重點

- 新固定參考 BA：seed42 0.674304、seed2026 0.678116。
- Nested proxy：0.675238／0.677027；nested adaptive：0.675938／0.676133。
- 相對同種子基準的增益方向未重現，CI 跨過0，故兩個規則均 No-Go。
- 上傳 Python 標頭 V6／JSON V5，且原樣執行因將 `c3` 轉數值而失敗。
  不能把本研究的 OOF 說成已知 Kaggle 0.679 的 OOF。
- `test_probabilities.csv` 是推論機率資料，不是可直接推薦上傳的提交檔。
- Label suspicion 是人工查核清單，不是已確認錯標；沒有修改任何標籤。

## 原始資料與目錄

程式會從既有儲存庫根目錄定位 `train.csv`、`test.csv`、`sample_submission.csv`，
不假設一定有 `/content`，也不建立 Git worktree。請保留整個儲存庫的根目錄結構。
`results/00_baseline/provided/` 保存此次使用者提供的三個檔案；所附程式僅作
原樣失敗取證，不是後續模型的無條件可信基準。

主要輸出：

- `results/experiment_log.csv`、`ablation_summary.csv`：包含成功、失敗、探索性 scope。
- `results/nested_rule_comparison.csv`：兩種種子的正式 nested 規則。
- 每個正式 run 的 `oof_predictions.csv`：id,label,p1,p2,p3,pred,fold，15,000筆。
- `results/01_data_forensics/`：近鄰、疑似錯標、Proxy、對抗驗證、負控制與圖表。
- `results/05_final_candidates/candidate_gate_decisions.csv`：候選通過／不通過原因。
- `reports/`：資料品質、Level2 歸因、消融與最終建議（繁體中文）。

## 環境與單一命令重現

Python 3.12、CPU 即可，實測使用單執行緒，無外部 API 金鑰需求。
從儲存庫根目錄執行：

```bash
uv venv --python python3.12 /workspace/.venvs/kaggle-0680
uv pip install --python /workspace/.venvs/kaggle-0680/bin/python -r research_0680/configs/requirements.txt
source /workspace/.venvs/kaggle-0680/bin/activate
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/kaggle0680-matplotlib

# 獨立結果目錄，避免覆寫既有產物；首次指定一個尚未使用的新目錄。
export RESEARCH0680_RESULTS="$PWD/research_0680/reproduction_results"
export RESEARCH0680_REPORTS="$PWD/research_0680/reproduction_reports"
python -B -m research_0680.src.run_all
```

若要再次獨立重訓，請換一組新的 `RESEARCH0680_RESULTS/REPORTS` 路徑。
預設路徑是 `research_0680/results`，已完成的正式模型折有快取，不會再次訓練。
完整重新運行可能花數十分鐘；每折都保存機率與 inner 選擇紀錄。
Colab Notebook 也使用獨立輸出目錄；若 GitHub 是私人 repo，可在瀏覽器先
Code → Download ZIP，再於 Colab 上傳，不要在聊天或 Notebook 貼 Token。

## 驗證邊界

外層嚴格使用 sklearn StratifiedKFold(5,shuffle=True,seed=42/2026)。
Proxy calibration、imputation、scale、類別統計均在每個允許訓練集擬合。
TE/KNN 訓練列使用 label-independent KFold(3) cross-fitting，避免自身標籤
變更時折分配跟著改變。模型與特徵族的選擇只在外層 train 的 inner
StratifiedKFold(2) 進行。開發切分及固定方法控制不被冒稱成最終無偏選模證據。

提交條件從嚴：兩種種子都須 ΔBA>0.001 且 paired CI 下界>0。
本次即使採原先較鬆的第二種子只需正向條件，也全部失敗；從嚴確認不改變結論。
沒有通過者就不做 full-data candidate refit、不強制1700筆配額、不產生新提交檔。
如果未來通過，最終 full retrain 也會重新 cross-fit 整套特徵、重訓模型，
不將100% TE偷換到既有80% fold模型。

測試包含 TE/KNN 自標籤干預、折隔離、真實 OOF ID 完整性、unknown fallback、
機率、修復保留可信主值與原資料 hash；兩次 fresh refit 的最大機率差0。
Notebook cells 已檢查語法；不宣稱已在 Google Colab 平台另行完整執行。

## 不應誤用

本地 BA 不等於 Kaggle Accuracy；開發切分約0.69不可當作 Kaggle0.680。
先前手工選定特徵可能有歷史全資料探索偏差，本研究無法追溯消除，亦沒有
把重複種子說成全新獨立資料。候選 CI 亦不涵蓋所有訓練與選模變異。
下一步優先取得能真正在目前 schema 執行、產生0.679提交的原始程式與配套OOF。
