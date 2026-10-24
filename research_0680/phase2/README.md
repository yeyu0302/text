# 第二輪：核對五項困難並實驗新方法

本輪沒有通過驗證的新候選，不能宣稱 Kaggle 已超過0.680。
[完整報告](reports/final_report.md)記錄五項說法的實測、新方法、消融、Level2錯誤及限制。
第一輪 S0–S6 留在上層目錄，既有0.679提交沒有覆寫。

- 屋齡代理重建：留出可信主值MAE由14.80降至9.49年；分類開發集無提升，停止此路線。
- 村落條件收縮與參考模型混合：開發集有提升；正式nested BA為0.674070／0.676405，低於同種子參考0.674304／0.678116，No-Go。
- 正式每份OOF為15000列；開發消融3750列有清楚scope，不當作Kaggle分數。
- [實驗總表](results/experiment_log.csv)、[兩種子比較](results/nested_comparison.csv)、[提交門檻](results/submission_gate.json)、[小群體比較](results/group_size_comparison.csv)。
- [下載完整第二輪程式／CSV／報告包](downloads/Kaggle_Phase2_complete.zip)；包含正式折快取及可重現基礎程式，原始資料不重複打包。資料可從儲存庫 Code → Download ZIP 取得。
- [開啟Colab](https://colab.research.google.com/github/yeyu0302/text/blob/research-0680-20261024/research_0680/phase2/notebooks/Kaggle_Phase2_Colab.ipynb)。私人儲存庫請先在GitHub下載ZIP，再開啟Notebook上傳功能。

從儲存庫根目錄使用Python3.12，先安裝 `research_0680/configs/requirements.txt`。如需完整重跑，在**新的、未使用過的目錄**執行：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export RESEARCH0680_RESULTS="$PWD/reproduction_phase2/reference"
export RESEARCH0680_REPORTS="$PWD/reproduction_phase2/reference_reports"
export RESEARCH0680_PHASE2_RESULTS="$PWD/reproduction_phase2/results"
export RESEARCH0680_PHASE2_REPORTS="$PWD/reproduction_phase2/reports"
python -B -m research_0680.phase2.src.run_all
```

這個命令會重新訓練；已交付的結果不會被覆寫。預設路徑已有折快取，不代表一次新的獨立重跑。Notebook語法已檢查，未宣稱在Colab平台重新完整執行。研究程式實際完成開發、外層、量測及14項測試；測試log在results。原主值只是silver reference，重建結果不是潛在真值精度保證。Bootstrap不涵蓋全部訓練／歷史選模變異，兩種子不是獨立的新資料。
