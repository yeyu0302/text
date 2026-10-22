# 研究結果下載

本分支 `research-results-20261022` 用來保存研究產出與下載檔案；沒有修改 `main`。

開啟 [downloads 資料夾](https://github.com/yeyu0302/text/tree/research-results-20261022/downloads)，點選檔案，再使用 GitHub 的 Download raw file／下載原始檔功能。

- `downloads/best_submission.csv`：主要提交檔。
- `downloads/alternative_submission.csv`：全資料重訓的替代提交檔。
- `downloads/text_code_and_main_results.zip`：程式、設定、CSV、報告及日誌，不含二進位預測快取與原始資料集。
- `downloads/text_research_deliverables.zip`：完整研究交付檔，包含各模型的二進位預測結果；原始 train/test/sample CSV 不重複打包。
- `results/experiment_log.csv`、`results/oof_predictions.csv`：實驗紀錄與 OOF 預測。
- `results/final_report.md`：完整報告。

兩份 Submission 各 5,100 筆，每類 1,700 筆，ID 與原測試集順序相同。OOF Balanced Accuracy 為 0.67882／0.67664，尚未達到 0.700；沒有新的 Kaggle 實測成績。

本次僅保存與上傳既有成果，沒有重新訓練模型。
