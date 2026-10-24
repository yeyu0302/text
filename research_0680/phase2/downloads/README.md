# GitHub 下載

- [第二輪完整研究包](Kaggle_Phase2_complete.zip)：7,278,259 bytes，包含程式、Colab、消融、兩種子OOF、折機率快取及報告。
- [SHA256](Kaggle_Phase2_complete.zip.sha256)／[逐檔大小與雜湊清單](bundle_inventory.json)。ZIP CRC與全部逐檔內容已核對。
- 沒有通過門檻的新提交，故包內沒有新 `submission_candidate_*.csv`。原 `submission_user_0679.csv` 保持原樣，僅作既有成果留存。

請先在瀏覽器登入GitHub，點ZIP後選 **Download raw file**。此包約7.28MB，可直接從GitHub下載，無須讀取Codex工作機。

ZIP不重複打包原始train/test/sample資料，也不包含第一輪全部研究結果。重現第二輪時，解壓後將原始三份CSV放在 `text/` 根目錄；或直接從此分支 **Code → Download ZIP** 取得含原始資料的完整儲存庫。第一輪包仍保留在 `research_0680/downloads/`。

完整研究報告見 [final_report.md](../reports/final_report.md)。完整儲存庫還包含交付後驗證紀錄 `results/delivery_checks.json`；ZIP內各檔以inventory為準。
