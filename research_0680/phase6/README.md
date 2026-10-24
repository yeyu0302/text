# 提交退步分析（不重訓受損分類模型）

先讀 [REPORT.md](REPORT.md)。已知最佳檔案是 [submission_recovery_known_best_0679.csv](downloads/submission_recovery_known_best_0679.csv)，與使用者 submission (3) 位元相同；不是新提升。

從研究分支下載本目錄的 ZIP，或下載整個 GitHub 分支。ZIP 是本輪診斷與證據的封存；重跑時使用完整 repository，內含原始資料、基礎程式與 phase5 匹配機率快取。不要拿硬標籤 CSV 充當模型機率。

```bash
git clone --branch research-0680-20261024 https://github.com/yeyu0302/text.git
cd text
python -m venv .venv
.venv/bin/pip install -r research_0680/configs/requirements.txt
# 指定新的目錄，避免覆蓋任何舊結果。
RESEARCH0680_PHASE6_RESULTS=/tmp/phase6-audit-new .venv/bin/python -m research_0680.phase6.run_all
.venv/bin/python -m pytest -q research_0680/phase6/tests
```

`run_all` 不訓練受損分類器；只有小型 X-only LMMSE 擬合、CSV 比對與快取機率 bootstrap。Colab notebook 提供同樣步驟，未聲稱已在 Colab 平台執行。數值估計與輸入檔案 hash 已保存。PDF 原文未放入下載封存。
