# V5 / V11 分數落差研究

先讀 [REPORT.md](REPORT.md)。本輪沒有通過驗證的新候選CSV；使用者確認新bagged提交Kaggle=.672。已知最高.679仍保存在phase6。

正式診斷結果為 **audit_results_v2/**。`audit_results/`及`audit_execution.log`保留初次輸出欄位重複造成中斷的紀錄；修正後重新寫入新目錄，沒有覆蓋。`results/`為最初核對與快取融合探索。`native_results/`只有一個新無TE LightGBM的小型篩選，失敗後未擴訓。

從完整GitHub研究分支重現（根目錄資料與先前快取是必要輸入）：

```bash
git clone --branch research-0680-20261024 https://github.com/yeyu0302/text.git
cd text
python -m venv .venv
.venv/bin/pip install -r research_0680/phase7/requirements.txt
# 每次用新的輸出目錄，既有結果會拒絕覆寫。
RESEARCH0680_PHASE7_RESULTS=/tmp/phase7-audit-new OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m research_0680.phase7.audit
.venv/bin/python -m pytest -q research_0680/phase7/tests
```

`transport_risk.py`內的`transport()`是報告敏感度試算的純函式；原始結果`risk_sensitivity.json`已保存，其main會拒絕覆寫。另存新結果時可在Python呼叫該函式，假設與方法參見報告。敏感度數字不是Kaggle預測。

`native.py`會讀取已保存的相同協定篩選，遇No-Go不再重訓。只有要重現該3.55秒實驗時，另設`RESEARCH0680_PHASE7_NATIVE_RESULTS`為新目錄；沒有必要反覆重做失敗路線。

下載ZIP為本輪程式、輸入機率、結果與報告封存；重跑整套比較請使用完整repository。Colab notebook尚未在Colab平台執行，先前環境的實際驗證在測試／執行紀錄中。原始PDF未重新分發。
