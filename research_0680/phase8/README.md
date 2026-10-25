# 提供程式的實際可重現性檢查與小型修正

先讀[REPORT.md](REPORT.md)。兩份原檔相同，原始main只載入歷史成果；本輪沒有新提交CSV。

檔案：
- `evidence/`：兩份不可變原始程式。
- `source_audit.py`、`audit_results/`：確定的入口／欄位／TE篩選／Plan B算式問題。
- `fixes.py`：具fit/transform狀態的特徵與要求A/B外層OOF的混合介面。
- `fold_base.py`：僅由訓練列建立類別／地區context及schema。
- `ablation.py`：可執行的重建配對流程，**不是**原始V5/V11重現。
- `ablation_results/`：初始300次求解，未收斂。
- `convergence_results/`：明示數值修訂後成功收斂；增益不足，沒有外層擴訓。

重現需完整研究分支的原始資料與共用程式：

```bash
git clone --branch research-0680-20261024 https://github.com/yeyu0302/text.git
cd text
python -m venv .venv
.venv/bin/pip install -r research_0680/configs/requirements.txt
RESEARCH0680_PHASE8_AUDIT_RESULTS=/tmp/phase8-audit-new OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research_0680.phase8.source_audit
# 使用新的結果目錄，初始與收斂修訂都有明確協定。
RESEARCH0680_PHASE8_RESULTS=/tmp/phase8-probe-new OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m research_0680.phase8.ablation
RESEARCH0680_PHASE8_PROTOCOL=research_0680/phase8/convergence_protocol.json RESEARCH0680_PHASE8_RESULTS=/tmp/phase8-converged-new OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -m research_0680.phase8.ablation
.venv/bin/python -m pytest -q research_0680/phase8/tests
```

不要再跑失敗路線來累積實驗次數。預設輸出若已有同協定的No-Go結果，訓練入口只讀快取、不重訓。不同協定不得共用同結果目錄。候選需要事先設定的篩選與外層檢查通過；本次沒有通過。

ZIP為本輪證據、程式及結果封存。重跑請clone完整repository；Colab notebook也使用完整研究分支，未宣稱已在Colab平台執行。
