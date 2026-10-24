# 第四輪：已生成可提交的實驗CSV

**尚未證實Kaggle能超過0.680。** 依最新要求提供實驗候選，不取代原0.679；完整限制與每項成功／失敗記錄見[報告](REPORT.md)。

- **[主提交CSV](downloads/submission_candidate.csv)**：5100筆，`id,label`，採固定模型pool的argmax。
- [各類1700筆備用CSV](downloads/submission_candidate_balanced1700.csv)：僅適用於測試確實各1700的額外假設；沒有穩定驗證優勢。
- [程式、來源機率、OOF、Colab與報告ZIP](downloads/Kaggle_Phase4_complete.zip)。不含原始三份資料CSV及二進位折快取，完整折快取仍在GitHub分支。
- [驗證狀態](candidate_status.json)／[逐檔CSV與折ID查核](verification.json)／[25項測試](all_tests.log)。
- [Colab](https://colab.research.google.com/github/yeyu0302/text/blob/research-0680-20261024/research_0680/phase4/Kaggle_Phase4_Colab.ipynb)：預設從保存機率重現CSV，避免重訓；可切換完整重訓。

pool兩種子OOF BA為0.675899／0.679787；相對參考+.001595／+.001672，但CI跨0。pool是在檢視組成規則的歷史OOF之後提出的探索性組合，不能宣稱無偏確認或已知Kaggle分數。兩個種子不是獨立的新資料。seed2026普通Accuracy0.6814也不能代替主要代理指標BA。

研究主程式294行，共用既有nested runner，沒有大型參數搜尋。新測試：全物理／TE擴充、相鄰類別專家、訓練內FDR篩選排除欄位、有界損失及固定pool；弱路線的失敗完整保存。

## 不訓練，快速重現主／備用CSV

使用Python3.12，從完整儲存庫或研究包的text根目錄執行；先安裝 `research_0680/configs/requirements.txt`。研究包使用者另將原始train/test/sample_submission放在text根目錄。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export RESEARCH0680_PHASE4_POOL_RESULTS="$PWD/reproduction_phase4_pool"
python -B -m research_0680.phase4.pool
```

請使用全新輸出目錄，避免覆寫。這個命令只使用已保存的來源機率，生成的新主／備用CSV已核對與本次逐byte相同；不是一次重新訓練。

## 完整重訓主候選來源

```bash
export RESEARCH0680_PHASE3_RESULTS="$PWD/reproduction_phase4/compact"
export RESEARCH0680_PHASE4_RESULTS="$PWD/reproduction_phase4/expanded"
export RESEARCH0680_PHASE4_PAIR_RESULTS="$PWD/reproduction_phase4/pairwise"
export RESEARCH0680_PHASE4_POOL_RESULTS="$PWD/reproduction_phase4/pool"
python -B -m research_0680.phase3.run
python -B -m research_0680.phase4.run
python -B -m research_0680.phase4.pairwise
python -B -m research_0680.phase4.pool
```

這會重新訓練；每折源模型及TE使用同一訓練範圍，不將fold模型TE換成全資料。其他消融可用 `research_0680.phase4.robust`、`research_0680.phase4.discovery`，分別設定全新 `RESEARCH0680_PHASE4_ROBUST_RESULTS`／`RESEARCH0680_PHASE4_DISCOVERY_RESULTS`。為重現主CSV不需要重跑這些失敗消融。

GitHub僅新增研究分支成果；main、原始資料、歷史CSV均保持不變。Kaggle實際分數必須在競賽評分後確認，這裡沒有取得新分數。
