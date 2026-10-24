# 精簡搜尋第三輪

未找到通過既有驗證門檻的新Submission。最好的樹模型／樣條規則在兩個種子皆略優於參考，但CI跨0，不能宣稱Kaggle達0.680。

- [完整報告與消融](REPORT.md)
- [8個正式OOF run比較](nested_comparison.csv)
- [查核紀錄](verification.json)／[18項測試紀錄](all_tests.log)
- [程式、Colab與全部第三輪結果 ZIP](downloads/Kaggle_Phase3_complete.zip)
- [Colab](https://colab.research.google.com/github/yeyu0302/text/blob/research-0680-20261024/research_0680/phase3/Kaggle_Phase3_Colab.ipynb)

主程式234行，只有一份驗證runner；模型只增加固定400棵淺樹、分段樣條、平衡損失、48個物理交互項、固定規則量測修復。沒有Optuna、大網格、外層early stopping或可調混合權重。逐折只存ID雜湊，避免重複長清單。

重現時從儲存庫根目錄執行，先安裝 `research_0680/configs/requirements.txt`。使用**全新輸出目錄**保留已交付成果：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export RESEARCH0680_PHASE3_RESULTS="$PWD/reproduction_phase3/trees_spline"
export RESEARCH0680_PHASE3_BALANCED_RESULTS="$PWD/reproduction_phase3/balanced"
export RESEARCH0680_PHASE3_INTERACTION_RESULTS="$PWD/reproduction_phase3/interaction"
export RESEARCH0680_PHASE3_REPAIR_RESULTS="$PWD/reproduction_phase3/repair"
python -B -m research_0680.phase3.run
python -B -m research_0680.phase3.balanced
python -B -m research_0680.phase3.interactions
python -B -m research_0680.phase3.repair
```

每個命令先篩選再進正式驗證；若弱於預設條件就停。預設目錄有快取，直接重跑會重用快取，不是一輪新的獨立驗證。若只想查看結果，不必重跑模型。ZIP不含原始train/test/sample資料；可從此分支Code → Download ZIP取得含原始資料的完整儲存庫，或將三份CSV放到研究包解壓後的text根目錄。

兩種子都須ΔBA>.001且paired95%CI下界>0，才會建立新候選。本輪四條路線均失敗，沒有新submission_candidate檔；test_probabilities.csv僅作研究紀錄。原0.679提交、所有歷史結果、main保持原樣。
