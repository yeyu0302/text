"""Render the measured evidence; no claimed Kaggle score is generated here."""
import json
import pandas as pd
from .paths import OUT, REPORTS

def table(df,columns):
    rows=['| '+' | '.join(columns)+' |','| '+' | '.join(['---']*len(columns))+' |']
    for _,r in df.iterrows():
        rows.append('| '+' | '.join(f'{r[c]:.6f}' if isinstance(r[c],float) else str(r[c]) for c in columns)+' |')
    return '\n'.join(rows)

def run():
    REPORTS.mkdir(parents=True,exist_ok=True)
    formal=pd.read_csv(OUT/'nested_comparison.csv');screen=pd.read_csv(OUT/'screening_summary.csv')
    recon=pd.read_csv(OUT/'reconstruction_summary.csv');changes=pd.read_csv(OUT/'level2_prediction_changes.csv')
    report='''# 五項困難的實測與第二輪實驗

**本輪未找到能推薦替換既有 0.679 提交的方法；沒有新的 Kaggle 成績，也沒有通過門檻的新提交檔。**
完成的是資料說法核對、新量測估計方法、新村落條件收縮模型，以及兩個種子的嚴格巢狀驗證。
第一輪 S0–S6 與使用者原提交完整保留於上層目錄；本報告只新增第二輪結果。

## 哪些「天花板」有證據？

1. **部分欄位相同而標籤不同，不能推導全模型 Bayes 上限。** 局部土壤、施工品質或判定差異是可能原因，表內無法驗證。對明確定義的八欄簽章（f141、屋齡、樓層、兩建材、屋頂、f62、f209），原值有462個衝突群、1131列；清理屋齡／樓層後495群、1210列。換樓板定義數字會改變。未重現2357列／1028群，全部定義見 `partial_duplicate_definitions.csv`。多數票是同資料、部分欄位的描述，既非未見資料測試，也非完整特徵的理論上限。完整原始 X 無重複，同樣不能證明沒有不可約誤差。
2. **實際238個預測欄位，沒有生成器證據證明175個全是純亂數。** 既有核心集合能限制方差，但這是特徵假設。本輪保留凍結參考，另建不含地理ID、村落均值或TE的物理專家做控制，不根據全資料特徵重要度重新篩選。純物理開發BA僅0.517，顯示目前模型強烈依賴地區資訊，不能直接刪除地理特徵。
3. **代理噪音很大，但14.9不是不可突破的誤差下限。** 單一屋齡代理殘差標準差52.4–52.9年，簡單均值MAE14.80年、殘差標準差18.68年；MAE與標準差不能混用。依明示規則偵測到11.96%列至少一項異常、四個主欄儲存格3.09%有標記；規則抓不到所有外觀正常的污染，因此不能據此估計真實污染率，更不能斷言是蓄意下毒。
4. **Level2確實較難，但召回率須和特定模型綁定。** 本次固定參考seed42三類召回為74.09%／57.27%／70.93%，不是使用者提到的73.8%／60.2%／70.3%。類別在中間不代表所有錯誤都由單一物理邊界造成。以下直接計算新方法修正與新增錯誤；不把低信心列當成已確認錯標，未修改任何y。
5. **45.8%小村落說法可重現。** 測試集45.7647%位於完整訓練集樣本數≤15的f141群，共945群。外層驗證時必須只用該折訓練樣本計數；因此驗證列的比例不同。小群體需要收縮，但是否加分仍需驗證。

## 兩種新方法

**量測後驗估計：** 只用允許訓練列X建立可信主量測的離散先驗，依每個代理的校準斜率、偏置與殘差尺度，使用Student-t(df4)似然整合，輸出後驗均值、標準差、熵及主值差距。保留通過規則的主量測，不使用損傷標籤修復物理量。代理條件獨立與可信主值分布都是建模假設，不是已知物理真相。沒有可信主值的列可能有不同分布。

下面是5折只以訓練X擬合、在留出且通過可信規則主值上評估的結果。量測評估只比較代理估計，沒有用保留主值取得零誤差。主值是silver reference，並非真實潛在物理量：

'''
    report+=table(recon,['source','method','n','mae','rmse'])
    report+='''

屋齡MAE降至9.49年，證明利用先驗可以改善這個重建指標。但分類開發BA幾乎不變，所以停止這條分類候選路線；不能將重建改善包裝成Kaggle進步。

**條件階層收縮：** 用純物理二次特徵邏輯模型的3折cross-fit機率作offset，依f90→f109→f141估計三類對數機率修正；各群以Gaussian懲罰向上層修正收縮，未知子群回退父群。這是分階段經驗MAP，並非完整貝氏後驗。區域修正是訓練資料擬合的模型參數，沒有讓外層驗證標籤進入特徵。以參考機率混合保留原模型訊號。內層重新選擇懲罰2／8／32，以及物理區域專家權重0／.25／.5／.75／1；相對內層參考增益不超過.001即退回參考。

## 開發消融（3750列；不能冒稱完整OOF或Kaggle分數）

'''
    report+=table(screen,['case','config','balanced_accuracy','recall_1','recall_2','recall_3','delta'])
    report+='''

開發切分seed847，分類候選及範圍於第二輪screening之前寫入config。混合.25的開發表現較好，但沒有直接拿這個贏家跑外層；所有λ與權重在每個外層訓練集內重新選擇。後驗重建分類未達開發+.001，停止其正式驗證以節省無效訓練。

## 嚴格外層結果（每種子15000列）

外層StratifiedKFold(5,shuffle=True,seed42／2026)；選模內層StratifiedKFold(2,seed391)；物理基底cross-fit為不依賴標籤的KFold(3,seed913)。所有校準、填補、縮放、先驗、物理模型與區域修正只在允許訓練資料擬合。不得使用外層標籤挑權重、處理錯標或做最終閾值調整。

同種子固定參考BA為0.674303922／0.678115614；此參考是另外凍結的可重現模型。使用者提供的0.679原程式在目前資料上因將`c3`轉float而失敗，V6程式與V5參數亦不匹配；因此不能將它冒稱為0.679提交的OOF，也不能將下列差值說成相對Kaggle0.679的差值。

'''
    report+=table(formal,['seed','balanced_accuracy','recall_1','recall_2','recall_3','delta','ci_low','ci_high'])
    report+='''

CI為1000次ordinary paired row bootstrap，不是按類別固定比例重抽。兩組CI均跨0；第二種子更差。門檻要求兩個種子皆ΔBA>.001、CI下界>0，本輪No-Go。CI不涵蓋全部重訓或歷史多次探索的不確定性；兩個種子不是獨立的新資料。這一輪受第一輪觀察啟發，亦沒有一份額外封存盲測集。

Level2逐列變化（fixed=參考錯而新模型對；broken=參考對而新模型錯）：

'''
    report+=table(changes,list(changes.columns))
    report+='''

seed42把2→1由829降到815，卻讓2→3由1002增到1012，只淨增4個Level2正確；seed2026淨少3個Level2正確，其他類別也退步。小群體BA：seed42參考0.632360→0.631930，seed2026為0.637080→0.638734；效果方向不穩定。完整大／小群體與各類Recall見`group_size_comparison.csv`。不得以某個子群或單一種子較好取代整體門檻。

## 交付與可重現邊界

- `../results/experiment_log.csv`：9個開發消融及2個正式規則／種子；scope明示。
- `../results/nested_conditional_seed{42,2026}/oof_predictions.csv`：每份15000列的機率、預測、折ID；另有confusion、fold metrics、test probabilities、原始折快取與完整內層選擇／train-val ID紀錄。
- `../results/submission_gate.json`：No-Go；沒有新submission_candidate檔。測試機率保留作研究記錄，並非推薦提交。
- `../results/reconstruction_*.csv`、`claim_audit.json`、`partial_duplicate_definitions.csv`：量測與五項說法核對。
- `../notebooks/Kaggle_Phase2_Colab.ipynb`與`../src/run_all.py`：單入口執行。重跑請使用新輸出目錄，保留本次結果。完整重跑會訓練模型；本輪既有檔案不會被重跑指令覆寫。
- 已實跑原程式失敗取證、開發與兩種子外層、量測重建及14項測試；Notebook只做語法檢查，未宣稱在Google Colab平台另外完整跑過。

沒有證據說目前已到物理極限，但也沒有證據支持這批候選能超過Kaggle0.680。較有價值的下一步是取得真正產生0.679的可執行pipeline與其完整OOF，再預留新的獨立驗證／以相同fold評估互補性；若沒有這些證據，持續按既有OOF調權重容易選模過擬合。既有提交、原始資料與第一輪成果均保持原樣。
'''
    (REPORTS/'final_report.md').write_text(report,encoding='utf-8')

if __name__=='__main__':run()
