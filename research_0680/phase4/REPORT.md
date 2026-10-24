# 第四輪：可提交的實驗候選與實測證據

**已依最新要求生成可提交CSV，但本輪仍沒有證據確認能超過Kaggle0.680。** 主檔是 `downloads/submission_candidate.csv`，備用為 `downloads/submission_candidate_balanced1700.csv`。兩者都是實驗候選，不取代既有0.679，也不假稱通過統計門檻。主檔5100列、id順序與sample相同、欄位僅id,label、類別1/2/3無缺值。

## 新操作與停止結果

1. **釋放核心集合以外的既有物理／TE特徵**：原參考不是只有55欄，而是55核心+類別指示+原代理，共117欄；本輪擴至213欄，沒有加入ID、原始數值地區ID或未篩選噪音。比較二次邏輯、分段線性hinge及32單元tanh MLP。MLP固定最多90輪、class-balanced loss，不做架構網格；擴充路線的整體選擇規則未改善。
2. **相鄰類別專家**：分開學Level1↔2、Level2↔3，使用兩類平衡權重，以兩組相鄰odds還原三分類機率；比較小型hist模型與二次邏輯。不以任意類別配額假裝Level2變準。seed42有正差，seed2026退步。
3. **訓練折內重新檢查186個排除欄位**：只用允許訓練資料建立數值十分位分箱／類別one-hot，以chi-square + BH FDR .05選最多8欄。開發訓練部分只選到f5（q=.0304）；加進去開發BA由.687803降至.686403，停止外層訓練。不能把這個關聯叫成確定訊號，也不能說其餘欄位已證明全是噪音。
4. **有界損失抑制疑似錯標影響**：Brier或低真類機率處封頂梯度的cross-entropy，係數以參考為anchor，L-BFGS最多60步，不刪列、不改標籤。內層全部退回參考，所以兩種子結果與參考相同，沒有增益。
5. **重用不同模型的預測**：對前一輪樹／樣條、本輪擴充、本輪相鄰專家三個已內層選擇的規則，固定50%算術平均+50%幾何平均。沒有重新搜尋權重，也沒有重新訓練所有模型。此pool在看過組成規則的歷史OOF之後提出，必須標為探索性組合。

主研究程式294行，共用既有驗證runner；沒有Optuna或大量公榜提交。固定參考兩種子BA為.674303922／.678115614。

## 完整OOF與各類召回

每份15000列。各新基底路線外層StratifiedKFold5 seed42／2026；模型、特徵或混合選擇只在外層train的2折seed391進行。TE使用不依賴標籤分折的KFold3，主值修復、縮放、knots、欄位篩選均在允許訓練資料擬合。查核每折train/val ID雜湊及原資料hash。

| route | seed | balanced_accuracy | recall_1 | recall_2 | recall_3 | delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| expanded features / hinge / MLP | 42 | 0.673525 | 0.736228 | 0.566161 | 0.718185 | -0.000779 | -0.004293 | 0.002522 |
| expanded features / hinge / MLP | 2026 | 0.677686 | 0.744631 | 0.561027 | 0.727400 | -0.000430 | -0.004537 | 0.003736 |
| adjacent class specialists | 42 | 0.676872 | 0.746499 | 0.577130 | 0.706987 | 0.002568 | -0.000532 | 0.006058 |
| adjacent class specialists | 2026 | 0.676443 | 0.743231 | 0.574562 | 0.711536 | -0.001673 | -0.005447 | 0.002538 |
| bounded training losses | 42 | 0.674304 | 0.740896 | 0.572695 | 0.709320 | 0.000000 | 0.000000 | 0.000000 |
| bounded training losses | 2026 | 0.678116 | 0.747899 | 0.571995 | 0.714452 | 0.000000 | 0.000000 | 0.000000 |
| fixed hybrid pool (exploratory) | 42 | 0.675899 | 0.743231 | 0.575496 | 0.708970 | 0.001595 | -0.001401 | 0.005006 |
| fixed hybrid pool (exploratory) | 2026 | 0.679787 | 0.746032 | 0.574796 | 0.718535 | 0.001672 | -0.001930 | 0.005232 |

pool相對參考在兩種子皆正向，但1000次paired row bootstrap CI均跨0；既有門檻要求兩種子Δ>.001且下界>0，仍未通過。這些CI不涵蓋歷史選模、重訓或pool提出時的事後選擇偏差；兩個種子亦不是兩份獨立資料。

**seed2026普通Accuracy=.681400不能用來聲稱已超過Kaggle0.68**：訓練驗證分布不均衡，主要代理指標是Balanced Accuracy=.679787。pool的OOF結果只是各種子下的外層機率合併；最後test再平均兩種子的fold模型，沒有宣稱這個test平均另外具有一份獨立OOF成績。

Level2逐列變化（fixed=參考錯而新對，broken=參考對而新錯）：

| seed | level2_fixed | level2_broken | other_fixed | other_broken |
| --- | --- | --- | --- | --- |
| 42 | 109 | 97 | 120 | 118 |
| 2026 | 118 | 106 | 154 | 123 |

## 開發消融

以下皆3750列探索切分，不冒稱完整OOF或Kaggle分數：

| route | choice | balanced_accuracy | recall_1 | recall_2 | recall_3 | delta |
| --- | --- | --- | --- | --- | --- | --- |
| expanded features / hinge / MLP | reference | 0.687803 | 0.733209 | 0.619048 | 0.711153 | 0.000000 |
| expanded features / hinge / MLP | expanded_quad | 0.685473 | 0.722015 | 0.629318 | 0.705086 | -0.002330 |
| expanded features / hinge / MLP | hinge | 0.683448 | 0.729478 | 0.618114 | 0.702753 | -0.004355 |
| expanded features / hinge / MLP | mlp | 0.687480 | 0.759328 | 0.566760 | 0.736351 | -0.000323 |
| expanded features / hinge / MLP | reference+hinge | 0.688425 | 0.735075 | 0.621849 | 0.708353 | 0.000622 |
| expanded features / hinge / MLP | reference+mlp | 0.691375 | 0.748134 | 0.600373 | 0.725618 | 0.003572 |
| expanded features / hinge / MLP | hinge+mlp | 0.685463 | 0.746269 | 0.590103 | 0.720019 | -0.002340 |
| adjacent class specialists | reference | 0.687803 | 0.733209 | 0.619048 | 0.711153 | 0.000000 |
| adjacent class specialists | pair_hist | 0.689046 | 0.738806 | 0.615313 | 0.713019 | 0.001243 |
| adjacent class specialists | pair_quad | 0.684069 | 0.729478 | 0.608777 | 0.713952 | -0.003734 |
| adjacent class specialists | reference+pair_hist | 0.689358 | 0.735075 | 0.619981 | 0.713019 | 0.001555 |
| adjacent class specialists | reference+pair_quad | 0.689047 | 0.733209 | 0.619048 | 0.714886 | 0.001244 |
| adjacent class specialists | pair_hist+pair_quad | 0.689512 | 0.738806 | 0.614379 | 0.715352 | 0.001709 |
| bounded training losses | reference | 0.687803 | 0.733209 | 0.619048 | 0.711153 | 0.000000 |
| bounded training losses | brier | 0.685937 | 0.731343 | 0.621849 | 0.704620 | -0.001866 |
| bounded training losses | capped_ce | 0.684070 | 0.731343 | 0.616246 | 0.704620 | -0.003733 |
| bounded training losses | reference+brier | 0.689514 | 0.735075 | 0.623716 | 0.709753 | 0.001711 |
| bounded training losses | reference+capped_ce | 0.686713 | 0.735075 | 0.615313 | 0.709753 | -0.001090 |
| bounded training losses | brier+capped_ce | 0.685780 | 0.735075 | 0.617180 | 0.705086 | -0.002023 |
| training-local omitted-feature discovery | reference | 0.687803 | 0.733209 | 0.619048 | 0.711153 | 0.000000 |
| training-local omitted-feature discovery | discovery | 0.686403 | 0.733209 | 0.616246 | 0.709753 | -0.001400 |
| training-local omitted-feature discovery | reference+discovery | 0.685004 | 0.725746 | 0.619048 | 0.710219 | -0.002799 |

## 兩份pool候選的差別

- **主檔 `downloads/submission_candidate.csv`**：預測數量1632／1770／1698，採argmax。沒有強制未知測試標籤分布。與原0.679檔4839列相同、261列不同；相同數量不是正確率。
- **備用 `downloads/submission_candidate_balanced1700.csv`**：各1700列，透過預先固定的exact-log transportation解全域最適配額；只有測試真為各1700時，這項約束才有依據。資訊來自使用者提供的競賽背景，不能由sample（全3）或舊提交也各1700就證明。
- 另外保留早期擴充路線的full-fit實驗檔 `results/submission_candidate_phase4_experimental.csv`，不是本輪主推薦，未刪除或覆寫。

平衡panel以每個外層留出折各真類等量抽樣，重複5次比較預先固定的argmax／exact-log。標籤只建立評估分布與算分，沒有傳給decoder或調整權重。重複panel彼此重疊，不是獨立重複試驗：

| seed | method | decision | mean | std |
| --- | --- | --- | --- | --- |
| 42 | pool | argmax | 0.673756 | 0.010420 |
| 42 | pool | exact_log | 0.674254 | 0.010221 |
| 42 | reference | argmax | 0.671888 | 0.010340 |
| 42 | reference | exact_log | 0.672542 | 0.010012 |
| 2026 | pool | argmax | 0.679334 | 0.012527 |
| 2026 | pool | exact_log | 0.679147 | 0.012795 |
| 2026 | reference | argmax | 0.678247 | 0.011046 |
| 2026 | reference | exact_log | 0.677438 | 0.013004 |

配額版沒有穩定改善，因此主檔不套配額。這些panel也未證實0.680。

## 重現、交付與限制

- 使用新版 `pool.py` 在全新目錄重跑合併與decoder，主／備用CSV都和本次檔案**逐byte相同**，未重訓任何模型。完整重新訓練的程式也已提供。
- 25項測試通過：包含新增的全部TE自身標籤排除、query batch隔離與有界損失解析梯度驗證。初次pytest因跨目錄測試同名而無法收集，已改成唯一檔名並保留失敗log；最終25項通過。小樣本MLP測試兩次達固定90輪上限，沒有宣稱其已收斂。
- `verification.json`保存每個正式run、來源機率一致性、CSV雜湊與schema驗證；`candidate_status.json`明示Kaggle分數unknown、統計門檻false。
- Notebook預設只重現已保存機率的CSV，省去不必要重訓；切換FULL_RETRAIN可完整重訓主候選三個來源規則。Notebook僅檢查語法，未宣稱Google Colab平台已另跑完整訓練。
- 使用者0.679的真正生成pipeline仍有版本／schema缺口，不能把本研究參考當它的OOF。本輪候選的正差是相對可重現參考，不是相對Kaggle0.679。
- 原0.679、原始資料、既有全部結果保持原樣，GitHub只新增研究分支內容。依最新明確要求交付實驗檔，不等於放寬證據或保證分數。是否超過0.68仍須Kaggle實際評分。
