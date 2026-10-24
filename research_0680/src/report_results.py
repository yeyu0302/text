"""Generate the S0-S6 evidence reports, including every failure and no-go."""
import json,hashlib,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def md(frame,cols=None):
    d=frame if cols is None else frame[cols]
    def fmt(v):
        if isinstance(v,(float,np.floating)):return '' if np.isnan(v) else f'{v:.6f}'
        return str(v).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(d.columns)+' |','| '+' | '.join(['---']*len(d.columns))+' |']+
                     ['| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None)])

def run():
    REPORTS.mkdir(parents=True,exist_ok=True)
    x,y,test,sample=read_data();baseline={};allrows=[]
    for seed in [42,2026]:
        folder=RESULTS/'00_baseline'/f'fixed_baseline_seed{seed}'
        if (folder/'complete.json').exists():baseline[seed]=json.loads((folder/'complete.json').read_text())
    for file in sorted(RESULTS.rglob('complete.json')):
        item=json.loads(file.read_text())
        if 'balanced_accuracy' not in item:continue
        item['artifact_directory']=str(file.parent.relative_to(RESULTS))
        item['scope']='screening_holdout_3750' if 'stage' in item else 'strict_outer_oof_15000'
        item['status']='complete';allrows.append(item)
    allrows.append(dict(name='provided_V6_unmodified_reproduction',scope='S0_reproduction',status='failed',
        error="ValueError: could not convert string to float: 'c3'",artifact_directory='00_baseline/provided'))
    allrows.append(dict(name='provided_V5_parameter_score_claims',scope='S0_metadata',status='unverified',error='JSON describes V5; executable header V6; no matching OOF supplied'))
    for file in RESULTS.rglob('failure.json'):
        item=json.loads(file.read_text());item['scope']='screening_failed';allrows.append(item)
    table=pd.DataFrame(allrows)
    table.to_csv(RESULTS/'experiment_log.csv',index=False)
    old=RESULTS/'ablation_summary.csv'
    if old.exists() and not (RESULTS/'ablation_screening_summary.csv').exists():shutil.copyfile(old,RESULTS/'ablation_screening_summary.csv')
    table.to_csv(RESULTS/'ablation_summary.csv',index=False)
    outer=table[table.scope=='strict_outer_oof_15000'].copy()
    outer.to_csv(RESULTS/'outer_validation_summary.csv',index=False)
    paired=[]
    for _,r in outer.iterrows():
        if r['name']=='fixed_baseline':continue
        paired.append({k:r.get(k) for k in ['name','seed','delta','ci_low','ci_high','draws']})
    pd.DataFrame(paired).to_csv(RESULTS/'paired_delta_bootstrap.csv',index=False)
    # Per-row trades and subgroup changes, with ID joins rather than position assumptions.
    trades=[];sub=[]
    for _,r in outer.iterrows():
        if r['name']=='fixed_baseline':continue
        seed=int(r['seed']);basepath=RESULTS/'00_baseline'/f'fixed_baseline_seed{seed}'/'oof_predictions.csv'
        a=pd.read_csv(basepath).set_index('id').reindex(x.id)
        b=pd.read_csv(RESULTS/r.artifact_directory/'oof_predictions.csv').set_index('id').reindex(x.id)
        assert not b.isna().any().any()
        oldp=a.pred.to_numpy();newp=b.pred.to_numpy()
        for k in [1,2,3]:
            mask=y+1==k;gain=int(np.sum(mask&(oldp!=k)&(newp==k)));loss=int(np.sum(mask&(oldp==k)&(newp!=k)))
            trades.append(dict(name=r['name'],seed=seed,level=k,newly_correct=gain,newly_wrong=loss,net_correct=gain-loss))
        for field in ['roofing_kind','f_90','f_109','f_141','f_182']:
            for value,idx in x.groupby(field).indices.items():
                idx=np.array(idx);mask=y[idx]==1
                if not mask.any():continue
                sub.append(dict(name=r['name'],seed=seed,field=field,value=str(value),n=len(idx),level2_n=int(mask.sum()),
                    recall2_baseline=float(np.mean(oldp[idx][mask]==2)),recall2_new=float(np.mean(newp[idx][mask]==2)),
                    level2_newly_correct=int(np.sum((y[idx]==1)&(oldp[idx]!=2)&(newp[idx]==2))),
                    level2_newly_wrong=int(np.sum((y[idx]==1)&(oldp[idx]==2)&(newp[idx]!=2)))))
    pd.DataFrame(trades).to_csv(RESULTS/'level_tradeoffs.csv',index=False);pd.DataFrame(sub).to_csv(RESULTS/'subgroup_metrics.csv',index=False)
    forensic=RESULTS/'01_data_forensics'
    neighbor=pd.read_csv(forensic/'neighbor_summary.csv')
    auc=pd.read_csv(forensic/'adversarial_auc.csv');auc['feature_set']='supplied_52_feature_hypothesis'
    auc=pd.concat([auc,pd.read_csv(forensic/'adversarial_auc_all238.csv')],ignore_index=True)
    suspicion=pd.read_csv(forensic/'suspicious_labels_all.csv');quality=json.loads((forensic/'complete.json').read_text())
    association=pd.read_csv(forensic/'proxy_level2_error_associations.csv')
    px=pd.read_csv(forensic/'proxy_forensics.csv')
    recon=pd.read_csv(forensic/'proxy_reconstruction_outer.csv').groupby(['variant','source'])[['mae','rmse']].mean().reset_index()
    residual=pd.read_csv(forensic/'residual_and_weighting_holdout_mse.csv')
    schema=json.loads((RESULTS/'00_baseline/provided_schema_audit.json').read_text())
    bad=[r['feature'] for r in schema if r['non_numeric_count']>0]
    nonbinary=[r['feature'] for r in schema if not r['is_actually_binary']]
    source_dir=PROJECT/'results/00_baseline/provided'
    source_hash=hashlib.sha256((source_dir/'submission_user_0679.csv').read_bytes()).hexdigest()
    REPORTS.joinpath('data_quality_level2.md').write_text(f'''# S0–S1 資料品質、版本與 Level 2 取證

## 0.679 對應與 S0 狀態

使用者在本次 0.679 脈絡提供 `submission (3).csv`，已保留為
`results/00_baseline/provided/submission_user_0679.csv`。
SHA256：`{source_hash}`。5,100 筆、ID 與 sample 順序、標籤格式均符合。
這是**使用者回報的 Kaggle 分數**，未連線核對競賽提交歷史，也沒有相應 OOF。

上傳 Python 標頭是 V6，JSON 是 V5。直接、未修改的執行在 base feature
階段失敗：`'c3'` 無法轉為浮點數；尚未開始模型訓練。宣告二元但實為文字的
欄位有 {bad}；宣告二元但資料不僅 0/1 的欄位共有 {len(nonbinary)} 個。
沒有用強制轉型、移除欄位或重新編碼把這個失敗隱藏起來。
因此 **0.679 生成流程未重現，S0 此部分是 No-Go**，不是成功重現 V5/V6。

現有可執行程式提供本研究固定參考模型：Gemini core、原始 proxy、低基數
one-hot，fold-local 前處理及標準 regularized quadratic multinomial。
它不是 V4 Standalone，也不是所附 V6。所有後續 ΔBA 都是相對此固定模型，
**不是相對 Kaggle 0.679 的 ΔAccuracy**。V4 Standalone、V7、原始 V5 與
配套 OOF 均未找到；歷史報告分數沒有被當成已重現結果。

## 資料與可觀測分類衝突

Train 15,000、Test 5,100、預測欄位 238，train 計數 2142/4285/8573。
所附競賽 PDF 明示 Test 各 1,700 與 Accuracy；此證據被沿用，Test 標籤未知。
當前 Test 的 f_141 都在 train 出現；群組名稱只作代碼，不宣稱地理單位。
完全相同原始 X 的重複列有 {quality['exact_raw_duplicate_rows']}；清洗後物理特徵
存在不同標籤的簽章群有 {quality['conflicting_signatures']}。
這些特徵忽略未觀測因子，**不等於已證實錯標，更不能宣稱資料造假**。

`near_conflicting_pairs_for_review.csv` 列出跨外層折、距離很近但標籤不同的配對。
`label_review_candidates.csv` 結合主欄位、結構與群組資料，僅供人工審查。
真實 Level 2 且固定參考 OOF P(true)<0.1 的列有
{int(((suspicion.true_label==2)&(suspicion.p_true<.1)).sum())} 筆。
舊 CatBoost 是另一個獨立訓練的演算法；其歷史 OOF 折不同，僅作逐列共識診斷，
沒有把這種跨模型共識或全資料排名當作新模型訓練權重。
所有 train.csv 標籤保持不變。

## Proxy 測量品質

下表描述各 proxy，主欄位只作 silver reference，沒有將欄位角色假說當成金標準。

{md(px,['source','proxy','silver_correlation','peer_correlation','skew','excess_kurtosis','missing'])}

f_80 有顯著重尾；不是所有誤差都可視為同質 Gaussian。silver-reference
條件殘差的 pairwise-complete 相關見 `proxy_residual_correlations_corrected.csv`；
接近零不證明統計獨立。早期未處理缺失的相關與早期錯序的 conditional
permutation 表格保留為歷史中間產物，應使用檔名含 corrected 的修正版。

折內 calibration bias/scale/variance 在各正式模型目錄保存；修復行數與例子在
`repair_counts_and_examples.json`。V5 的可信主欄位完全不變已通過測試。
Proxy-only reconstruction 在外層未見的 silver main 值上評分，不用修復後
仍保留主值的零誤差來假裝去噪成功：

{md(recon)}

## Train/Test 分布

{md(auc,['feature_set','prior_standardized','roc_auc'])}

在兩種特徵集合都接近 chance，支持多數 domain 可辨識性與已知 label shift
有關；不能證明完全沒有 covariate/conditional shift，亦不能證明 Test Level 2
有問題。沒有基於這些結果採用 propensity reweighting。KS、PSI、SMD 與 FDR
在 `distribution_shift.csv`；未看到證據就不加入高方差權重。
''',encoding='utf-8')
    base42=baseline[42]
    REPORTS.joinpath('error_attribution_level2.md').write_text(f'''# Level 2 錯誤歸因

固定新參考的標準 sklearn 五折 OOF：BA {base42['balanced_accuracy']:.6f}；
Recalls {base42['recall_1']:.6f}/{base42['recall_2']:.6f}/{base42['recall_3']:.6f}。
Level 2→1 **{base42['level2_to1']}** 筆，Level 2→3 **{base42['level2_to3']}** 筆。
屋頂、屋齡、群組及 f182×storey 的確切錯判分解見
`01_data_forensics/subgroup_metrics.csv`；群組太小時不宣稱穩定因果關係。

## 近鄰混合並非單一方向

{md(neighbor)}

全域近鄰中，Level 2 錯判列的 entropy 較高；同群組近鄰則相反。
所以不能概括說所有 Level 2 錯判都因高熵。k=20 的區域方法約 75% 回退，
k=40 幾乎全數回退，細群樣本稀少限制其增量。這是可觀測可分性有限的
部分證據，並非錯標證明。距離只含標準化物理欄位、類別 mismatch 與二元
指標，不把 945 個群組 ID 當連續距離；所有 query 的標籤來源均在外層 fit。

## Proxy 分歧與錯判的關聯

{md(association,['feature','wrong_mean','correct_mean','fdr_q','controlled_cv_logloss_gain'])}

部分缺失指標有描述性關聯，但控制建築、群組與屋頂後的額外診斷 log-loss
收益很小。這些關聯不是因果，也不是最終模型的無偏改善證據。
兩組 bootstrap mean-difference CI 在原 CSV；conditional 分析屬探索性。

## 殘差與樣本權重

{md(residual)}

殘差模型確實在允許訓練集內產生 cross-fitted baseline p2，再擬合 residual，
沒有拿全域 OOF 的其他折來訓練當前驗證折校正器。它降低普通 MSE 卻未改善
balanced MSE／BA，故沒有保留；soft weighting 亦未採用。
沒有自行改標、移除驗證列或用 Test Label 評估。

匿名欄位 f51/f5/f125 的 corruption-flag conditional permutation 與隨機負控制
見 corrected CSV。這不是 damage-label 重要性，更不是將這三欄加入候選的理由。

## 新增判對與新增判錯

{md(pd.DataFrame(trades)[pd.DataFrame(trades).name.str.startswith('nested_')])}

個別屋頂、區域與建材的增損見研究根目錄 `results/subgroup_metrics.csv`。
不能只看 Level 2 Recall，而忽略 Level 1/3 或稀有群的新增錯誤。
''',encoding='utf-8')
    columns=['name','seed','balanced_accuracy','recall_1','recall_2','recall_3','delta','ci_low','ci_high']
    REPORTS.joinpath('ablation_summary.md').write_text(f'''# S2–S6 消融與選模

## 固定控制與外層流程

{md(outer,columns)}

兩個最終規則事先限定：proxy 規則只在外層 train 的 inner CV 從 V0–V5 選擇；
adaptive 規則依序在同一允許範圍內選 Proxy、TE、KNN、刪除特徵群與模型。
inner 每步要求超過 0.001 才取代簡單配置。Outer validation 完全不參與這些
選擇。所有 inner training TE/KNN 使用 label-independent KFold cross-fitting；
own-label intervention 測試涵蓋 TE 與 KNN。Inner 選擇用 StratifiedKFold(2)，
Outer 則按任務書 sklearn StratifiedKFold(5,shuffle=True,random_state=42/2026)。
為省算力沒有數百組 grid；S3 representation 與 smoothing 分成小組控制，
S5 的每一個刪除欄群分開跑。每項 scope 與負面結果見完整 CSV。

開發切分最高約 0.69，不等於完整 OOF；所有開發候選均標記 screening。
`S5_proxy_uncertainty` 在原 V0 沒有新不確定性欄，是明確 no-op；應使用
V3 uncertainty_control vs remove_uncertainty 這組配對，而非誤稱零效果已證明無用。

## TE 訓練／推論一致性

普通 TE、同平滑下 balanced TE、同平滑下 balanced-log TE 分開比較；
額外只有 finest group 的 pseudo-count 8/16/32。父群回退及 prior 全是 fit-local。
每折 `te_distribution_fold*.csv` 保存 cross-fitted train 與 query 的均值、變異及 NaN。
Full strategy 用該訓練集的 cross-fitted features 訓練單一模型，query TE 用該
訓練集全部標籤；fold ensemble 的每個組件連模型與 TE 都用同一子集。
沒有把 100% TE 偷換到既有 80% fold 模型。這不宣稱 cross-fitting 的樣本量
差造成的分布差異已完全消失，而是使完整方案可在外層公平模擬。

## 低成本開發切分（非最終 OOF）

{md(table[table.scope=='screening_holdout_3750'],['stage','name','balanced_accuracy','recall_1','recall_2','recall_3','status'])}

## 未執行或停止

V6 原樣重現失敗於資料 schema；沒有偽造後續分數。Residual／soft weighting
在 development 無改善即停。KNN 高 fallback 且增益未證實，沒有再放大頻寬搜尋。
未做 propensity weight、全資料 label ranking 選欄、公開榜權重調整或外部標籤搜尋。
當前 Test 無觀測到新 f141，因此沒有執行額外 GroupKFold 壓力測試，未知群回退
已由單元測試覆蓋。沒有通過 gate 的方法不做 full-data candidate refit。
''',encoding='utf-8')
    gates=pd.read_csv(RESULTS/'05_final_candidates/candidate_gate_decisions.csv')
    accepted=int(gates.passed.sum())
    primary=outer[(outer['name']=='nested_adaptive')&(outer.seed==42)].iloc[0]
    secondary=outer[(outer['name']=='nested_adaptive')&(outer.seed==2026)].iloc[0]
    REPORTS.joinpath('final_recommendation.md').write_text(f'''# 最終建議：未證實優於使用者的 Kaggle 0.679

## 結論

本次沒有達成「已驗證 Kaggle Accuracy 超過 0.680」；沒有上傳到 Kaggle，也沒有
看到隱藏 Test Label。通過提交 gate 的方法 **{accepted}** 個。
保留使用者的 `submission (3).csv`，**不替換目前回報 0.679 的最佳提交**。

完整 nested 規則 OOF BA：seed42 **{primary.balanced_accuracy:.6f}**，
seed2026 **{secondary.balanced_accuracy:.6f}**。它相對同種子新參考模型的
ΔBA 分別為 **{primary.delta:+.6f}** 與 **{secondary.delta:+.6f}**，
方向沒有重現。95% paired CI 分別是
[{primary.ci_low:.6f},{primary.ci_high:.6f}] 與
[{secondary.ci_low:.6f},{secondary.ci_high:.6f}]。
這些 conditional bootstrap 不重新訓練，不能當成涵蓋所有選模不確定性的證明。

{md(gates)}

## 必須回答的十個問題

1. **0.679 對應哪版？** 使用者提供的 `submission (3).csv` 已完整保存與雜湊。
   但新 Python 是 V6、JSON 是 V5，原樣執行因錯誤資料型別假設失敗；沒有它的
   OOF。不能把新參考或前次程式的 OOF 冒充該提交的 OOF。
2. **Level 2 具體錯判？** 新固定基準 seed42：2→1 {base42['level2_to1']}，
   2→3 {base42['level2_to3']}。結構子群及屋齡分解在 error_attribution 報告與 CSV。
3. **Proxy 分歧是否較高？** 並非各群一致；部分 missingness 有關聯，控制後
   額外 log-loss 改善很小，不能宣稱它就是錯誤根因。
4. **近鄰是否支持可分性有限？** 全域支持部分混合，同區域方向相反；k20
   75% fallback、k40 約99% fallback，不能以單一 entropy 敘事概括。
5. **可能 label noise？** 高信心錯判與衝突配對已列供人工查核，但也可能是
   未觀測因子或罕見正確樣本；沒有改任何 label。
6. **Adversarial AUC？** 全238欄 raw 約0.5680，prior-standardized 約0.4983；
   52欄則約0.5818→0.5083。尚無一致的強額外 shift 證據，非不存在的證明。
7. **Proxy 最有效者？** reconstruction 與 BA 分開記錄。部分 calibration 可減少
   silver-reference 誤差，但 proxy 選擇規則在第二種子沒有改善，不能稱可部署增益。
8. **Balanced Log TE／重訓／KNN 貢獻？** 查看固定外層控制表的逐項 ΔBA。
   開發切分的正向結果沒有變成可信外層增益；full/fold 方案沒有非法 TE 替換。
9. **各類與子群影響？** `level_tradeoffs.csv` 列新增判對／新增判錯，
   `subgroup_metrics.csv` 列結構及群組差異；不能只用 Level 2 Recall 掩蓋其他類損失。
10. **是否真的比0.679方案可信且更強？** **沒有證據證明更強**。本研究的新
    驗證邊界可稽核，但它既沒有重現0.679的生成流程，也沒有通過新候選 gate。

## 保存與重現

- 程式、JSON-compatible YAML、所有成功／失敗紀錄與兩種種子的 outer OOF 已保存。
- 每個正式 run 的 `oof_predictions.csv` 有15,000筆且每個id一次；test probability
  CSV只有推論機率，**不是推薦上傳的 Submission**。小型 screening 檔則明示3,750筆。
- `experiment_manifest.json` 保存資料 hash、版本、seed、fold IDs、允許訓練ID、
  feature list、命令、runtime 和 memory。原始資料、前次結果、舊提交未覆寫。
- `Kaggle_0680_Colab_Reproducible.ipynb` 會在新的輸出目錄執行完整流程並以
  同樣 gate 控制候選產生。Notebook code cells 已做語法檢查；未宣稱已在 Google
  Colab 平台另跑一次。雲端本次已實際完成流程；兩次 fresh refit 的機率一致性
  另見 repeatability.json。
- unit tests 包含 own-label TE/KNN、fold隔離、OOF id、prior normalization、
  unknown-group、可信主值保留、schema 與原始資料 hash。

## 下一步資訊增益最高的工作

先取得**實際產生0.679檔案、可在目前資料schema執行的確切版本與其OOF**。
目前可重現性缺口比追逐0.001更重要。然後用固定、事先保留且未參與選模的
新 validation 設計，測試同平滑 Balanced Log TE 或 sensor-noise calibration。
現有開發切分並未給出改標籤／調公榜權重的正當理由。不要上傳多個未證實版本
探測公榜，也不要聲稱本地0.69開發分數就是Kaggle0.680。
''',encoding='utf-8')
    # Scientific plot: distinguish outer validation from exploratory split scores.
    plt.figure(figsize=(9,4));chosen=outer[outer.seed==42]
    plt.barh(chosen['name'],chosen.balanced_accuracy,color='#267f88')
    plt.axvline(.680,color='#b45535',linestyle='--',label='.680 reference, not Kaggle measurement')
    plt.xlim(.64,.70);plt.xlabel('Strict outer OOF Balanced Accuracy');plt.legend(fontsize=8);plt.tight_layout()
    plt.savefig(REPORTS/'outer_ablation.png',dpi=180);plt.close()
    plt.figure(figsize=(5,4));cm=np.array(base42['confusion_matrix']);plt.imshow(cm,cmap='Blues')
    for i in range(3):
        for j in range(3):plt.text(j,i,str(cm[i,j]),ha='center',va='center')
    plt.xticks(range(3),[1,2,3]);plt.yticks(range(3),[1,2,3]);plt.xlabel('Predicted level');plt.ylabel('True level');plt.tight_layout();plt.savefig(REPORTS/'baseline_confusion.png',dpi=180);plt.close()
    dump(RESULTS/'final_execution_manifest.json',manifest(dict(stage_status={'S0':'partial: provided generation failed; computational reference reproduced','S1':'complete','S2':'complete screening and fixed control','S3':'complete','S4':'complete; no-go','S5':'complete','S6':'complete; no candidate passed'},
        actual_candidate_count=accepted,original_kaggle_score_user_report=.679,new_kaggle_score=None,
        all_data_labels_unchanged=True,notebook_platform_execution='not performed; syntax verified',
        intermediate_corrections=['H6 encoded column order corrected in separate output','proxy residual correlations recomputed pairwise-complete'],
        outputs={'results':str(RESULTS),'reports':str(REPORTS)})))
    print('All S0-S6 evidence reports generated; candidate count:',accepted,flush=True)

if __name__=='__main__':run()
