#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2026 ML&DL Midterm Project — 進階資料前處理、特徵工程與超參數最佳化完整管線
================================================================================
核心突破總覽 (Key Breakthroughs from Deep EDA):
1. 【雜訊特徵剔除 (Noise Feature Pruning)】：
   原始 238 個特徵中，高達 186 個 (78.2%) 是助教人工注入的純隨機雜訊特徵：
   - 33 個均勻分佈類別雜訊 (Uniform Categorical Noise)
   - 104 個連續型亂數雜訊 (Uniform / Normal Noise)
   - 19 個離散卜瓦松雜訊 (Poisson Discrete Noise, var/mean ≈ 1.0)
   - 30 個伯努利二元雜訊 (Bernoulli Binary Noise)
   若直接將 238 個特徵丟入決策樹 (如 XGBoost colsample_bytree=0.589)，每次分裂有近 8 成機率抽到純雜訊！
   只保留 52 個真實訊號特徵並將 21 個連續感測雜訊副本摺疊為去噪估計量，可大幅降低過擬合。

2. 【感測器雜訊副本去噪與異常值修復 (Proxy Denoising & Sentinel Repair)】：
   資料集中隱藏了 4 組共 21 個「真實數值特徵 + 常態雜訊」的代理欄位 (Proxy Copies)：
   - `structure_age` 的 8 個代理欄位: f_168, f_210, f_65, f_118, f_108, f_76, f_64, f_231
     -> 可精準修復 `structure_age == 999.0` (極端哨兵值) 與 5.5% 的 NaN 缺失值！
   - `storey_count` 的 6 個代理欄位: f_167, f_115, f_58, f_163, f_96, f_170
     -> 可精準修復 `storey_count == 0` 與 `99` 的異常編碼，並結合 `f_209 == 'c2'` (代表單層建築無其他樓板) 修正！
   - `footprint_area` 的 3 個代理欄位: f_41, f_106, f_194
     -> 抓出助教在 `footprint_area >= 1500` (500 的倍數) 隨機注入的「乘以 10 倍 (10x multiplier)」污染並除以 10 還原！
   - `f_165` (建築高度) 的 4 個代理欄位: f_186, f_151, f_20, f_80
     -> 抓出 `f_165 < 0` 的「正負號反轉 (Sign-Flip)」污染 (取絕對值即可還原) 以及 `96, 100` 與 NaN 缺失值！

3. 【三層地理階層結構挖掘 (3-Level Nested Geographic Hierarchy)】：
   經卡方檢定與階層驗證發現：
   `f_90` (11 個大區) -> `f_109` (110 個中區) -> `f_141` (945 個微區/村落)
   構成嚴格的 100% 巢狀地理階層，且 `f_141` 與受損等級的 Cramér's V 高達 0.5882（全資料集最強特徵）！
   若把 `f_141` (1..945) 當作一般連續整數丟入 XGBoost，樹模型只能做無意義的數值大小切分。
   本管線實作「Out-of-Fold (OOF) 階層式經驗貝氏目標編碼 (Hierarchical Empirical Bayes Target Encoding)」
   以及「村落群組統計特徵 (Group Aggregations)」，徹底釋放地理微區與建築結構的交互預測力。

4. 【類別先驗偏移校正與 1:1:1 精確基數匹配 (Prior Shift & Exact Cardinality Matching)】：
   訓練集類別比例為 1 : 2 : 4 (14.3% : 28.6% : 57.1%)，但測試集為嚴格平衡的 1 : 1 : 1 (各 1,700 筆)。
   除了貝氏先驗偏移校正 P_test(y|x) ∝ P_train(y|x) / P_train(y)^α 外，本管線實作對偶乘子座標上升與邊界微調
   (Exact Cardinality Constrained Assignment)，確保測試集預測完美符合 1700 : 1700 : 1700 分佈。
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import special, optimize

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. 真實訊號特徵清單定義 (由 EDA 嚴格統計檢定篩選出之 52 個真實特徵)
# ==============================================================================
REAL_CAT_COLS = [
    "roofing_kind", "terrain_slope", "f_13", "f_45",
    "f_62", "f_69", "f_105", "f_209"
]
GEO_COLS = ["f_90", "f_109", "f_141"]
DISC_COLS = ["structure_age", "storey_count", "footprint_area", "f_165", "f_119"]

# 4 組帶有常態雜訊之連續代理欄位 (Proxy Copies)
AGE_PROXIES = ["f_168", "f_210", "f_65", "f_118", "f_108", "f_76", "f_64", "f_231"]
STOREY_PROXIES = ["f_167", "f_115", "f_58", "f_163", "f_96", "f_170"]
AREA_PROXIES = ["f_41", "f_106", "f_194"]
HEIGHT_PROXIES = ["f_186", "f_151", "f_20", "f_80"]

# 真實二元結構與用途指標 (Superstructure & Secondary Use Flags)
REAL_BIN_COLS = [
    "f_182", "f_214", "f_199", "f_103", "f_113", "f_124", "f_175",
    "f_23", "f_104", "f_226", "f_1", "f_75", "f_26", "f_127", "f_48"
]


# ==============================================================================
# 2. 評估指標與後處理函式 (Balanced Accuracy, Prior Shift, Exact Cardinality)
# ==============================================================================
def compute_balanced_accuracy(y_true, y_pred):
    """計算三類別平均召回率 (Macro-Recall / Balanced Accuracy)，完全對應 1:1:1 測試集 Accuracy。"""
    recalls = [float(np.mean(y_pred[y_true == k] == k)) for k in range(3)]
    return float(np.mean(recalls)), recalls


def apply_prior_shift(probs, train_priors, alpha=1.0, class_mults=None):
    """
    貝氏先驗偏移校正：
    P_test(y=k|x) ∝ P_train(y=k|x) / (P_train(y=k) ^ alpha) * class_mults[k]
    """
    w = (1.0 / (train_priors ** alpha))[None, :]
    if class_mults is not None:
        w = w * np.asarray(class_mults)[None, :]
    p_adj = probs * w
    p_adj /= np.sum(p_adj, axis=1, keepdims=True)
    return p_adj


def enforce_exact_cardinality(probs, target_counts=None):
    """
    精確基數匹配 (Exact Cardinality Post-Processing)：
    已知測試集三類別嚴格為 1:1:1 (1700, 1700, 1700)，求解對偶平移量 (lambda_0, lambda_1, lambda_2)
    最大化總對數似然 sum_i log P(y_i = c_i | x_i)，並保證每個類別預測數量精確等於 1,700 筆。
    """
    N = len(probs)
    if target_counts is None:
        target_counts = [N // 3, N // 3, N - 2 * (N // 3)]
    log_p = np.log(np.clip(probs, 1e-12, 1.0))

    lam = np.zeros(3, dtype=np.float64)
    for step in range(400):
        adj = log_p + lam[None, :]
        pred = np.argmax(adj, axis=1)
        cnts = np.array([np.sum(pred == k) for k in range(3)])
        diff = cnts - np.array(target_counts)
        if np.max(np.abs(diff)) <= 1:
            break
        lr = 0.06 / (1.0 + 0.015 * step)
        lam -= lr * (diff / (N / 3.0))
        lam -= lam[0]

    # 針對邊界樣本做最小成本交換，確保完全符合目標筆數
    adj = log_p + lam[None, :]
    pred = np.argmax(adj, axis=1)
    for _ in range(50):
        cnts = [int(np.sum(pred == k)) for k in range(3)]
        if cnts == list(target_counts):
            break
        over = [k for k in range(3) if cnts[k] > target_counts[k]]
        under = [k for k in range(3) if cnts[k] < target_counts[k]]
        if not over or not under:
            break
        k_from, k_to = over[0], under[0]
        idx_from = np.where(pred == k_from)[0]
        costs = adj[idx_from, k_from] - adj[idx_from, k_to]
        pred[idx_from[np.argmin(costs)]] = k_to
    return pred, lam


def optimize_oof_multipliers(oof_probs_shifted, y_true, folds_arr):
    """在 Out-of-Fold 上搜尋最佳類別權重微調乘子 (w0=1.0, w1, w2)。"""
    best_oof_preds = np.zeros(len(y_true), dtype=np.int64)
    mults_list = []
    for f in np.unique(folds_arr):
        tr_m = (folds_arr != f)
        va_m = (folds_arr == f)
        best_sc = -1.0
        best_w = np.array([1.0, 1.0, 1.0])
        for w1 in np.linspace(0.88, 1.15, 28):
            for w2 in np.linspace(0.88, 1.15, 28):
                w_vec = np.array([1.0, w1, w2])[None, :]
                sc, _ = compute_balanced_accuracy(y_true[tr_m], np.argmax(oof_probs_shifted[tr_m] * w_vec, axis=1))
                if sc > best_sc:
                    best_sc = sc
                    best_w = np.array([1.0, w1, w2])
        mults_list.append(best_w)
        best_oof_preds[va_m] = np.argmax(oof_probs_shifted[va_m] * best_w[None, :], axis=1)
    return np.mean(mults_list, axis=0), best_oof_preds


# ==============================================================================
# 3. 核心資料前處理與特徵工程管線 (Data Cleaning & Feature Engineering)
# ==============================================================================
def build_clean_and_engineered_features(train_df, test_df, y_all, folds_arr):
    """
    執行完整資料清洗、異常值修復、雜訊剔除、結構物理特徵衍生與階層式 OOF 目標編碼。
    """
    N_tr = len(train_df)
    N_te = len(test_df)
    df = pd.concat([train_df.drop(columns=["id", "label"]), test_df.drop(columns=["id"])], ignore_index=True)
    N = len(df)
    out = {}

    # --- Step 3.1: 計算 4 組代理變數 (Proxy Copies) 的去噪平均值與標準差 ---
    age_p_mean = df[AGE_PROXIES].mean(axis=1).values
    age_p_std = df[AGE_PROXIES].std(axis=1).values
    storey_p_mean = df[STOREY_PROXIES].mean(axis=1).values
    area_p_mean = df[AREA_PROXIES].mean(axis=1).values

    # f_80 含有少量重尾極端值，先截斷至 [-40, 80] 再做反變異數加權平均
    h_df = df[HEIGHT_PROXIES].copy()
    h_df["f_80"] = h_df["f_80"].clip(-40, 80)
    height_p_mean = (0.28 * (h_df["f_186"] + h_df["f_151"] + h_df["f_20"]) + 0.16 * h_df["f_80"]).values

    out["age_proxy_mean"] = age_p_mean
    out["age_proxy_std"] = age_p_std
    out["storey_proxy_mean"] = storey_p_mean
    out["area_proxy_mean"] = area_p_mean
    out["height_proxy_mean"] = height_p_mean

    # --- Step 3.2: 修復 4 大核心數值欄位的哨兵異常值與缺失值 ---
    # (a) structure_age: 修復 999.0 與 NaN
    raw_age = df["structure_age"].values.copy()
    out["is_age_999"] = (raw_age == 999.0).astype(np.float32)
    out["is_age_nan"] = np.isnan(raw_age).astype(np.float32)
    clean_age = raw_age.copy()
    bad_age = np.isnan(raw_age) | (raw_age > 200)
    clean_age[bad_age] = np.clip(np.round(age_p_mean[bad_age] / 10.0) * 10.0, 0, 100)
    out["clean_age"] = clean_age
    out["log_clean_age"] = np.log1p(np.maximum(clean_age, 0))
    out["age_blended"] = 0.8 * clean_age + 0.2 * np.clip(age_p_mean, 0, 120)

    # (b) storey_count: 修復 0 與 99 異常值
    raw_storey = df["storey_count"].values.copy()
    clean_storey = raw_storey.astype(np.float32).copy()
    bad_storey = (raw_storey == 0) | (raw_storey >= 99)
    clean_storey[bad_storey] = np.clip(np.round(storey_p_mean[bad_storey]), 1, 7)
    clean_storey[bad_storey & (df["f_209"].values == "c2") & (storey_p_mean < 2.2)] = 1.0
    out["clean_storey"] = clean_storey
    out["storey_blended"] = 0.8 * clean_storey + 0.2 * np.clip(storey_p_mean, 1, 7)

    # (c) footprint_area: 偵測並還原 10 倍放大污染 (1500, 2000, 2500...)
    raw_area = df["footprint_area"].values.astype(np.float32).copy()
    is_10x_area = (
        (raw_area >= 1500)
        & ((raw_area % 500) == 0)
        & (np.abs(raw_area / 10.0 - area_p_mean) < np.abs(raw_area - area_p_mean))
    )
    clean_area = np.where(is_10x_area, raw_area / 10.0, raw_area)
    out["clean_area"] = clean_area
    out["log_clean_area"] = np.log1p(clean_area)

    # (d) f_165 (建築高度): 修復負號反轉 (-28..-4) 與 96/100 異常值及 NaN
    raw_h = df["f_165"].values.copy()
    clean_h = raw_h.copy()
    flip_h = (raw_h < 0) & (raw_h >= -48)
    clean_h[flip_h] = -raw_h[flip_h]
    bad_h = np.isnan(raw_h) | (raw_h < -48) | (raw_h >= 96)
    clean_h[bad_h] = np.clip(np.round(height_p_mean[bad_h] / 4.0) * 4.0, 4, 64)
    out["clean_height"] = clean_h
    out["height_blended"] = 0.8 * clean_h + 0.2 * np.clip(height_p_mean, 4, 64)

    out["f_119_clean"] = df["f_119"].fillna(1.0).values.astype(np.float32)

    # --- Step 3.3: 建築物理與結構力學衍生特徵 ---
    out["height_per_storey"] = out["clean_height"] / np.maximum(out["clean_storey"], 1.0)
    out["slenderness"] = out["clean_height"] / np.sqrt(np.maximum(out["clean_area"], 50.0))
    out["area_per_storey"] = out["clean_area"] / np.maximum(out["clean_storey"], 1.0)
    out["volume_proxy"] = np.log1p(out["clean_area"] * out["clean_height"])

    # --- Step 3.4: 真實二元特徵與缺失值條件插補 ---
    for c in REAL_BIN_COLS:
        out[c] = df[c].values.astype(np.float32)
    f103_grp = df.groupby(["f_141", "f_113"])["f_103"].transform("mean").fillna(df.groupby("f_113")["f_103"].transform("mean"))
    f175_grp = df.groupby(["f_141", "f_104"])["f_175"].transform("mean").fillna(df.groupby("f_104")["f_175"].transform("mean"))
    out["f_103_imp"] = df["f_103"].fillna(f103_grp).values.astype(np.float32)
    out["f_175_imp"] = df["f_175"].fillna(f175_grp).values.astype(np.float32)

    super_cols = ["f_182", "f_214", "f_199", "f_103_imp", "f_113", "f_124", "f_175_imp", "f_23", "f_104", "f_226", "f_1"]
    out["superstructure_sum"] = sum(out[c] for c in super_cols)
    out["weak_masonry_score"] = (
        out["f_182"] * 2.0 + out["f_124"] * 1.5 + out["f_175_imp"] * 1.0
        - out["f_214"] * 2.0 - out["f_199"] * 2.5
    )

    # --- Step 3.5: 8 個真實類別特徵編碼 ---
    f209_clean = df["f_209"].copy()
    f209_clean = np.where(f209_clean.isnull() & (out["clean_storey"] == 1), "c2", f209_clean)
    f209_clean_s = pd.Series(f209_clean, index=df.index).fillna("c0")

    for c in REAL_CAT_COLS:
        s = df[c].fillna("MISSING").astype(str)
        vc = s.value_counts()
        out[f"{c}_freq"] = (s.map(vc).values / N).astype(np.float32)
        for val, cnt in vc.items():
            if cnt >= 25:
                out[f"{c}_is_{val}"] = (s.values == val).astype(np.float32)

    out_df = pd.DataFrame(out, index=df.index)

    # --- Step 3.6: 地理階層群組統計特徵 (f_90 -> f_109 -> f_141, 零標籤洩漏) ---
    agg_cols = {}
    for g_col in ["f_90", "f_109", "f_141"]:
        agg_cols[f"{g_col}_count"] = np.log1p(df.groupby(g_col)[g_col].transform("count").values).astype(np.float32)
        for num_c in ["clean_age", "clean_storey", "clean_area", "clean_height", "f_182", "f_214", "f_199", "weak_masonry_score"]:
            g_mean = out_df.groupby(df[g_col])[num_c].transform("mean").values.astype(np.float32)
            agg_cols[f"{g_col}_mean_{num_c}"] = g_mean
            if num_c in ["clean_age", "clean_height", "clean_area", "weak_masonry_score"]:
                agg_cols[f"{g_col}_diff_{num_c}"] = out_df[num_c].values - g_mean

    # --- Step 3.7: 階層式經驗貝氏 Out-of-Fold 目標編碼 (Hierarchical OOF Target Encoding) ---
    inter_df = pd.DataFrame(index=df.index)
    inter_df["f_90"] = df["f_90"]
    inter_df["f_109"] = df["f_109"]
    inter_df["f_141"] = df["f_141"]
    inter_df["inter_141_182"] = df["f_141"].astype(str) + "_" + df["f_182"].astype(str)
    inter_df["inter_141_roof"] = df["f_141"].astype(str) + "_" + df["roofing_kind"].astype(str)
    inter_df["inter_141_f62"] = df["f_141"].astype(str) + "_" + df["f_62"].fillna("M").astype(str)
    inter_df["inter_141_214"] = df["f_141"].astype(str) + "_" + df["f_214"].astype(str)
    inter_df["inter_109_182_roof"] = df["f_109"].astype(str) + "_" + df["f_182"].astype(str) + "_" + df["roofing_kind"].astype(str)
    inter_df["inter_109_f62"] = df["f_109"].astype(str) + "_" + df["f_62"].fillna("M").astype(str)
    inter_df["inter_109_f69"] = df["f_109"].astype(str) + "_" + df["f_69"].fillna("M").astype(str)
    inter_df["inter_109_storey"] = df["f_109"].astype(str) + "_" + pd.Series(out["clean_storey"]).astype(int).astype(str)
    inter_df["inter_struct_full"] = (
        df["roofing_kind"].astype(str) + "_" +
        df["f_62"].fillna("M").astype(str) + "_" +
        df["f_69"].fillna("M").astype(str) + "_" +
        f209_clean_s.astype(str) + "_" +
        df["f_182"].astype(str) + "_" +
        df["f_214"].astype(str)
    )
    inter_df["inter_90_struct"] = df["f_90"].astype(str) + "_" + inter_df["inter_struct_full"]
    inter_df["inter_109_struct"] = df["f_109"].astype(str) + "_" + inter_df["inter_struct_full"]

    Y_tr_oh = np.eye(3, dtype=np.float64)[y_all]
    global_prior = Y_tr_oh.mean(axis=0)

    def add_hierarchical_te(col_child, col_parent, m_smooth, prefix):
        te_tr = np.zeros((N_tr, 3), dtype=np.float32)
        te_te = np.zeros((N_te, 3), dtype=np.float32)

        def compute_te(fit_idx, apply_idx):
            s_child_fit = inter_df[col_child].iloc[fit_idx]
            s_child_app = inter_df[col_child].iloc[apply_idx]
            y_fit = Y_tr_oh[fit_idx]
            if col_parent is not None:
                s_par_fit = inter_df[col_parent].iloc[fit_idx]
                s_par_app = inter_df[col_parent].iloc[apply_idx]
                par_df = pd.DataFrame(y_fit, index=s_par_fit).groupby(level=0).agg(["count", "mean"])
                par_cnt = par_df[(0, "count")]
                par_means = np.column_stack([par_df[(k, "mean")] for k in range(3)])
                par_smooth = (par_means * par_cnt.values[:, None] + 8.0 * global_prior[None, :]) / (par_cnt.values[:, None] + 8.0)
                par_map = pd.DataFrame(par_smooth, index=par_df.index)
                prior_app = par_map.reindex(s_par_app.values).values
                nan_m = np.isnan(prior_app[:, 0])
                if np.any(nan_m):
                    prior_app[nan_m] = global_prior
            else:
                prior_app = np.tile(global_prior, (len(apply_idx), 1))

            ch_df = pd.DataFrame(y_fit, index=s_child_fit).groupby(level=0).agg(["count", "sum"])
            ch_cnt = ch_df[(0, "count")].reindex(s_child_app.values).fillna(0.0).values[:, None]
            ch_sum = np.column_stack([ch_df[(k, "sum")].reindex(s_child_app.values).fillna(0.0).values for k in range(3)])
            return (ch_sum + m_smooth * prior_app) / (ch_cnt + m_smooth)

        for f in np.unique(folds_arr):
            fit_idx = np.where(folds_arr != f)[0]
            val_idx = np.where(folds_arr == f)[0]
            te_tr[val_idx] = compute_te(fit_idx, val_idx)
        te_te[:] = compute_te(np.arange(N_tr), np.arange(N_tr, N))
        te_all = np.vstack([te_tr, te_te])
        for k in range(3):
            agg_cols[f"{prefix}_te_c{k}"] = te_all[:, k]
        agg_cols[f"{prefix}_te_expected"] = te_all[:, 1] + 2.0 * te_all[:, 2]
        p_bal = apply_prior_shift(te_all, global_prior, alpha=1.0)
        for k in range(3):
            agg_cols[f"{prefix}_te_bal_c{k}"] = np.log(np.clip(p_bal[:, k], 1e-4, 1.0)).astype(np.float32)

    add_hierarchical_te("f_90", None, 10.0, "f90")
    add_hierarchical_te("f_109", "f_90", 7.0, "f109")
    add_hierarchical_te("f_141", "f_109", 3.0, "f141_m3")
    add_hierarchical_te("f_141", "f_109", 10.0, "f141_m10")
    add_hierarchical_te("f_141", "f_109", 25.0, "f141_m25")
    add_hierarchical_te("inter_141_182", "f_141", 3.5, "i141_182")
    add_hierarchical_te("inter_141_roof", "f_141", 4.0, "i141_roof")
    add_hierarchical_te("inter_141_f62", "f_141", 4.0, "i141_f62")
    add_hierarchical_te("inter_141_214", "f_141", 4.0, "i141_214")
    add_hierarchical_te("inter_109_182_roof", "f_109", 5.0, "i109_182_roof")
    add_hierarchical_te("inter_109_f62", "f_109", 5.0, "i109_f62")
    add_hierarchical_te("inter_109_f69", "f_109", 5.0, "i109_f69")
    add_hierarchical_te("inter_109_storey", "f_109", 5.0, "i109_storey")
    add_hierarchical_te("inter_struct_full", None, 10.0, "istruct")
    add_hierarchical_te("inter_90_struct", "f_90", 6.0, "i90_struct")
    add_hierarchical_te("inter_109_struct", "inter_90_struct", 5.0, "i109_struct")

    full_df = pd.concat([out_df, pd.DataFrame(agg_cols, index=df.index)], axis=1)
    return full_df.iloc[:N_tr].copy(), full_df.iloc[N_tr:].copy()


# ==============================================================================
# 4. 內建高效率模型群 (支援無外部套件環境，並自動整合 XGBoost / LightGBM / CatBoost)
# ==============================================================================
def fit_predict_spline_softmax(X_tr_df, y_tr, folds_arr, X_te_df, train_priors, l2_reg=35.0, sample_weights=None, n_iter=45):
    """Böhning 二次上界預條件牛頓多類別邏輯斯迴歸 (含二階非線性展開)。"""
    X_tr_raw = X_tr_df.fillna(0.0).values.astype(np.float32)
    med = np.median(X_tr_raw, axis=0)
    iqr = np.percentile(X_tr_raw, 75, axis=0) - np.percentile(X_tr_raw, 25, axis=0)
    scale = np.where(iqr > 1e-5, iqr, np.std(X_tr_raw, axis=0))
    scale = np.where(scale > 1e-5, scale, 1.0)
    X_tr_s = np.clip((X_tr_raw - med) / scale, -5.0, 5.0)
    X_tr = np.hstack([X_tr_s, (X_tr_s ** 2) * 0.15])
    N, D = X_tr.shape
    X_b = np.hstack([X_tr, np.ones((N, 1), dtype=np.float32)]).astype(np.float32)

    X_te_raw = X_te_df.fillna(0.0).values.astype(np.float32)
    X_te_s = np.clip((X_te_raw - med) / scale, -5.0, 5.0)
    X_te = np.hstack([X_te_s, (X_te_s ** 2) * 0.15])
    X_te_b = np.hstack([X_te, np.ones((len(X_te), 1), dtype=np.float32)]).astype(np.float32)

    D_b = D + 1
    Y_oh = np.eye(3, dtype=np.float32)[y_tr]
    sw = np.ones(N, dtype=np.float32) if sample_weights is None else sample_weights.astype(np.float32)

    oof_p = np.zeros((N, 3), dtype=np.float64)
    te_p = np.zeros((len(X_te_df), 3), dtype=np.float64)
    reg_vec = np.full(D_b, l2_reg, dtype=np.float32)
    reg_vec[-1] = 1e-3

    n_folds = len(np.unique(folds_arr))
    for f in np.unique(folds_arr):
        tr_m = (folds_arr != f)
        va_m = (folds_arr == f)
        Xf, Yf, wf = X_b[tr_m], Y_oh[tr_m], sw[tr_m, None]
        Xw = Xf * np.sqrt(wf * 0.24)
        H_inv = np.linalg.inv((Xw.T @ Xw) + np.diag(reg_vec)).astype(np.float32)
        W = np.zeros((D_b, 3), dtype=np.float32)
        W[-1, :] = np.log(train_priors + 1e-12).astype(np.float32)
        for _ in range(n_iter):
            logits = Xf @ W
            logits -= np.max(logits, axis=1, keepdims=True)
            exp_l = np.exp(logits)
            P = exp_l / np.sum(exp_l, axis=1, keepdims=True)
            W -= 1.25 * (H_inv @ (Xf.T @ (wf * (P - Yf)) + reg_vec[:, None] * W))
        oof_p[va_m] = special.softmax(X_b[va_m] @ W, axis=1)
        te_p += special.softmax(X_te_b @ W, axis=1) / n_folds
    return oof_p, te_p


def fit_predict_fast_hist_gbdt(
    X_tr_df, y_tr, folds_arr, X_te_df, train_priors,
    n_estimators=120, lr=0.06, max_depth=5, max_leaves=20, colsample=0.65, B=32
):
    """純 NumPy + SciPy Sparse 矩陣加速之直方圖梯度提升決策樹 (Hist-GBDT)。"""
    X_tr = X_tr_df.values.astype(np.float32)
    X_te = X_te_df.values.astype(np.float32)
    N, M = X_tr.shape

    X_all = np.vstack([X_tr, X_te])
    X_all_bin = np.zeros_like(X_all, dtype=np.int32)
    for j in range(M):
        col = X_all[:, j]
        valid = col[~np.isnan(col)]
        edges = np.unique(np.percentile(valid, np.linspace(0, 100, B - 1)))
        if len(edges) >= B - 1:
            edges = edges[:B - 2]
        b = np.where(np.isnan(col), 0, np.searchsorted(edges, col, side="right") + 1)
        X_all_bin[:, j] = np.clip(b, 0, B - 1)

    X_tr_bin = X_all_bin[:N]
    X_te_bin = X_all_bin[N:]

    oof_p = np.zeros((N, 3), dtype=np.float64)
    te_p = np.zeros((len(X_te), 3), dtype=np.float64)
    Y_oh = np.eye(3, dtype=np.float32)[y_tr]
    init_logits = np.log(train_priors + 1e-12).astype(np.float32)
    n_sub = max(1, int(M * colsample))
    n_folds = len(np.unique(folds_arr))

    for f in np.unique(folds_arr):
        rng = np.random.RandomState(42 + int(f) * 10)
        tr_m = (folds_arr != f)
        va_m = (folds_arr == f)
        X_f_bin = X_tr_bin[tr_m]
        X_v_bin = X_tr_bin[va_m]
        Y_f = Y_oh[tr_m]
        Nf = len(X_f_bin)

        rows = (X_f_bin + (np.arange(M) * B)[None, :]).ravel()
        cols = np.repeat(np.arange(Nf), M)
        S_all = sp.csr_matrix((np.ones(Nf * M, dtype=np.float32), (rows, cols)), shape=(M * B, Nf))

        F_tr = np.tile(init_logits, (Nf, 1))
        F_va = np.tile(init_logits, (len(X_v_bin), 1))
        F_te = np.tile(init_logits, (len(X_te_bin), 1))

        for _ in range(n_estimators):
            P = special.softmax(F_tr, axis=1)
            G_all = P - Y_f
            H_all = P * (1.0 - P)

            for k in range(3):
                g, h = G_all[:, k], H_all[:, k]
                feat_idx = np.sort(rng.choice(M, size=n_sub, replace=False))
                row_indices = (feat_idx[:, None] * B + np.arange(B)[None, :]).ravel()
                S_sub = S_all[row_indices, :]
                K_f = len(feat_idx)

                leaf_tr = np.zeros(Nf, dtype=np.int32)
                leaf_va = np.zeros(len(X_v_bin), dtype=np.int32)
                leaf_te = np.zeros(len(X_te_bin), dtype=np.int32)
                n_leaves = 1

                for _ in range(max_depth):
                    G_mat = np.zeros((Nf, n_leaves), dtype=np.float32)
                    H_mat = np.zeros((Nf, n_leaves), dtype=np.float32)
                    G_mat[np.arange(Nf), leaf_tr] = g
                    H_mat[np.arange(Nf), leaf_tr] = h

                    G_tot = G_mat.sum(axis=0)
                    H_tot = H_mat.sum(axis=0)

                    GH_bins = (S_sub @ np.hstack([G_mat, H_mat])).reshape(K_f, B, 2 * n_leaves)
                    G_left = np.cumsum(GH_bins[:, :, :n_leaves], axis=1)[:, :-1, :]
                    H_left = np.cumsum(GH_bins[:, :, n_leaves:], axis=1)[:, :-1, :]
                    G_right = G_tot[None, None, :] - G_left
                    H_right = H_tot[None, None, :] - H_left

                    valid_split = (H_left >= 12.0) & (H_right >= 12.0)
                    gain = np.where(
                        valid_split,
                        0.5 * (
                            (G_left ** 2) / (H_left + 3.0)
                            + (G_right ** 2) / (H_right + 3.0)
                            - (G_tot[None, None, :] ** 2) / (H_tot[None, None, :] + 3.0)
                        ) - 0.8,
                        -1e9,
                    )
                    gain_flat = gain.reshape(-1, n_leaves)
                    best_flat_idx = np.argmax(gain_flat, axis=0)
                    best_gains = gain_flat[best_flat_idx, np.arange(n_leaves)]
                    if np.max(best_gains) <= 0.0:
                        break

                    new_tr, new_va, new_te = leaf_tr.copy(), leaf_va.copy(), leaf_te.copy()
                    next_l = n_leaves
                    for l_id in range(n_leaves):
                        if best_gains[l_id] > 0.0 and next_l < max_leaves:
                            f_loc = best_flat_idx[l_id] // (B - 1)
                            b_th = best_flat_idx[l_id] % (B - 1)
                            f_glb = feat_idx[f_loc]
                            new_tr[(leaf_tr == l_id) & (X_f_bin[:, f_glb] > b_th)] = next_l
                            new_va[(leaf_va == l_id) & (X_v_bin[:, f_glb] > b_th)] = next_l
                            new_te[(leaf_te == l_id) & (X_te_bin[:, f_glb] > b_th)] = next_l
                            next_l += 1
                    leaf_tr, leaf_va, leaf_te, n_leaves = new_tr, new_va, new_te, next_l
                    if n_leaves >= max_leaves:
                        break

                G_fin = np.bincount(leaf_tr, weights=g, minlength=n_leaves)
                H_fin = np.bincount(leaf_tr, weights=h, minlength=n_leaves)
                l_vals = -lr * G_fin / (H_fin + 3.0)
                F_tr[:, k] += l_vals[leaf_tr]
                F_va[:, k] += l_vals[leaf_va]
                F_te[:, k] += l_vals[leaf_te]

        oof_p[va_m] = special.softmax(F_va, axis=1)
        te_p += special.softmax(F_te, axis=1) / n_folds
    return oof_p, te_p


def try_fit_external_gbdt_models(X_tr_df, y_tr, folds_arr, X_te_df, train_priors):
    """
    若執行環境已安裝 xgboost / lightgbm / catboost，自動啟用並加入 Ensemble！
    並針對去噪後的特徵集採用最佳化抗過擬合超參數配置。
    """
    ext_oof_list = []
    ext_te_list = []
    n_folds = len(np.unique(folds_arr))

    # 1. XGBoost (若環境中有安裝)
    try:
        from xgboost import XGBClassifier
        print("[External Model] 偵測到 XGBoost，啟動 5-Fold 訓練 (最佳化抗過擬合參數)...")
        oof_xgb = np.zeros((len(X_tr_df), 3), dtype=np.float64)
        te_xgb = np.zeros((len(X_te_df), 3), dtype=np.float64)

        # 去噪且加入階層 Target Encoding 後，樹深應從 9 調降至 5~6，並拉高 min_child_weight 與 reg_lambda
        xgb_params = {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 16,
            "subsample": 0.85,
            "colsample_bytree": 0.65,
            "gamma": 1.2,
            "reg_alpha": 1.5,
            "reg_lambda": 5.0,
            "objective": "multi:softprob",
            "num_class": 3,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "early_stopping_rounds": 50,
            "random_state": 42,
            "n_jobs": -1,
        }
        for f in np.unique(folds_arr):
            tr_m = (folds_arr != f)
            va_m = (folds_arr == f)
            model = XGBClassifier(**xgb_params)
            model.fit(
                X_tr_df.iloc[tr_m], y_tr[tr_m],
                eval_set=[(X_tr_df.iloc[va_m], y_tr[va_m])],
                verbose=False
            )
            oof_xgb[va_m] = model.predict_proba(X_tr_df.iloc[va_m])
            te_xgb += model.predict_proba(X_te_df) / n_folds

        p_xgb_tr = apply_prior_shift(oof_xgb, train_priors, alpha=1.0)
        p_xgb_te = apply_prior_shift(te_xgb, train_priors, alpha=1.0)
        sc, rec = compute_balanced_accuracy(y_tr, np.argmax(p_xgb_tr, axis=1))
        print(f"  -> XGBoost 5-Fold Balanced Acc: {sc:.5f} (Recalls: {[round(r, 4) for r in rec]})")
        ext_oof_list.append(("xgb", p_xgb_tr, 0.45))
        ext_te_list.append(("xgb", p_xgb_te, 0.45))
    except ImportError:
        pass

    # 2. LightGBM (若環境中有安裝)
    try:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation
        print("[External Model] 偵測到 LightGBM，啟動 5-Fold 訓練...")
        oof_lgb = np.zeros((len(X_tr_df), 3), dtype=np.float64)
        te_lgb = np.zeros((len(X_te_df), 3), dtype=np.float64)

        lgb_params = {
            "n_estimators": 1000,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 35,
            "subsample": 0.85,
            "subsample_freq": 1,
            "colsample_bytree": 0.65,
            "reg_alpha": 1.5,
            "reg_lambda": 5.0,
            "objective": "multiclass",
            "num_class": 3,
            "random_state": 1042,
            "n_jobs": -1,
            "verbose": -1,
        }
        for f in np.unique(folds_arr):
            tr_m = (folds_arr != f)
            va_m = (folds_arr == f)
            model = LGBMClassifier(**lgb_params)
            model.fit(
                X_tr_df.iloc[tr_m], y_tr[tr_m],
                eval_set=[(X_tr_df.iloc[va_m], y_tr[va_m])],
                callbacks=[early_stopping(50, verbose=False), log_evaluation(0)]
            )
            oof_lgb[va_m] = model.predict_proba(X_tr_df.iloc[va_m])
            te_lgb += model.predict_proba(X_te_df) / n_folds

        p_lgb_tr = apply_prior_shift(oof_lgb, train_priors, alpha=1.0)
        p_lgb_te = apply_prior_shift(te_lgb, train_priors, alpha=1.0)
        sc, rec = compute_balanced_accuracy(y_tr, np.argmax(p_lgb_tr, axis=1))
        print(f"  -> LightGBM 5-Fold Balanced Acc: {sc:.5f} (Recalls: {[round(r, 4) for r in rec]})")
        ext_oof_list.append(("lgb", p_lgb_tr, 0.35))
        ext_te_list.append(("lgb", p_lgb_te, 0.35))
    except ImportError:
        pass

    # 3. CatBoost (若環境中有安裝)
    try:
        from catboost import CatBoostClassifier
        print("[External Model] 偵測到 CatBoost，啟動 5-Fold 訓練...")
        oof_cat = np.zeros((len(X_tr_df), 3), dtype=np.float64)
        te_cat = np.zeros((len(X_te_df), 3), dtype=np.float64)

        cat_params = {
            "iterations": 1200,
            "learning_rate": 0.04,
            "depth": 6,
            "l2_leaf_reg": 6.0,
            "loss_function": "MultiClass",
            "random_seed": 2042,
            "verbose": False,
        }
        for f in np.unique(folds_arr):
            tr_m = (folds_arr != f)
            va_m = (folds_arr == f)
            model = CatBoostClassifier(**cat_params)
            model.fit(
                X_tr_df.iloc[tr_m], y_tr[tr_m],
                eval_set=(X_tr_df.iloc[va_m], y_tr[va_m]),
                early_stopping_rounds=60,
                verbose=False
            )
            oof_cat[va_m] = model.predict_proba(X_tr_df.iloc[va_m])
            te_cat += model.predict_proba(X_te_df) / n_folds

        p_cat_tr = apply_prior_shift(oof_cat, train_priors, alpha=1.0)
        p_cat_te = apply_prior_shift(te_cat, train_priors, alpha=1.0)
        sc, rec = compute_balanced_accuracy(y_tr, np.argmax(p_cat_tr, axis=1))
        print(f"  -> CatBoost 5-Fold Balanced Acc: {sc:.5f} (Recalls: {[round(r, 4) for r in rec]})")
        ext_oof_list.append(("cat", p_cat_tr, 0.30))
        ext_te_list.append(("cat", p_cat_te, 0.30))
    except ImportError:
        pass

    return ext_oof_list, ext_te_list


# ==============================================================================
# 5. 主執行流程 (Main Pipeline)
# ==============================================================================
def main():
    t0 = time.time()
    data_dir = os.environ.get("DATA_DIR", ".")
    if not os.path.exists(os.path.join(data_dir, "train.csv")):
        alt_dir = "/working_dir/c_e728d7953746c430/user_uploaded_files/ttl=63d/user_uploaded"
        if os.path.exists(os.path.join(alt_dir, "train.csv")):
            data_dir = alt_dir

    print(f"1. 讀取資料集 (來自 {data_dir})...")
    train_df = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_dir, "test.csv"))

    y_all = (train_df["label"].values - 1).astype(np.int64)
    train_priors = np.bincount(y_all) / len(y_all)
    print(f"   Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"   Train class priors (1:2:3): {np.round(train_priors, 4)}")

    # 建立固定 5-Fold Stratified Split
    np.random.seed(42)
    folds = np.zeros(len(train_df), dtype=np.int64)
    for k in range(3):
        idx_k = np.where(y_all == k)[0]
        np.random.shuffle(idx_k)
        for f in range(5):
            folds[idx_k[f::5]] = f

    print("\n2. 執行進階資料清洗、代理變數去噪、雜訊特徵剔除與階層式 OOF 目標編碼...")
    X_tr_fe, X_te_fe = build_clean_and_engineered_features(train_df, test_df, y_all, folds)
    print(f"   完成特徵工程！精煉特徵維度: {X_tr_fe.shape[1]} 維 (耗時 {time.time() - t0:.2f}s)")

    print("\n3. 訓練多模型 Ensemble (5-Fold Stratified CV)...")
    # 模型 A: 二階非線性 Böhning MM Softmax
    oof_mm, te_mm = fit_predict_spline_softmax(X_tr_fe, y_all, folds, X_te_fe, train_priors, l2_reg=35.0)
    p_mm_tr = apply_prior_shift(oof_mm, train_priors, alpha=1.0)
    p_mm_te = apply_prior_shift(te_mm, train_priors, alpha=1.0)
    acc_mm, rec_mm = compute_balanced_accuracy(y_all, np.argmax(p_mm_tr, axis=1))
    print(f"   [Model A] SplineSoftmax (FE) 5-Fold Balanced Acc: {acc_mm:.5f} (Recalls: {[round(r, 4) for r in rec_mm]})")

    # 模型 B: 高訊號特徵子集之直方圖梯度提升決策樹 (FastHistGBDT)
    gbdt_cols = [
        "clean_age", "age_blended", "is_age_999", "clean_storey", "storey_blended",
        "clean_area", "clean_height", "height_per_storey", "slenderness", "volume_proxy",
        "f_119_clean", "f_182", "f_214", "f_199", "f_103_imp", "f_113", "f_124", "f_175_imp",
        "superstructure_sum", "weak_masonry_score", "roofing_kind_freq", "f_62_freq", "f_69_freq", "f_209_freq",
        "f_141_count", "f_141_mean_clean_age", "f_141_mean_f_182", "f_141_diff_clean_age",
        "f90_te_c0", "f90_te_c1", "f90_te_c2",
        "f109_te_c0", "f109_te_c1", "f109_te_c2",
        "f141_m3_te_c0", "f141_m3_te_c1", "f141_m3_te_c2",
        "f141_m10_te_c0", "f141_m10_te_c1", "f141_m10_te_c2",
        "i141_182_te_c0", "i141_182_te_c1", "i141_182_te_c2",
        "i141_roof_te_c0", "i141_roof_te_c1", "i141_roof_te_c2",
        "i109_182_roof_te_c0", "i109_182_roof_te_c1", "i109_182_roof_te_c2",
        "i109_struct_te_c0", "i109_struct_te_c1", "i109_struct_te_c2",
        "istruct_te_c0", "istruct_te_c1", "istruct_te_c2"
    ]
    oof_gb, te_gb = fit_predict_fast_hist_gbdt(X_tr_fe[gbdt_cols], y_all, folds, X_te_fe[gbdt_cols], train_priors)
    p_gb_tr = apply_prior_shift(oof_gb, train_priors, alpha=1.0)
    p_gb_te = apply_prior_shift(te_gb, train_priors, alpha=1.0)
    acc_gb, rec_gb = compute_balanced_accuracy(y_all, np.argmax(p_gb_tr, axis=1))
    print(f"   [Model B] FastHistGBDT  (FE) 5-Fold Balanced Acc: {acc_gb:.5f} (Recalls: {[round(r, 4) for r in rec_gb]})")

    # 若環境中有安裝 XGBoost / LightGBM / CatBoost，自動納入加權融合
    ext_oof, ext_te = try_fit_external_gbdt_models(X_tr_fe, y_all, folds, X_te_fe, train_priors)

    if ext_oof:
        tot_w = 0.25 + 0.25 + sum(w for _, _, w in ext_oof)
        p_ens_tr = (0.25 * p_gb_tr + 0.25 * p_mm_tr + sum(w * p for _, p, w in ext_oof)) / tot_w
        p_ens_te = (0.25 * p_gb_te + 0.25 * p_mm_te + sum(w * p for _, p, w in ext_te)) / tot_w
    else:
        p_ens_tr = 0.60 * p_gb_tr + 0.40 * p_mm_tr
        p_ens_te = 0.60 * p_gb_te + 0.40 * p_mm_te

    acc_ens, rec_ens = compute_balanced_accuracy(y_all, np.argmax(p_ens_tr, axis=1))
    print(f"\n4. [Ensemble] 加權融合 5-Fold Balanced Acc: {acc_ens:.5f} (Recalls: {[round(r, 4) for r in rec_ens]})")

    # 執行精確 1:1:1 基數匹配 (1700 : 1700 : 1700)
    print("\n5. 執行測試集 1:1:1 精確基數匹配 (Exact Cardinality Post-Processing: 1700 / 1700 / 1700)...")
    te_pred_card, lam_te = enforce_exact_cardinality(p_ens_te, [1700, 1700, 1700])
    sub = pd.DataFrame({
        "id": test_df["id"].values,
        "label": te_pred_card + 1
    })
    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)
    print(f"   成功輸出預測檔案至 {out_path}！各類別預測筆數: {sub['label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()