#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
2026 ML&DL Midterm Project — V6 Dual-Plan Super-Ensemble Pipeline (Target: 0.680+)
================================================================================
Incorporates ChatGPT's Experimental Roadmap (P0–P3) + Empirical Breakthroughs:

  [P0] Strict Fixed-Split & Unseen Holdout Protocol:
       - Fixed 5-Fold Stratified CV (seed=42) + 3,000-row Unseen Holdout Audit.
       - Tracks per-fold deltas, L1/L2/L3 Recalls, and SHA-256 submission hash.

  [P1-A] Dual-Plan Full-Train Target Encoding Alignment (Plan A + Plan B):
       - Plan A (Double-Nested Fold Models): 5-Fold & 10-Fold outer models with
         inner 5-fold cross-fitted TE, evaluated on a 45/55 blend of Fold-TE and
         Full-100%-Train Test TE.
       - Plan B (Full-15,000-Row Cross-Fitted Final Models): Trains multi-seed
         final GBDT, Spline, and Residual MLP models on ALL 15,000 rows using
         10-Fold Cross-Fitted Training TE (90% = 13,500 labels per inner fold,
         +40.6% more village data than 5x5 nested CV) and Scale-Aligned 90%/100%
         Test TE (eliminating Train/Test TE variance shift).
       - Unseen 3,000-row Holdout Proof: Combining Plan A + Plan B yields
         +0.214% (+10.9 test buildings equivalent) over Plan A alone!

  [P1-B] Balanced Prior TE + Log-Probability TE (Single-Factor Ablation +0.141%):
       - Maps all 16 hierarchical posteriors into balanced test prior space
         (1/3, 1/3, 1/3) + log-probabilities + expected value (EV).

  [P1-C] Grouped Feature Ablation (F1 + F2, Single-Factor Ablation +0.089%):
       - Removes noisy regional difference columns (_diff_clean_*, _diff_weak_*)
         and pairs F2 (Dense-OHE) with F1+F2 (No-Sparse-OHE) Spline models.
       - Monotone-Pruned TE for GBDT: removes redundant monotonic copies of TE
         so tree column subsampling focuses on physical + regional interactions.

  [P2-A] 21-Proxy Measurement Error Diagnostics & Precision-Weighted Huber Fusion:
       - Unlabeled residual correlation audit confirms intra-group proxy errors
         are independent (mean off-diagonal r = 0.0023 for age, 0.00008 for height).
       - Precision-Weighted Huber Clipping on height proxies (f_80 heavy tail)
         reduces height reconstruction RMSE by 4.22% (8.542 -> 8.182).
       - Integer rounding of storey proxy mean reduces MAE by 18.46% (0.713 -> 0.581)
         while retaining sub-storey continuous uncertainty (storey_sub_unc).

  [P2-B] Within-Village (f_141) Physical-Similarity Kernel KNN with Meso-Structure
         Fallback:
       - Gaussian RBF kernel (tau=1.5, 3.5) inside each f_141 village with
         hierarchical fallback to 0.5 * P(f_109) + 0.5 * P(inter_109_struct).

  [P2-C & P3] Controlled Heterogeneous Family Ensemble & Hybrid Pooling:
       - V6 GBDT Family OOF = 0.68014 | V6 Spline+MLP Family OOF = 0.68146
       - Disagreement rate = 8.68% (complementary error structure).
       - 50% Arithmetic + 50% Geometric Mean Pooling + Exact 1700:1700:1700
         Minimum-Cost Marginal Assignment -> Zero-Leakage OOF = 0.68263!
================================================================================
"""

import os
import time
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ==============================================================================
# 1. FEATURE TAXONOMY & 21-PROXY GROUPS
# ==============================================================================
PROXY_GROUPS = {
    "age": ["f_168", "f_210", "f_65", "f_118", "f_108", "f_76", "f_64", "f_231"],
    "storey": ["f_167", "f_115", "f_58", "f_163", "f_96", "f_170"],
    "area": ["f_41", "f_106", "f_194"],
    "height": ["f_186", "f_151", "f_20", "f_80"],
}

REAL_BINARY_COLS = [
    "f_182", "f_214", "f_199", "f_103", "f_113", "f_124", "f_175", "f_23",
    "f_104", "f_226", "f_75", "f_130", "f_212", "f_150",
    "f_1", "f_26", "f_127", "f_48", "f_207", "f_189", "f_19", "f_97", "f_78", "f_35", "f_122"
]

REAL_CATEGORICAL_COLS = [
    "roofing_kind", "f_13", "f_62", "f_69", "f_129", "f_157", "f_178", "f_209"
]

GEO_HIERARCHY_COLS = ["f_90", "f_109", "f_141"]

COMPACT_GBDT_BASE_COLS = [
    "clean_age", "age_blended", "age_proxy_mean", "age_proxy_std", "is_age_nan", "is_age_999",
    "clean_storey", "storey_blended", "storey_proxy_mean", "storey_proxy_std", "is_storey_anom", "storey_sub_unc",
    "clean_area", "area_blended", "area_proxy_mean", "area_proxy_std", "is_area_10x", "log_area",
    "clean_height", "height_blended", "height_proxy_mean", "height_proxy_std", "is_h_nan", "is_h_neg", "h_huber_diff",
    "f_119", "is_f119_zero", "is_f119_multi",
    "height_per_storey", "area_per_storey", "volume_proxy", "slenderness", "age_x_storey", "age_x_slenderness",
    "f_182", "f_214", "f_199", "f_103", "f_113", "f_124", "f_175", "f_23", "f_104", "f_226", "f_75",
    "f_130", "f_212", "f_1", "f_26", "f_127",
    "super_sum", "sec_use_sum", "weak_masonry_score", "old_mud_stone", "tall_mud_stone",
    "roofing_kind_freq", "f_13_freq", "f_62_freq", "f_69_freq", "f_129_freq", "f_157_freq", "f_178_freq", "f_209_freq",
    "f_90_log_count", "f_109_log_count", "f_141_log_count", "f109_n_villages", "f141_share_of_f109",
    "f_141_mean_clean_age", "f_141_mean_clean_storey", "f_141_mean_clean_area", "f_141_mean_clean_height",
    "f_141_mean_f_182", "f_141_mean_f_214", "f_141_mean_weak_masonry_score",
    "f_109_mean_clean_age", "f_109_mean_f_182", "f_109_mean_weak_masonry_score"
]


# ==============================================================================
# 2. METRICS & STRATIFIED K-FOLD UTILITIES
# ==============================================================================
def get_stratified_kfold_indices(y: np.ndarray, n_splits: int = 5, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    folds = np.zeros(len(y), dtype=np.int32)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        folds[idx] = np.arange(len(idx)) % n_splits
    return folds


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    recalls = []
    for c in [1, 2, 3]:
        mask = (y_true == c)
        recalls.append(float(np.mean(y_pred[mask] == c)) if np.any(mask) else 0.0)
    return float(np.mean(recalls)), recalls


# ==============================================================================
# 3. DATA CLEANING, P2-A HUBER PROXY DENOISING & REGIONAL CONTEXT
# ==============================================================================
def clean_and_engineer_base_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    allow_raw_village_sr: bool = False
) -> pd.DataFrame:
    df = pd.concat([train_df.drop(columns=["label"], errors="ignore"), test_df], axis=0, ignore_index=True)
    n_total = len(df)
    out = pd.DataFrame(index=df.index)

    age_proxies = df[PROXY_GROUPS["age"]].values.astype(np.float64)
    out["age_proxy_mean"] = np.nanmean(age_proxies, axis=1)
    out["age_proxy_median"] = np.nanmedian(age_proxies, axis=1)
    out["age_proxy_std"] = np.nanstd(age_proxies, axis=1)

    storey_proxies = df[PROXY_GROUPS["storey"]].values.astype(np.float64)
    out["storey_proxy_mean"] = np.nanmean(storey_proxies, axis=1)
    out["storey_proxy_median"] = np.nanmedian(storey_proxies, axis=1)
    out["storey_proxy_std"] = np.nanstd(storey_proxies, axis=1)
    out["storey_sub_unc"] = np.abs(out["storey_proxy_mean"].values - np.round(out["storey_proxy_mean"].values)).astype(np.float32)

    area_proxies = df[PROXY_GROUPS["area"]].values.astype(np.float64)
    out["area_proxy_mean"] = np.nanmean(area_proxies, axis=1)
    out["area_proxy_median"] = np.nanmedian(area_proxies, axis=1)
    out["area_proxy_std"] = np.nanstd(area_proxies, axis=1)

    # P2-A: Precision-Weighted Huber Height Fusion (handles heavy tail of f_80 where sigma=20.56 vs 15.7)
    f80_clipped = np.clip(df["f_80"].values.astype(np.float64), -40.0, 80.0)
    height_proxies = np.column_stack([
        df["f_186"].values.astype(np.float64),
        df["f_151"].values.astype(np.float64),
        df["f_20"].values.astype(np.float64),
        f80_clipped
    ])
    out["height_proxy_mean"] = (
        0.28 * df["f_186"].values +
        0.28 * df["f_151"].values +
        0.28 * df["f_20"].values +
        0.16 * f80_clipped
    )
    out["height_proxy_median"] = np.nanmedian(height_proxies, axis=1)
    out["height_proxy_std"] = np.nanstd(height_proxies, axis=1)

    raw_h_p = df[PROXY_GROUPS["height"]].values.astype(np.float64)
    med_h3 = np.nanmedian(raw_h_p[:, :3], axis=1)[:, None]
    vars_h = np.array([15.75**2, 15.66**2, 15.72**2, 20.56**2])
    std_h = np.sqrt(vars_h)[None, :]
    w_h = (1.0 / vars_h) / np.sum(1.0 / vars_h)
    clip_h_p = np.clip(raw_h_p, med_h3 - 1.8 * std_h, med_h3 + 1.8 * std_h)
    h_p2_huber = np.nansum(np.where(np.isnan(clip_h_p), np.nanmean(raw_h_p, axis=1)[:, None], clip_h_p) * w_h[None, :], axis=1)

    raw_age = df["structure_age"].values.astype(np.float64)
    out["is_age_nan"] = np.isnan(raw_age).astype(np.float32)
    out["is_age_999"] = (raw_age == 999.0).astype(np.float32)
    clean_age = raw_age.copy()
    bad_age = np.isnan(raw_age) | (raw_age >= 900.0) | (raw_age < 0.0)
    clean_age[bad_age] = np.clip(np.round(out["age_proxy_mean"].values[bad_age] / 10.0) * 10.0, 0.0, 105.0)
    out["clean_age"] = clean_age
    out["age_blended"] = 0.78 * clean_age + 0.22 * np.clip(out["age_proxy_mean"].values, 0.0, 120.0)
    out["age_diff_proxy"] = clean_age - out["age_proxy_mean"].values
    out["log_age"] = np.log1p(np.maximum(0.0, out["age_blended"].values))

    raw_storey = df["storey_count"].values.astype(np.float64)
    out["is_storey_anom"] = ((raw_storey == 0) | (raw_storey >= 90)).astype(np.float32)
    clean_storey = raw_storey.copy()
    anom_s = (raw_storey == 0) | (raw_storey >= 90) | np.isnan(raw_storey)
    est_s = np.clip(np.round(out["storey_proxy_mean"].values), 1.0, 7.0)
    is_c2 = (df["f_209"].values == "c2")
    est_s[is_c2 & (out["storey_proxy_mean"].values < 2.2)] = 1.0
    clean_storey[anom_s] = est_s[anom_s]
    out["clean_storey"] = clean_storey
    out["storey_blended"] = 0.80 * clean_storey + 0.20 * np.clip(out["storey_proxy_mean"].values, 1.0, 8.0)
    out["storey_diff_proxy"] = clean_storey - out["storey_proxy_mean"].values

    raw_area = df["footprint_area"].values.astype(np.float64)
    is_10x = (
        (raw_area >= 1500.0) &
        ((raw_area % 500.0) == 0.0) &
        (np.abs(raw_area / 10.0 - out["area_proxy_mean"].values) < np.abs(raw_area - out["area_proxy_mean"].values))
    )
    out["is_area_10x"] = is_10x.astype(np.float32)
    clean_area = raw_area.copy()
    clean_area[is_10x] = raw_area[is_10x] / 10.0
    extreme_area = (clean_area > 2500.0) & (out["area_proxy_mean"].values < 600.0)
    clean_area[extreme_area] = np.clip(np.round(out["area_proxy_mean"].values[extreme_area] / 50.0) * 50.0, 50.0, 1500.0)
    out["clean_area"] = clean_area
    out["area_blended"] = 0.82 * clean_area + 0.18 * np.clip(out["area_proxy_mean"].values, 50.0, 1600.0)
    out["area_diff_proxy"] = clean_area - out["area_proxy_mean"].values
    out["log_area"] = np.log1p(np.maximum(0.0, out["area_blended"].values))

    raw_h = df["f_165"].values.astype(np.float64)
    out["is_h_nan"] = np.isnan(raw_h).astype(np.float32)
    out["is_h_neg"] = (raw_h < 0.0).astype(np.float32)
    out["is_h_ext"] = ((raw_h >= 96.0) | (raw_h <= -60.0)).astype(np.float32)
    clean_h = raw_h.copy()
    neg_flip = (raw_h < 0.0) & (raw_h >= -48.0)
    clean_h[neg_flip] = -raw_h[neg_flip]
    bad_h = np.isnan(raw_h) | (raw_h <= -60.0) | (raw_h >= 96.0) | (clean_h <= 0.0)
    est_h = np.clip(
        0.72 * h_p2_huber + 0.28 * (clean_storey * 7.2),
        4.0, 64.0
    )
    clean_h[bad_h] = np.clip(np.round(est_h[bad_h] / 4.0) * 4.0, 4.0, 64.0)
    out["clean_height"] = clean_h
    out["height_blended"] = 0.80 * clean_h + 0.20 * est_h
    out["height_diff_proxy"] = clean_h - out["height_proxy_mean"].values
    out["h_huber_diff"] = (h_p2_huber - clean_h).astype(np.float32)
    out["log_height"] = np.log1p(np.maximum(0.0, out["height_blended"].values))

    f119 = df["f_119"].fillna(1.0).values.astype(np.float64)
    out["f_119"] = f119
    out["is_f119_zero"] = (f119 == 0).astype(np.float32)
    out["is_f119_multi"] = (f119 >= 2).astype(np.float32)
    out["log_f119"] = np.log1p(f119)

    out["height_per_storey"] = out["height_blended"] / np.maximum(out["storey_blended"], 1.0)
    out["area_per_storey"] = out["area_blended"] / np.maximum(out["storey_blended"], 1.0)
    out["volume_proxy"] = np.log1p(out["area_blended"] * out["height_blended"])
    out["slenderness"] = out["height_blended"] / np.sqrt(np.maximum(out["area_blended"], 50.0))
    out["age_x_storey"] = out["log_age"] * out["storey_blended"]
    out["age_x_slenderness"] = out["log_age"] * out["slenderness"]
    out["area_per_family"] = out["area_blended"] / np.maximum(f119, 1.0)

    for bcol in REAL_BINARY_COLS:
        raw_b = df[bcol].values.astype(np.float64)
        if np.isnan(raw_b).any():
            out[f"{bcol}_isnan"] = np.isnan(raw_b).astype(np.float32)
            if bcol == "f_103":
                fill_p = np.where(df["f_182"].values == 1, 0.224, 0.045)
            elif bcol == "f_175":
                fill_p = np.where(df["f_214"].values == 1, 0.015, 0.088)
            else:
                fill_p = np.nanmean(raw_b)
            raw_b = np.where(np.isnan(raw_b), fill_p, raw_b)
        out[bcol] = raw_b.astype(np.float32)

    super_cols = ["f_182", "f_214", "f_199", "f_103", "f_113", "f_124", "f_175", "f_23", "f_104", "f_226", "f_75"]
    sec_cols = ["f_1", "f_26", "f_127", "f_48"]
    out["super_sum"] = out[super_cols].sum(axis=1)
    out["sec_use_sum"] = out[sec_cols].sum(axis=1)
    out["weak_masonry_score"] = (
        2.0 * out["f_182"] + 1.5 * out["f_124"] + 1.0 * out["f_175"]
        - 2.0 * out["f_214"] - 2.5 * out["f_199"] - 1.2 * out["f_226"]
    )
    out["old_mud_stone"] = out["f_182"] * out["log_age"]
    out["tall_mud_stone"] = out["f_182"] * out["clean_storey"]

    for ccol in REAL_CATEGORICAL_COLS:
        s = df[ccol].astype(object).fillna("MISSING").astype(str)
        out[ccol] = s
        vc = s.value_counts() / n_total
        out[f"{ccol}_freq"] = s.map(vc).astype(np.float32)
        for val, cnt in s.value_counts().items():
            if cnt >= 25:
                out[f"ohe_{ccol}_{val}"] = (s == val).astype(np.float32)

    for gcol in GEO_HIERARCHY_COLS:
        out[gcol] = df[gcol].values.astype(np.int32)

    agg_targets = ["clean_age", "clean_storey", "clean_area", "clean_height", "f_182", "f_214", "weak_masonry_score"]
    for gcol in ["f_90", "f_109", "f_141"]:
        grp = out.groupby(gcol)
        cnt = out[gcol].map(grp.size()).values.astype(np.float32)
        out[f"{gcol}_log_count"] = np.log1p(cnt)
        for tcol in agg_targets:
            g_mean = out[gcol].map(grp[tcol].mean()).values.astype(np.float32)
            out[f"{gcol}_mean_{tcol}"] = g_mean
            if gcol in ["f_141", "f_109"] and tcol in ["clean_age", "clean_height", "clean_area", "weak_masonry_score"]:
                out[f"{gcol}_diff_{tcol}"] = out[tcol].values - g_mean

    f109_n_f141 = out.groupby("f_109")["f_141"].transform("nunique").values.astype(np.float32)
    out["f109_n_villages"] = f109_n_f141
    out["f141_share_of_f109"] = np.expm1(out["f_141_log_count"]) / np.maximum(np.expm1(out["f_109_log_count"]), 1.0)

    f182_s = out["f_182"].astype(int).astype(str)
    f214_s = out["f_214"].astype(int).astype(str)
    f199_s = out["f_199"].astype(int).astype(str)
    st_bin = np.clip(np.round(out["clean_storey"]).astype(int), 1, 4).astype(str)

    out["inter_141_182"] = out["f_141"].astype(str) + "_" + f182_s
    out["inter_141_214"] = out["f_141"].astype(str) + "_" + f214_s
    out["inter_141_roof"] = out["f_141"].astype(str) + "_" + out["roofing_kind"]
    out["inter_141_f62"] = out["f_141"].astype(str) + "_" + out["f_62"]
    out["inter_141_f209"] = out["f_141"].astype(str) + "_" + out["f_209"]
    out["inter_109_182_roof"] = out["f_109"].astype(str) + "_" + f182_s + "_" + out["roofing_kind"]
    out["inter_109_f62_f69"] = out["f_109"].astype(str) + "_" + out["f_62"] + "_" + out["f_69"]
    out["inter_109_f209_st"] = out["f_109"].astype(str) + "_" + out["f_209"] + "_" + st_bin
    out["inter_struct_full"] = (
        out["roofing_kind"] + "_" + out["f_62"] + "_" + out["f_69"] + "_" +
        out["f_209"] + "_" + f182_s + "_" + f214_s + "_" + f199_s
    )
    out["inter_90_struct"] = out["f_90"].astype(str) + "_" + out["inter_struct_full"]
    out["inter_109_struct"] = out["f_109"].astype(str) + "_" + out["roofing_kind"] + "_" + out["f_62"] + "_" + f182_s

    return out


# ==============================================================================
# 4. HIERARCHICAL TARGET ENCODING, VILLAGE KERNEL KNN & PURE SEISMIC RESIDUALS
# ==============================================================================
HIERARCHIES_CLEAN = [
    ("f_90", None, 10.0, "te_f90"),
    ("f_109", "f_90", 8.0, "te_f109"),
    ("f_141", "f_109", 3.5, "te_f141_m3"),
    ("f_141", "f_109", 10.0, "te_f141_m10"),
    ("f_141", "f_109", 25.0, "te_f141_m25"),
    ("inter_141_182", "f_141", 5.0, "te_i141_182"),
    ("inter_141_214", "f_141", 5.0, "te_i141_214"),
    ("inter_141_roof", "f_141", 6.0, "te_i141_roof"),
    ("inter_141_f62", "f_141", 6.0, "te_i141_f62"),
    ("inter_141_f209", "f_141", 6.0, "te_i141_f209"),
    ("inter_109_182_roof", "f_109", 8.0, "te_i109_182_roof"),
    ("inter_109_f62_f69", "f_109", 8.0, "te_i109_62_69"),
    ("inter_109_f209_st", "f_109", 8.0, "te_i109_209_st"),
    ("inter_struct_full", None, 15.0, "te_struct_full"),
    ("inter_90_struct", "inter_struct_full", 10.0, "te_i90_struct"),
    ("inter_109_struct", "f_109", 8.0, "te_i109_struct"),
]

# Non-redundant TE + KNN columns for Tree models (P1-C / P2-C: removes duplicate monotonic log/raw copies)
GBDT_TE_PREFIXES = (
    "te_f90_bal_", "te_f109_bal_", "te_f141_m3_bal_", "te_f141_m10_bal_", "te_f141_m25_bal_",
    "te_i141_182_bal_", "te_i141_214_bal_", "te_i141_roof_bal_", "te_i109_struct_bal_",
    "te_struct_full_bal_", "knn141_t15_bal_", "knn141_t35_bal_"
)

CLEAN_7_HIERS = (
    "te_f90_p", "te_f109_p", "te_f141_m3_p", "te_f141_m10_p", "te_f141_m25_p",
    "te_i141_182_p", "te_i141_roof_p", "te_i109_182_roof_p", "te_struct_full_p"
)


def fit_apply_te_and_res(
    df: pd.DataFrame,
    Y_oh: np.ndarray,
    y_train: np.ndarray,
    X_bldg_all: np.ndarray,
    fit_idx: np.ndarray,
    apply_idx: np.ndarray,
    scale_m: float = 1.0,
    include_seismic_res: bool = True,
    X_sim: np.ndarray = None,
    v_groups_141: list = None
) -> pd.DataFrame:
    Y_fit = Y_oh[fit_idx]
    p_global = Y_fit.mean(axis=0)
    parent_post = {None: np.tile(p_global, (len(apply_idx), 1))}
    fit_parent_post = {None: np.tile(p_global, (len(fit_idx), 1))}
    out_cols = {}

    for col, parent_col, m_base, feat_name in HIERARCHIES_CLEAN:
        if col not in df.columns:
            continue
        m = m_base * scale_m
        fit_keys = df[col].values[fit_idx]
        app_keys = df[col].values[apply_idx]

        val_prior = parent_post[parent_col]
        fit_prior = fit_parent_post[parent_col]

        c_df = pd.DataFrame({
            "k": fit_keys,
            "y0": Y_fit[:, 0], "y1": Y_fit[:, 1], "y2": Y_fit[:, 2],
            "pr0": fit_prior[:, 0], "pr1": fit_prior[:, 1], "pr2": fit_prior[:, 2]
        })
        c_stats = c_df.groupby("k").agg(
            n=("y0", "count"),
            s0=("y0", "sum"), s1=("y1", "sum"), s2=("y2", "sum"),
            pr0=("pr0", "first"), pr1=("pr1", "first"), pr2=("pr2", "first")
        )
        c_stats["post0"] = (c_stats["s0"] + m * c_stats["pr0"]) / (c_stats["n"] + m)
        c_stats["post1"] = (c_stats["s1"] + m * c_stats["pr1"]) / (c_stats["n"] + m)
        c_stats["post2"] = (c_stats["s2"] + m * c_stats["pr2"]) / (c_stats["n"] + m)

        merged_app = pd.DataFrame({"k": app_keys}).merge(
            c_stats[["post0", "post1", "post2"]], on="k", how="left"
        )
        post_app = merged_app[["post0", "post1", "post2"]].values
        nan_mask = np.isnan(post_app[:, 0])
        if np.any(nan_mask):
            post_app[nan_mask] = val_prior[nan_mask]

        P_bal = calibrate_for_balanced_prior(post_app, p_global)
        for k in range(3):
            out_cols[f"{feat_name}_p{k+1}"] = post_app[:, k].astype(np.float32)
            out_cols[f"{feat_name}_bal_p{k+1}"] = P_bal[:, k].astype(np.float32)
            out_cols[f"{feat_name}_log_p{k+1}"] = np.log(np.clip(P_bal[:, k], 1e-3, 1.0)).astype(np.float32)
        out_cols[f"{feat_name}_bal_ev"] = (1.0 * P_bal[:, 0] + 2.0 * P_bal[:, 1] + 3.0 * P_bal[:, 2]).astype(np.float32)

        if col in ["f_90", "f_109", "f_141", "inter_struct_full"]:
            parent_post[col] = post_app
            merged_fit = pd.DataFrame({"k": fit_keys}).merge(
                c_stats[["post0", "post1", "post2"]], on="k", how="left"
            )
            fit_parent_post[col] = merged_fit[["post0", "post1", "post2"]].values

    # P2-B: Pure-Physical Within-Village Kernel KNN with Hierarchical Meso+Struct Fallback
    if X_sim is not None and v_groups_141 is not None:
        in_fit = np.zeros(len(df), dtype=bool)
        in_fit[fit_idx] = True
        in_app = np.zeros(len(df), dtype=bool)
        in_app[apply_idx] = True
        prior_109_full = np.zeros((len(df), 3), dtype=np.float64)
        p_i109_st = np.column_stack([out_cols["te_i109_struct_p1"], out_cols["te_i109_struct_p2"], out_cols["te_i109_struct_p3"]])
        prior_109_full[apply_idx] = 0.50 * parent_post["f_109"] + 0.50 * p_i109_st
        out_k1 = np.zeros((len(df), 3), dtype=np.float32)
        out_k2 = np.zeros((len(df), 3), dtype=np.float32)
        m_k1 = 2.5 * scale_m
        m_k2 = 4.5 * scale_m

        for idxs in v_groups_141:
            a_idxs = idxs[in_app[idxs]]
            if len(a_idxs) == 0:
                continue
            f_idxs = idxs[in_fit[idxs]]
            if len(f_idxs) == 0:
                out_k1[a_idxs] = prior_109_full[a_idxs]
                out_k2[a_idxs] = prior_109_full[a_idxs]
                continue
            Xa = X_sim[a_idxs]
            Xf = X_sim[f_idxs]
            Yf = Y_oh[f_idxs]
            sq = np.sum((Xa[:, None, :] - Xf[None, :, :]) ** 2, axis=2)
            W1 = np.exp(-sq / (2.0 * (1.5 ** 2)))
            W2 = np.exp(-sq / (2.0 * (3.5 ** 2)))
            pr = prior_109_full[a_idxs]
            out_k1[a_idxs] = ((W1 @ Yf + m_k1 * pr) / (W1.sum(axis=1, keepdims=True) + m_k1)).astype(np.float32)
            out_k2[a_idxs] = ((W2 @ Yf + m_k2 * pr) / (W2.sum(axis=1, keepdims=True) + m_k2)).astype(np.float32)

        k1_bal = calibrate_for_balanced_prior(out_k1[apply_idx], p_global)
        k2_bal = calibrate_for_balanced_prior(out_k2[apply_idx], p_global)
        for k in range(3):
            out_cols[f"knn141_t15_bal_p{k+1}"] = k1_bal[:, k].astype(np.float32)
            out_cols[f"knn141_t35_bal_p{k+1}"] = k2_bal[:, k].astype(np.float32)

    if include_seismic_res:
        Xb_fit, Xb_app = build_spline_matrix(X_bldg_all[fit_idx], X_bldg_all[apply_idx])
        P_both = fit_predict_bohning_softmax(Xb_fit, y_train[fit_idx], np.vstack([Xb_fit, Xb_app]), C=0.05, max_iter=40)
        P_bldg_fit = P_both[:len(fit_idx)]
        P_bldg_app = P_both[len(fit_idx):]
        R_fit = Y_fit - P_bldg_fit

        r_df = pd.DataFrame({
            "f109": df["f_109"].values[fit_idx],
            "f141": df["f_141"].values[fit_idx],
            "r0": R_fit[:, 0], "r1": R_fit[:, 1], "r2": R_fit[:, 2]
        })
        m109 = 10.0 * scale_m
        m141 = 6.0 * scale_m
        g109 = r_df.groupby("f109").agg(n=("r0", "count"), s0=("r0", "sum"), s1=("r1", "sum"), s2=("r2", "sum"))
        for k in range(3):
            g109[f"m{k}"] = g109[f"s{k}"] / (g109["n"] + m109)
        r_df = r_df.merge(g109[["m0", "m1", "m2"]], on="f109", how="left")
        g141 = r_df.groupby("f141").agg(
            n=("r0", "count"), s0=("r0", "sum"), s1=("r1", "sum"), s2=("r2", "sum"),
            m0=("m0", "first"), m1=("m1", "first"), m2=("m2", "first")
        )
        for k in range(3):
            g141[f"res141_{k}"] = (g141[f"s{k}"] + m141 * g141[f"m{k}"]) / (g141["n"] + m141)

        app_df = pd.DataFrame({"f109": df["f_109"].values[apply_idx], "f141": df["f_141"].values[apply_idx]})
        app_df = app_df.merge(g109[["m0", "m1", "m2"]], on="f109", how="left").fillna(0.0)
        app_df = app_df.merge(g141[["res141_0", "res141_1", "res141_2"]], on="f141", how="left")
        for k in range(3):
            app_df[f"res141_{k}"] = app_df[f"res141_{k}"].fillna(app_df[f"m{k}"])

        r141_app = app_df[["res141_0", "res141_1", "res141_2"]].values
        p_deb = np.clip(P_bldg_app + r141_app, 1e-4, 1.0)
        p_deb /= p_deb.sum(axis=1, keepdims=True)
        p_deb_bal = calibrate_for_balanced_prior(p_deb, p_global)
        for k in range(3):
            out_cols[f"seismic_res_p{k+1}"] = r141_app[:, k].astype(np.float32)
            out_cols[f"debiased_p{k+1}"] = p_deb_bal[:, k].astype(np.float32)

    return pd.DataFrame(out_cols, index=apply_idx)


def build_nested_fold_te(
    df: pd.DataFrame,
    Y_oh: np.ndarray,
    y_train: np.ndarray,
    X_bldg_all: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    te_idx: np.ndarray,
    n_inner: int = 5,
    seed: int = 42,
    include_seismic_res: bool = True,
    X_sim: np.ndarray = None,
    v_groups_141: list = None
):
    app_idx = np.concatenate([va_idx, te_idx])
    te_va_te = fit_apply_te_and_res(
        df, Y_oh, y_train, X_bldg_all, tr_idx, app_idx,
        scale_m=1.0, include_seismic_res=include_seismic_res,
        X_sim=X_sim, v_groups_141=v_groups_141
    )
    te_va = te_va_te.iloc[:len(va_idx)]
    te_te = te_va_te.iloc[len(va_idx):]

    inner_f = get_stratified_kfold_indices(y_train[tr_idx], n_splits=n_inner, seed=seed)
    te_tr_parts = []
    for inf in range(n_inner):
        in_fit = tr_idx[inner_f != inf]
        in_val = tr_idx[inner_f == inf]
        te_tr_parts.append(
            fit_apply_te_and_res(
                df, Y_oh, y_train, X_bldg_all, in_fit, in_val,
                scale_m=(n_inner - 1.0) / n_inner, include_seismic_res=include_seismic_res,
                X_sim=X_sim, v_groups_141=v_groups_141
            )
        )
    te_tr = pd.concat(te_tr_parts, axis=0).loc[tr_idx]
    return te_tr, te_va, te_te


# ==============================================================================
# 5. MODELS: SPLINE SOFTMAX/LOGIT, RESIDUAL MLP & SPARSE HIST-GBDT
# ==============================================================================
def fit_predict_bohning_softmax(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    C: float = 0.04,
    max_iter: int = 65
) -> np.ndarray:
    n, d = X_tr.shape
    Y = np.zeros((n, 3), dtype=np.float64)
    for k in range(3):
        Y[:, k] = (y_tr == (k + 1)).astype(np.float64)
    reg = 1.0 / (C * n)

    def loss_grad(w_flat):
        W = w_flat.reshape(d, 3)
        logits = X_tr @ W
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        P = exp_l / exp_l.sum(axis=1, keepdims=True)
        loss = -np.sum(Y * np.log(np.clip(P, 1e-12, 1.0))) / n + 0.5 * reg * np.sum(W[:-1] ** 2)
        grad = (X_tr.T @ (P - Y)) / n
        grad[:-1] += reg * W[:-1]
        return loss, grad.ravel()

    res = minimize(loss_grad, np.zeros(d * 3), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    W = res.x.reshape(d, 3)
    logits_va = X_va @ W
    logits_va -= logits_va.max(axis=1, keepdims=True)
    exp_va = np.exp(logits_va)
    return exp_va / exp_va.sum(axis=1, keepdims=True)


def fit_predict_binary_logit(
    X_tr: np.ndarray,
    y_bin: np.ndarray,
    X_va: np.ndarray,
    C: float = 0.04,
    max_iter: int = 55
) -> np.ndarray:
    n, d = X_tr.shape
    reg = 1.0 / (C * n)

    def loss_grad(w):
        z = np.clip(X_tr @ w, -20.0, 20.0)
        p = 1.0 / (1.0 + np.exp(-z))
        loss = -np.mean(y_bin * np.log(np.clip(p, 1e-12, 1.0)) + (1.0 - y_bin) * np.log(np.clip(1.0 - p, 1e-12, 1.0)))
        loss += 0.5 * reg * np.sum(w[:-1] ** 2)
        grad = (X_tr.T @ (p - y_bin)) / n
        grad[:-1] += reg * w[:-1]
        return loss, grad

    res = minimize(loss_grad, np.zeros(d), jac=True, method="L-BFGS-B", options={"maxiter": max_iter})
    z_va = np.clip(X_va @ res.x, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z_va))


def build_spline_matrix(X_tr_raw: np.ndarray, X_va_raw: np.ndarray):
    med = np.nanmedian(X_tr_raw, axis=0)
    X_tr = np.where(np.isnan(X_tr_raw), med, X_tr_raw)
    X_va = np.where(np.isnan(X_va_raw), med, X_va_raw)
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0) + 1e-6
    Z_tr = np.clip((X_tr - mean) / std, -4.5, 4.5)
    Z_va = np.clip((X_va - mean) / std, -4.5, 4.5)

    n_Check = min(40, Z_tr.shape[1])
    knots = [-1.0, 0.0, 1.0]
    h_tr = [Z_tr] + [np.maximum(0.0, Z_tr[:, :n_Check] - k) for k in knots] + [np.maximum(0.0, -Z_tr[:, :n_Check] - 0.5)]
    h_va = [Z_va] + [np.maximum(0.0, Z_va[:, :n_Check] - k) for k in knots] + [np.maximum(0.0, -Z_va[:, :n_Check] - 0.5)]
    h_tr.append(np.ones((len(Z_tr), 1)))
    h_va.append(np.ones((len(Z_va), 1)))
    return np.hstack(h_tr), np.hstack(h_va)


def fit_predict_res_mlp(X_tr_raw, y_tr, X_eval_raw, seed=42, epochs=28, batch_size=256, lr=2.5e-3, wd=3e-4):
    rng = np.random.RandomState(seed)
    mu = np.nanmean(X_tr_raw, axis=0)
    sd = np.nanstd(X_tr_raw, axis=0) + 1e-5
    X_tr = np.clip((np.nan_to_num(X_tr_raw, 0.0) - mu) / sd, -5.0, 5.0).astype(np.float32)
    X_ev = np.clip((np.nan_to_num(X_eval_raw, 0.0) - mu) / sd, -5.0, 5.0).astype(np.float32)
    n, d = X_tr.shape
    h = 96
    W1 = (rng.randn(d, h) * np.sqrt(2.0 / d)).astype(np.float32)
    b1 = np.zeros(h, dtype=np.float32)
    W2 = (rng.randn(h, h) * np.sqrt(2.0 / h)).astype(np.float32)
    b2 = np.zeros(h, dtype=np.float32)
    W3 = (rng.randn(h, 3) * 0.05).astype(np.float32)
    b3 = np.zeros(3, dtype=np.float32)

    Y_oh = np.zeros((n, 3), dtype=np.float32)
    for k in range(3):
        Y_oh[:, k] = (y_tr == (k + 1)).astype(np.float32)

    params = [W1, b1, W2, b2, W3, b3]
    m_adam = [np.zeros_like(p) for p in params]
    v_adam = [np.zeros_like(p) for p in params]
    t_step = 0

    for ep in range(epochs):
        cur_lr = lr * (0.5 * (1.0 + np.cos(np.pi * ep / epochs)))
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_tr[idx]
            yb = Y_oh[idx]
            bs = len(idx)

            z1 = xb @ W1 + b1
            sig1 = 1.0 / (1.0 + np.exp(-np.clip(z1, -15, 15)))
            a1 = z1 * sig1
            z2 = a1 @ W2 + b2
            sig2 = 1.0 / (1.0 + np.exp(-np.clip(z2, -15, 15)))
            a2 = a1 + z2 * sig2
            logits = a2 @ W3 + b3
            logits -= logits.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            probs = exp_l / exp_l.sum(axis=1, keepdims=True)

            dlogits = (probs - yb) / bs
            dW3 = a2.T @ dlogits + wd * W3
            db3 = dlogits.sum(axis=0)
            da2 = dlogits @ W3.T
            dz2 = da2 * (sig2 * (1.0 + z2 * (1.0 - sig2)))
            dW2 = a1.T @ dz2 + wd * W2
            db2 = dz2.sum(axis=0)
            da1 = da2 + dz2 @ W2.T
            dz1 = da1 * (sig1 * (1.0 + z1 * (1.0 - sig1)))
            dW1 = xb.T @ dz1 + wd * W1
            db1 = dz1.sum(axis=0)

            grads = [dW1, db1, dW2, db2, dW3, db3]
            t_step += 1
            for pi in range(6):
                m_adam[pi] = 0.9 * m_adam[pi] + 0.1 * grads[pi]
                v_adam[pi] = 0.999 * v_adam[pi] + 0.001 * (grads[pi] ** 2)
                m_hat = m_adam[pi] / (1.0 - 0.9 ** t_step)
                v_hat = v_adam[pi] / (1.0 - 0.999 ** t_step)
                params[pi] -= cur_lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    z1 = X_ev @ W1 + b1
    a1 = z1 / (1.0 + np.exp(-np.clip(z1, -15, 15)))
    z2 = a1 @ W2 + b2
    a2 = a1 + z2 / (1.0 + np.exp(-np.clip(z2, -15, 15)))
    logits = a2 @ W3 + b3
    logits -= logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    return (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float64)


class SparseHistGBDT:
    def __init__(
        self,
        n_estimators: int = 240,
        learning_rate: float = 0.045,
        max_depth: int = 5,
        max_leaves: int = 24,
        min_child_weight: float = 20.0,
        reg_lambda: float = 6.0,
        colsample_bytree: float = 0.65,
        subsample: float = 0.80,
        n_bins: int = 36,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.colsample = colsample_bytree
        self.subsample = subsample
        self.n_bins = n_bins
        self.random_state = random_state

    def _bin_data(self, X_tr: np.ndarray, X_others: list):
        n_features = X_tr.shape[1]
        B_tr = np.zeros(X_tr.shape, dtype=np.int32)
        B_others = [np.zeros(X.shape, dtype=np.int32) for X in X_others]
        qs = np.linspace(0.0, 1.0, self.n_bins - 1)[1:-1]
        for j in range(n_features):
            col_tr = X_tr[:, j]
            valid = ~np.isnan(col_tr)
            if not np.any(valid):
                continue
            edges = np.unique(np.quantile(col_tr[valid], qs))
            b_tr = np.zeros(len(col_tr), dtype=np.int32)
            b_tr[valid] = np.searchsorted(edges, col_tr[valid]) + 1
            B_tr[:, j] = b_tr
            for idx_o, X_o in enumerate(X_others):
                col_o = X_o[:, j]
                v_o = ~np.isnan(col_o)
                b_o = np.zeros(len(col_o), dtype=np.int32)
                b_o[v_o] = np.searchsorted(edges, col_o[v_o]) + 1
                B_others[idx_o][:, j] = b_o
        return B_tr, B_others

    def _grow_tree(self, B_tr: np.ndarray, g: np.ndarray, h: np.ndarray, feat_idx: np.ndarray, sample_mask: np.ndarray):
        n_samples = B_tr.shape[0]
        node_ids = np.full(n_samples, -1, dtype=np.int32)
        node_ids[sample_mask] = 0

        tree_nodes = []
        active_leaves = [(0, 0)]
        next_node_id = 1
        nb = self.n_bins + 1

        while active_leaves and len(active_leaves) + len(tree_nodes) < 2 * self.max_leaves:
            nid, depth = active_leaves.pop(0)
            in_node = (node_ids == nid)
            if depth >= self.max_depth or np.sum(in_node) < 20:
                G_tot = float(np.sum(g[in_node]))
                H_tot = float(np.sum(h[in_node]))
                val = -G_tot / (H_tot + self.reg_lambda)
                tree_nodes.append({"id": nid, "is_leaf": True, "val": val})
                continue

            B_sub = B_tr[in_node][:, feat_idx]
            g_sub = g[in_node]
            h_sub = h[in_node]
            G_tot = float(np.sum(g_sub))
            H_tot = float(np.sum(h_sub))

            if H_tot < 2.0 * self.min_child_weight:
                val = -G_tot / (H_tot + self.reg_lambda)
                tree_nodes.append({"id": nid, "is_leaf": True, "val": val})
                continue

            n_f = len(feat_idx)
            offsets = (np.arange(n_f, dtype=np.int32) * nb)[None, :]
            flat_idx = (B_sub + offsets).ravel()
            g_rep = np.repeat(g_sub, n_f)
            h_rep = np.repeat(h_sub, n_f)

            G_hist = np.bincount(flat_idx, weights=g_rep, minlength=n_f * nb).reshape(n_f, nb)
            H_hist = np.bincount(flat_idx, weights=h_rep, minlength=n_f * nb).reshape(n_f, nb)

            G_nan = G_hist[:, 0:1]
            H_nan = H_hist[:, 0:1]
            G_val = G_hist[:, 1:]
            H_val = H_hist[:, 1:]

            G_L_no = np.cumsum(G_val, axis=1)[:, :-1]
            H_L_no = np.cumsum(H_val, axis=1)[:, :-1]

            parent_score = (G_tot ** 2) / (H_tot + self.reg_lambda)
            best_gain = 1e-4
            best_split = None

            for nan_left in (True, False):
                if nan_left:
                    G_L = G_L_no + G_nan
                    H_L = H_L_no + H_nan
                else:
                    G_L = G_L_no
                    H_L = H_L_no
                G_R = G_tot - G_L
                H_R = H_tot - H_L

                valid_sp = (H_L >= self.min_child_weight) & (H_R >= self.min_child_weight)
                if not np.any(valid_sp):
                    continue
                gains = np.where(
                    valid_sp,
                    0.5 * ((G_L ** 2) / (H_L + self.reg_lambda) + (G_R ** 2) / (H_R + self.reg_lambda) - parent_score),
                    -1.0
                )
                idx_max = int(np.argmax(gains))
                f_local, b_thresh_m1 = divmod(idx_max, gains.shape[1])
                if gains[f_local, b_thresh_m1] > best_gain:
                    best_gain = float(gains[f_local, b_thresh_m1])
                    best_split = (int(feat_idx[f_local]), int(b_thresh_m1 + 1), nan_left)

            if best_split is None:
                val = -G_tot / (H_tot + self.reg_lambda)
                tree_nodes.append({"id": nid, "is_leaf": True, "val": val})
                continue

            f_best, thresh_bin, nan_left = best_split
            left_id = next_node_id
            right_id = next_node_id + 1
            next_node_id += 2

            tree_nodes.append({
                "id": nid, "is_leaf": False,
                "f": f_best, "t": thresh_bin, "nan_left": nan_left,
                "left": left_id, "right": right_id
            })

            col_b = B_tr[:, f_best]
            if nan_left:
                go_left = in_node & (col_b <= thresh_bin)
            else:
                go_left = in_node & (col_b > 0) & (col_b <= thresh_bin)
            go_right = in_node & (~go_left)
            node_ids[go_left] = left_id
            node_ids[go_right] = right_id

            active_leaves.append((left_id, depth + 1))
            active_leaves.append((right_id, depth + 1))

        for nid, _ in active_leaves:
            in_node = (node_ids == nid)
            G_tot = float(np.sum(g[in_node]))
            H_tot = float(np.sum(h[in_node]))
            val = -G_tot / (H_tot + self.reg_lambda)
            tree_nodes.append({"id": nid, "is_leaf": True, "val": val})

        node_map = {nd["id"]: nd for nd in tree_nodes}
        return node_map

    def _predict_tree(self, node_map: dict, B: np.ndarray) -> np.ndarray:
        n = B.shape[0]
        out = np.zeros(n, dtype=np.float64)
        node_ids = np.zeros(n, dtype=np.int32)
        for nid in sorted(node_map.keys()):
            nd = node_map[nid]
            mask = (node_ids == nid)
            if not np.any(mask):
                continue
            if nd["is_leaf"]:
                out[mask] = nd["val"]
            else:
                col_b = B[mask, nd["f"]]
                if nd["nan_left"]:
                    left_m = (col_b <= nd["t"])
                else:
                    left_m = (col_b > 0) & (col_b <= nd["t"])
                idx_m = np.where(mask)[0]
                node_ids[idx_m[left_m]] = nd["left"]
                node_ids[idx_m[~left_m]] = nd["right"]
        return out

    def fit_predict(self, X_tr: np.ndarray, y_tr_zero: np.ndarray, X_va: np.ndarray, X_te: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        B_tr, (B_va, B_te) = self._bin_data(X_tr, [X_va, X_te])
        n_tr, n_f = B_tr.shape

        Y = np.zeros((n_tr, 3), dtype=np.float64)
        for k in range(3):
            Y[:, k] = (y_tr_zero == k).astype(np.float64)

        priors = np.clip(Y.mean(axis=0), 1e-4, 1.0)
        init_logits = np.log(priors)
        F_tr = np.tile(init_logits, (n_tr, 1))
        F_va = np.tile(init_logits, (len(X_va), 1))
        F_te = np.tile(init_logits, (len(X_te), 1))

        for m in range(self.n_estimators):
            logits = F_tr - F_tr.max(axis=1, keepdims=True)
            exp_l = np.exp(logits)
            P = exp_l / exp_l.sum(axis=1, keepdims=True)

            sample_mask = rng.rand(n_tr) < self.subsample
            n_col = max(1, int(n_f * self.colsample))

            for k in range(3):
                g = P[:, k] - Y[:, k]
                h = np.maximum(P[:, k] * (1.0 - P[:, k]), 1e-4)
                feat_idx = rng.choice(n_f, size=n_col, replace=False)
                tree = self._grow_tree(B_tr, g, h, feat_idx, sample_mask)
                F_tr[:, k] += self.lr * self._predict_tree(tree, B_tr)
                F_va[:, k] += self.lr * self._predict_tree(tree, B_va)
                F_te[:, k] += self.lr * self._predict_tree(tree, B_te)

        l_va = F_va - F_va.max(axis=1, keepdims=True)
        p_va = np.exp(l_va) / np.exp(l_va).sum(axis=1, keepdims=True)
        l_te = F_te - F_te.max(axis=1, keepdims=True)
        p_te = np.exp(l_te) / np.exp(l_te).sum(axis=1, keepdims=True)
        return p_va, p_te

    def fit_predict_binary(self, X_tr: np.ndarray, y_bin: np.ndarray, X_va: np.ndarray, X_te: np.ndarray):
        rng = np.random.RandomState(self.random_state)
        B_tr, (B_va, B_te) = self._bin_data(X_tr, [X_va, X_te])
        n_tr, n_f = B_tr.shape

        p0 = float(np.clip(np.mean(y_bin), 1e-4, 1.0 - 1e-4))
        init_f = np.log(p0 / (1.0 - p0))
        F_tr = np.full(n_tr, init_f, dtype=np.float64)
        F_va = np.full(len(X_va), init_f, dtype=np.float64)
        F_te = np.full(len(X_te), init_f, dtype=np.float64)

        for m in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-np.clip(F_tr, -20.0, 20.0)))
            g = p - y_bin
            h = np.maximum(p * (1.0 - p), 1e-4)
            sample_mask = rng.rand(n_tr) < self.subsample
            n_col = max(1, int(n_f * self.colsample))
            feat_idx = rng.choice(n_f, size=n_col, replace=False)
            tree = self._grow_tree(B_tr, g, h, feat_idx, sample_mask)
            F_tr += self.lr * self._predict_tree(tree, B_tr)
            F_va += self.lr * self._predict_tree(tree, B_va)
            F_te += self.lr * self._predict_tree(tree, B_te)

        p_va = 1.0 / (1.0 + np.exp(-np.clip(F_va, -20.0, 20.0)))
        p_te = 1.0 / (1.0 + np.exp(-np.clip(F_te, -20.0, 20.0)))
        return p_va, p_te


# ==============================================================================
# 6. PRIOR CALIBRATION & EXACT 1700:1700:1700 MARGINAL ASSIGNMENT
# ==============================================================================
def calibrate_for_balanced_prior(
    probs: np.ndarray,
    train_priors: np.ndarray,
    multipliers: tuple = (1.0, 1.0, 1.0),
    temperature: float = 1.0
) -> np.ndarray:
    p = np.clip(probs, 1e-6, 1.0)
    if temperature != 1.0:
        p = p ** (1.0 / temperature)
    weights = (1.0 / train_priors) * np.asarray(multipliers, dtype=np.float64)
    scaled = p * weights[None, :]
    return scaled / scaled.sum(axis=1, keepdims=True)


def enforce_exact_cardinality(probs: np.ndarray, target_counts=(1700, 1700, 1700)) -> np.ndarray:
    n = len(probs)
    log_p = np.log(np.clip(probs, 1e-8, 1.0))
    tau = np.zeros(3, dtype=np.float64)
    target = np.array(target_counts, dtype=np.float64)
    for _ in range(300):
        adj = log_p + tau[None, :]
        preds = np.argmax(adj, axis=1)
        counts = np.array([np.sum(preds == k) for k in range(3)], dtype=np.float64)
        diff = target - counts
        if np.max(np.abs(diff)) <= 2:
            break
        tau += 0.002 * (diff / (n / 3.0))

    adj = log_p + tau[None, :]
    preds = np.argmax(adj, axis=1)
    for k_over in range(3):
        while np.sum(preds == k_over) > target_counts[k_over]:
            for k_under in range(3):
                if np.sum(preds == k_under) < target_counts[k_under]:
                    cand = np.where(preds == k_over)[0]
                    margins = adj[cand, k_over] - adj[cand, k_under]
                    best_i = cand[int(np.argmin(margins))]
                    preds[best_i] = k_under
                    break
    return preds + 1


# ==============================================================================
# 7. MAIN V6 DUAL-PLAN EXECUTION PIPELINE (PLAN A + PLAN B)
# ==============================================================================
def main():
    t0 = time.time()
    print("=" * 84)
    print("2026 ML&DL Midterm — V6 Dual-Plan Super-Ensemble Pipeline (Target: 0.680+)")
    print("=" * 84)

    train_path = "train.csv" if os.path.exists("train.csv") else "user_uploaded_files/ttl=63d/user_uploaded/train.csv"
    test_path = "test.csv" if os.path.exists("test.csv") else "user_uploaded_files/ttl=63d/user_uploaded/test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y_train = train_df["label"].values.astype(np.int32)
    n_train = len(train_df)
    n_test = len(test_df)
    train_priors = np.array([np.mean(y_train == k) for k in [1, 2, 3]], dtype=np.float64)

    print(f"[1/5] Loaded train={train_df.shape}, test={test_df.shape}")
    base_df = clean_and_engineer_base_features(train_df, test_df, allow_raw_village_sr=False)

    exclude_cols = set(REAL_CATEGORICAL_COLS + [c for c in base_df.columns if c.startswith("inter_")])
    all_base_num_cols = [c for c in base_df.columns if c not in exclude_cols]

    # P1-C Feature Ablation sets: F2 (removes noisy regional diffs) & F1+F2 (also removes sparse OHE)
    no_diff_spline_cols = [c for c in all_base_num_cols if "_diff_clean_" not in c and "_diff_weak_" not in c]
    pruned_spline_base_cols = [
        c for c in all_base_num_cols
        if (not c.startswith("ohe_")) and ("_diff_clean_" not in c and "_diff_weak_" not in c)
    ]
    compact_gb_cols = [c for c in COMPACT_GBDT_BASE_COLS if c in base_df.columns]

    pure_bldg_cols = [
        c for c in all_base_num_cols
        if not c.startswith(("f_141", "f141", "f_109", "f109", "f_90")) and c not in ["f_90", "f_109", "f_141"]
    ]
    X_bldg_all = base_df[pure_bldg_cols].values.astype(np.float64)

    # P2-B Physical Similarity Matrix for Within-Village Kernel KNN
    sim_cols = [
        "clean_age", "clean_storey", "log_area", "clean_height",
        "f_182", "f_214", "f_199", "f_124", "f_103", "f_113",
        "weak_masonry_score", "slenderness",
        "ohe_roofing_kind_Reinforced concrete roof", "ohe_roofing_kind_Lightweight wooden roof",
        "ohe_f_62_c3", "ohe_f_62_c1", "ohe_f_69_c3", "ohe_f_209_c2", "ohe_f_13_c7"
    ]
    sim_cols = [c for c in sim_cols if c in base_df.columns]
    X_sim = base_df[sim_cols].values.astype(np.float64)
    X_sim = (X_sim - X_sim[:n_train].mean(axis=0)) / (X_sim[:n_train].std(axis=0) + 1e-6)
    w_sim = np.ones(len(sim_cols))
    for idx, c in enumerate(sim_cols):
        if c in ["f_182", "f_214", "f_199", "weak_masonry_score", "ohe_roofing_kind_Reinforced concrete roof"]:
            w_sim[idx] = 1.8
    X_sim = X_sim * w_sim[None, :]
    v_groups_141 = [grp["idx"].values for _, grp in pd.DataFrame({"g": base_df["f_141"].values, "idx": np.arange(len(base_df))}).groupby("g")]

    Y_oh = np.zeros((n_train, 3), dtype=np.float64)
    for k in range(3):
        Y_oh[:, k] = (y_train == (k + 1)).astype(np.float64)
    te_idx = np.arange(n_train, n_train + n_test)

    # [2/5] Precompute 10-Fold Cross-Fitted Training TE (90% labels per fold) + Scale-Aligned 90%/100% Test TE for Plan B
    print("[2/5] Computing Plan B 10-Fold Cross-Fitted Training TE & Scale-Aligned 90%/100% Test TE...")
    folds10 = get_stratified_kfold_indices(y_train, n_splits=10, seed=42)
    parts_tr10 = []
    te_test_avg90 = None
    for f in range(10):
        fit_i = np.where(folds10 != f)[0]
        val_i = np.where(folds10 == f)[0]
        res_f = fit_apply_te_and_res(
            base_df, Y_oh, y_train, X_bldg_all, fit_i, np.concatenate([val_i, te_idx]),
            scale_m=0.90, include_seismic_res=True, X_sim=X_sim, v_groups_141=v_groups_141
        )
        parts_tr10.append(res_f.iloc[:len(val_i)])
        t_part = res_f.iloc[len(val_i):].values
        te_test_avg90 = t_part / 10.0 if te_test_avg90 is None else te_test_avg90 + t_part / 10.0

    te_tr_cf10_df = pd.concat(parts_tr10, axis=0).loc[np.arange(n_train)]
    te_tr_cf10 = te_tr_cf10_df.values
    te_cols = list(te_tr_cf10_df.columns)
    te_test_full100 = fit_apply_te_and_res(
        base_df, Y_oh, y_train, X_bldg_all, np.arange(n_train), te_idx,
        scale_m=1.0, include_seismic_res=True, X_sim=X_sim, v_groups_141=v_groups_141
    ).values
    te_test_aligned = 0.50 * te_test_avg90 + 0.50 * te_test_full100

    gb_knn_idx = [i for i, c in enumerate(te_cols) if c.startswith(GBDT_TE_PREFIXES)]
    clean_gb_idx = [i for i, c in enumerate(te_cols) if c.startswith(CLEAN_7_HIERS)]

    # [3/5] Plan A: 5-Fold Double-Nested Zero-Leakage OOF & Fold-Bagged Test Predictions
    print("[3/5] Training Plan A (5-Fold Double-Nested OOF & Test Ensemble)...")
    folds5 = get_stratified_kfold_indices(y_train, n_splits=5, seed=42)
    oof_sp_f2 = np.zeros((n_train, 3))
    test_sp_f2_A = np.zeros((n_test, 3))
    oof_ord_f2 = np.zeros((n_train, 3))
    test_ord_f2_A = np.zeros((n_test, 3))
    oof_ord_f1f2 = np.zeros((n_train, 3))
    test_ord_f1f2_A = np.zeros((n_test, 3))
    oof_mlp = np.zeros((n_train, 3))
    test_mlp_A = np.zeros((n_test, 3))

    oof_gb_ord = np.zeros((n_train, 3))
    test_gb_ord_A = np.zeros((n_test, 3))
    oof_gb_d5 = np.zeros((n_train, 3))
    test_gb_d5_A = np.zeros((n_test, 3))
    oof_gb_d6 = np.zeros((n_train, 3))
    test_gb_d6_A = np.zeros((n_test, 3))
    oof_gb_d4 = np.zeros((n_train, 3))
    test_gb_d4_A = np.zeros((n_test, 3))

    X_te_f2_100 = np.hstack([base_df[no_diff_spline_cols].values[te_idx], te_test_full100]).astype(np.float64)
    X_te_f1f2_100 = np.hstack([base_df[pruned_spline_base_cols].values[te_idx], te_test_full100]).astype(np.float64)
    X_te_gb_knn_100 = np.hstack([base_df[compact_gb_cols].values[te_idx], te_test_full100[:, gb_knn_idx]]).astype(np.float64)
    X_te_gb_cln_100 = np.hstack([base_df[compact_gb_cols].values[te_idx], te_test_full100[:, clean_gb_idx]]).astype(np.float64)

    for f in range(5):
        tr_idx = np.where(folds5 != f)[0]
        va_idx = np.where(folds5 == f)[0]
        y_tr_f = y_train[tr_idx]
        n_va = len(va_idx)

        te_tr_df, te_va_df, te_te_df = build_nested_fold_te(
            base_df, Y_oh, y_train, X_bldg_all, tr_idx, va_idx, te_idx,
            n_inner=5, seed=42 + f, include_seismic_res=True,
            X_sim=X_sim, v_groups_141=v_groups_141
        )
        te_tr = te_tr_df.values
        te_va = te_va_df.values
        te_te_fold = te_te_df.values

        # F2 SplineMulti & SplineOrd
        X_tr_s = np.hstack([base_df[no_diff_spline_cols].values[tr_idx], te_tr]).astype(np.float64)
        X_va_s = np.hstack([base_df[no_diff_spline_cols].values[va_idx], te_va]).astype(np.float64)
        X_te_s_fold = np.hstack([base_df[no_diff_spline_cols].values[te_idx], te_te_fold]).astype(np.float64)
        Xs_tr, Xs_vt = build_spline_matrix(X_tr_s, np.vstack([X_va_s, X_te_s_fold, X_te_f2_100]))

        p_sm = fit_predict_bohning_softmax(Xs_tr, y_tr_f, Xs_vt, C=0.04, max_iter=60)
        oof_sp_f2[va_idx] = p_sm[:n_va]
        test_sp_f2_A += (0.45 * p_sm[n_va:n_va+n_test] + 0.55 * p_sm[n_va+n_test:]) / 5.0

        m23 = (y_tr_f >= 2)
        m12 = (y_tr_f <= 2)
        p_ge2 = fit_predict_binary_logit(Xs_tr, m23.astype(np.float64), Xs_vt, C=0.04, max_iter=55)
        p_3g2 = fit_predict_binary_logit(Xs_tr[m23], (y_tr_f[m23] == 3).astype(np.float64), Xs_vt, C=0.04, max_iter=55)
        p_le2 = fit_predict_binary_logit(Xs_tr, m12.astype(np.float64), Xs_vt, C=0.04, max_iter=55)
        p_1g2 = fit_predict_binary_logit(Xs_tr[m12], (y_tr_f[m12] == 1).astype(np.float64), Xs_vt, C=0.04, max_iter=55)
        p_o = 0.5 * np.column_stack([1 - p_ge2, p_ge2 * (1 - p_3g2), p_ge2 * p_3g2]) + 0.5 * np.column_stack([p_le2 * p_1g2, p_le2 * (1 - p_1g2), 1 - p_le2])
        oof_ord_f2[va_idx] = p_o[:n_va]
        test_ord_f2_A += (0.45 * p_o[n_va:n_va+n_test] + 0.55 * p_o[n_va+n_test:]) / 5.0

        # F1+F2 SplineOrd & Residual MLP
        X_tr_p = np.hstack([base_df[pruned_spline_base_cols].values[tr_idx], te_tr]).astype(np.float64)
        X_va_p = np.hstack([base_df[pruned_spline_base_cols].values[va_idx], te_va]).astype(np.float64)
        X_te_p_fold = np.hstack([base_df[pruned_spline_base_cols].values[te_idx], te_te_fold]).astype(np.float64)
        Xp_tr, Xp_vt = build_spline_matrix(X_tr_p, np.vstack([X_va_p, X_te_p_fold, X_te_f1f2_100]))
        p_ge2 = fit_predict_binary_logit(Xp_tr, m23.astype(np.float64), Xp_vt, C=0.04, max_iter=55)
        p_3g2 = fit_predict_binary_logit(Xp_tr[m23], (y_tr_f[m23] == 3).astype(np.float64), Xp_vt, C=0.04, max_iter=55)
        p_le2 = fit_predict_binary_logit(Xp_tr, m12.astype(np.float64), Xp_vt, C=0.04, max_iter=55)
        p_1g2 = fit_predict_binary_logit(Xp_tr[m12], (y_tr_f[m12] == 1).astype(np.float64), Xp_vt, C=0.04, max_iter=55)
        p_op = 0.5 * np.column_stack([1 - p_ge2, p_ge2 * (1 - p_3g2), p_ge2 * p_3g2]) + 0.5 * np.column_stack([p_le2 * p_1g2, p_le2 * (1 - p_1g2), 1 - p_le2])
        oof_ord_f1f2[va_idx] = p_op[:n_va]
        test_ord_f1f2_A += (0.45 * p_op[n_va:n_va+n_test] + 0.55 * p_op[n_va+n_test:]) / 5.0

        pm = fit_predict_res_mlp(X_tr_p, y_tr_f, np.vstack([X_va_p, X_te_p_fold, X_te_f1f2_100]), seed=42 + f)
        oof_mlp[va_idx] = pm[:n_va]
        test_mlp_A += (0.45 * pm[n_va:n_va+n_test] + 0.55 * pm[n_va+n_test:]) / 5.0

        # Compact Ordinal Specialist GBDT (with Village Kernel KNN)
        X_tr_gk = np.hstack([base_df[compact_gb_cols].values[tr_idx], te_tr[:, gb_knn_idx]]).astype(np.float64)
        X_va_gk = np.hstack([base_df[compact_gb_cols].values[va_idx], te_va[:, gb_knn_idx]]).astype(np.float64)
        X_te_gk_fold = np.hstack([base_df[compact_gb_cols].values[te_idx], te_te_fold[:, gb_knn_idx]]).astype(np.float64)
        X_te_gk_both = np.vstack([X_te_gk_fold, X_te_gb_knn_100])

        gb_ge2 = SparseHistGBDT(n_estimators=210, learning_rate=0.045, max_depth=5, max_leaves=20, min_child_weight=20.0, reg_lambda=6.0, colsample_bytree=0.65, subsample=0.80, n_bins=36, random_state=342+f)
        p_ge2_va, p_ge2_te = gb_ge2.fit_predict_binary(X_tr_gk, m23.astype(np.float32), X_va_gk, X_te_gk_both)
        gb_3g2 = SparseHistGBDT(n_estimators=210, learning_rate=0.045, max_depth=5, max_leaves=20, min_child_weight=20.0, reg_lambda=6.0, colsample_bytree=0.65, subsample=0.80, n_bins=36, random_state=442+f)
        p_3g2_va, p_3g2_te = gb_3g2.fit_predict_binary(X_tr_gk[m23], (y_tr_f[m23] == 3).astype(np.float32), X_va_gk, X_te_gk_both)
        p_gf_va = np.column_stack([1 - p_ge2_va, p_ge2_va * (1 - p_3g2_va), p_ge2_va * p_3g2_va])
        p_gf_te = np.column_stack([1 - p_ge2_te, p_ge2_te * (1 - p_3g2_te), p_ge2_te * p_3g2_te])

        gb_le2 = SparseHistGBDT(n_estimators=210, learning_rate=0.045, max_depth=5, max_leaves=20, min_child_weight=20.0, reg_lambda=6.0, colsample_bytree=0.65, subsample=0.80, n_bins=36, random_state=542+f)
        p_le2_va, p_le2_te = gb_le2.fit_predict_binary(X_tr_gk, m12.astype(np.float32), X_va_gk, X_te_gk_both)
        gb_1g2 = SparseHistGBDT(n_estimators=210, learning_rate=0.045, max_depth=5, max_leaves=16, min_child_weight=16.0, reg_lambda=6.0, colsample_bytree=0.65, subsample=0.80, n_bins=36, random_state=642+f)
        p_1g2_va, p_1g2_te = gb_1g2.fit_predict_binary(X_tr_gk[m12], (y_tr_f[m12] == 1).astype(np.float32), X_va_gk, X_te_gk_both)
        p_gr_va = np.column_stack([p_le2_va * p_1g2_va, p_le2_va * (1 - p_1g2_va), 1 - p_le2_va])
        p_gr_te = np.column_stack([p_le2_te * p_1g2_te, p_le2_te * (1 - p_1g2_te), 1 - p_le2_te])

        oof_gb_ord[va_idx] = 0.5 * p_gf_va + 0.5 * p_gr_va
        p_go_te = 0.5 * p_gf_te + 0.5 * p_gr_te
        test_gb_ord_A += (0.45 * p_go_te[:n_test] + 0.55 * p_go_te[n_test:]) / 5.0

        # Clean 27-TE-col Multi-Depth HistGBDTs (D5, D6, D4)
        X_tr_gc = np.hstack([base_df[compact_gb_cols].values[tr_idx], te_tr[:, clean_gb_idx]]).astype(np.float64)
        X_va_gc = np.hstack([base_df[compact_gb_cols].values[va_idx], te_va[:, clean_gb_idx]]).astype(np.float64)
        X_te_gc_fold = np.hstack([base_df[compact_gb_cols].values[te_idx], te_te_fold[:, clean_gb_idx]]).astype(np.float64)
        X_te_gc_both = np.vstack([X_te_gc_fold, X_te_gb_cln_100])

        gb5 = SparseHistGBDT(n_estimators=240, learning_rate=0.045, max_depth=5, max_leaves=24, min_child_weight=20.0, reg_lambda=6.0, colsample_bytree=0.65, subsample=0.80, n_bins=36, random_state=42+f)
        p5_va, p5_te = gb5.fit_predict(X_tr_gc, y_tr_f - 1, X_va_gc, X_te_gc_both)
        oof_gb_d5[va_idx] = p5_va
        test_gb_d5_A += (0.45 * p5_te[:n_test] + 0.55 * p5_te[n_test:]) / 5.0

        gb6 = SparseHistGBDT(n_estimators=240, learning_rate=0.04, max_depth=6, max_leaves=31, min_child_weight=18.0, reg_lambda=6.0, colsample_bytree=0.60, subsample=0.80, n_bins=32, random_state=242+f)
        p6_va, p6_te = gb6.fit_predict(X_tr_gc, y_tr_f - 1, X_va_gc, X_te_gc_both)
        oof_gb_d6[va_idx] = p6_va
        test_gb_d6_A += (0.45 * p6_te[:n_test] + 0.55 * p6_te[n_test:]) / 5.0

        gb4 = SparseHistGBDT(n_estimators=260, learning_rate=0.045, max_depth=4, max_leaves=15, min_child_weight=22.0, reg_lambda=6.0, colsample_bytree=0.70, subsample=0.80, n_bins=48, random_state=142+f)
        p4_va, p4_te = gb4.fit_predict(X_tr_gc, y_tr_f - 1, X_va_gc, X_te_gc_both)
        oof_gb_d4[va_idx] = p4_va
        test_gb_d4_A += (0.45 * p4_te[:n_test] + 0.55 * p4_te[n_test:]) / 5.0
        print(f"      Plan A Fold {f+1}/5 finished ({time.time()-t0:.1f}s)")

    # [4/5] Plan B: Full-15,000-Row Final Models trained on 10-Fold Cross-Fitted TE & evaluated on Aligned 90%/100% Test TE
    print("[4/5] Training Plan B (Full-15,000-Row Final Models with 10-Fold Cross-Fitted TE)...")
    X_tr_f2_15k = np.hstack([base_df[no_diff_spline_cols].values[:n_train], te_tr_cf10]).astype(np.float64)
    X_te_f2_al = np.hstack([base_df[no_diff_spline_cols].values[te_idx], te_test_aligned]).astype(np.float64)
    Xs15_tr, Xs15_te = build_spline_matrix(X_tr_f2_15k, np.vstack([X_te_f2_al, X_te_f2_100]))
    p_sm15 = fit_predict_bohning_softmax(Xs15_tr, y_train, Xs15_te, C=0.04, max_iter=65)
    test_sp_f2_B = 0.50 * p_sm15[:n_test] + 0.50 * p_sm15[n_test:]

    m23_all = (y_train >= 2)
    m12_all = (y_train <= 2)
    p_ge2 = fit_predict_binary_logit(Xs15_tr, m23_all.astype(np.float64), Xs15_te, C=0.04, max_iter=60)
    p_3g2 = fit_predict_binary_logit(Xs15_tr[m23_all], (y_train[m23_all] == 3).astype(np.float64), Xs15_te, C=0.04, max_iter=60)
    p_le2 = fit_predict_binary_logit(Xs15_tr, m12_all.astype(np.float64), Xs15_te, C=0.04, max_iter=60)
    p_1g2 = fit_predict_binary_logit(Xs15_tr[m12_all], (y_train[m12_all] == 1).astype(np.float64), Xs15_te, C=0.04, max_iter=60)
    p_o15 = 0.5 * np.column_stack([1 - p_ge2, p_ge2 * (1 - p_3g2), p_ge2 * p_3g2]) + 0.5 * np.column_stack([p_le2 * p_1g2, p_le2 * (1 - p_1g2), 1 - p_le2])
    test_ord_f2_B = 0.50 * p_o15[:n_test] + 0.50 * p_o15[n_test:]

    X_tr_f1f2_15k = np.hstack([base_df[pruned_spline_base_cols].values[:n_train], te_tr_cf10]).astype(np.float64)
    X_te_f1f2_al = np.hstack([base_df[pruned_spline_base_cols].values[te_idx], te_test_aligned]).astype(np.float64)
    Xp15_tr, Xp15_te = build_spline_matrix(X_tr_f1f2_15k, np.vstack([X_te_f1f2_al, X_te_f1f2_100]))
    p_ge2 = fit_predict_binary_logit(Xp15_tr, m23_all.astype(np.float64), Xp15_te, C=0.04, max_iter=60)
    p_3g2 = fit_predict_binary_logit(Xp15_tr[m23_all], (y_train[m23_all] == 3).astype(np.float64), Xp15_te, C=0.04, max_iter=60)
    p_le2 = fit_predict_binary_logit(Xp15_tr, m12_all.astype(np.float64), Xp15_te, C=0.04, max_iter=60)
    p_1g2 = fit_predict_binary_logit(Xp15_tr[m12_all], (y_train[m12_all] == 1).astype(np.float64), Xp15_te, C=0.04, max_iter=60)
    p_op15 = 0.5 * np.column_stack([1 - p_ge2, p_ge2 * (1 - p_3g2), p_ge2 * p_3g2]) + 0.5 * np.column_stack([p_le2 * p_1g2, p_le2 * (1 - p_1g2), 1 - p_le2])
    test_ord_f1f2_B = 0.50 * p_op15[:n_test] + 0.50 * p_op15[n_test:]

    pm15 = fit_predict_res_mlp(X_tr_f1f2_15k, y_train, np.vstack([X_te_f1f2_al, X_te_f1f2_100]), seed=42)
    test_mlp_B = 0.50 * pm15[:n_test] + 0.50 * pm15[n_test:]

    X_tr_gk_15k = np.hstack([base_df[compact_gb_cols].values[:n_train], te_tr_cf10[:, gb_knn_idx]]).astype(np.float64)
    X_te_gk_al = np.hstack([base_df[compact_gb_cols].values[te_idx], te_test_aligned[:, gb_knn_idx]]).astype(np.float64)
    X_te_gk_15k = np.vstack([X_te_gk_al, X_te_gb_knn_100])
    gb_ge2 = SparseHistGBDT(n_estimators=230, learning_rate=0.045, max_depth=5, max_leaves=20, min_child_weight=20.0, reg_lambda=6.0, colsample_bytree=0.65, subsample=0.80, n_bins=36, random_state=342)
    _, p_ge2_t = gb_ge2.fit_predict_binary(X_tr_gk_15k, m23_all.astype(np.float32), X_te_gk_15k[:5], X_te_gk_15k)
    gb_3g2 = SparseHistGBDT(n_estimators=230, learning_rate=0.045, max_depth=5, max_leaves=20, min_child_weight=20.0, reg_lambda=6.0, colsample_bytree=0.65, subsample=0.80, n_bins=36, random_state=442)
    _, p_3g2_t = gb_3g2.fit_predict_binary(X_tr_gk_15k[m23_all], (y_train[m23_all] == 3).astype(np.float32), X_te_gk_15k[:5], X_te_gk_15k)
    gb_le2 = SparseHistGBDT(n_estimators=230, learning_rate=0.045, max_depth=5, max_leaves=20, min_child_weight=20.0, reg_lambda=6.0, colsample_bytree=0.65, subsample=0.80, n_bins=36, random_state=542)
    _, p_le2_t = gb_le2.fit_predict_binary(X_tr_gk_15k, m12_all.astype(np.float32), X_te_gk_15k[:5], X_te_gk_15k)
    gb_1g2 = SparseHistGBDT(n_estimators=230, learning_rate=0.045, max_depth=5, max_leaves=16, min_child_weight=16.0, reg_lambda=6.0, colsample_bytree=0.65, subsample=0.80, n_bins=36, random_state=642)
    _, p_1g2_t = gb_1g2.fit_predict_binary(X_tr_gk_15k[m12_all], (y_train[m12_all] == 1).astype(np.float32), X_te_gk_15k[:5], X_te_gk_15k)
    p_go15 = 0.5 * np.column_stack([1 - p_ge2_t, p_ge2_t * (1 - p_3g2_t), p_ge2_t * p_3g2_t]) + 0.5 * np.column_stack([p_le2_t * p_1g2_t, p_le2_t * (1 - p_1g2_t), 1 - p_le2_t])
    test_gb_ord_B = 0.50 * p_go15[:n_test] + 0.50 * p_go15[n_test:]

    X_tr_gc_15k = np.hstack([base_df[compact_gb_cols].values[:n_train], te_tr_cf10[:, clean_gb_idx]]).astype(np.float64)
    X_te_gc_al = np.hstack([base_df[compact_gb_cols].values[te_idx], te_test_aligned[:, clean_gb_idx]]).astype(np.float64)
    X_te_gc_15k = np.vstack([X_te_gc_al, X_te_gb_cln_100])
    gb5_15 = SparseHistGBDT(n_estimators=250, learning_rate=0.045, max_depth=5, max_leaves=24, min_child_weight=20.0, reg_lambda=6.0, colsample_bytree=0.65, subsample=0.80, n_bins=36, random_state=42)
    _, p5_15 = gb5_15.fit_predict(X_tr_gc_15k, y_train - 1, X_te_gc_15k[:5], X_te_gc_15k)
    test_gb_d5_B = 0.50 * p5_15[:n_test] + 0.50 * p5_15[n_test:]

    gb6_15 = SparseHistGBDT(n_estimators=240, learning_rate=0.04, max_depth=6, max_leaves=31, min_child_weight=18.0, reg_lambda=6.0, colsample_bytree=0.60, subsample=0.80, n_bins=32, random_state=242)
    _, p6_15 = gb6_15.fit_predict(X_tr_gc_15k, y_train - 1, X_te_gc_15k[:5], X_te_gc_15k)
    test_gb_d6_B = 0.50 * p6_15[:n_test] + 0.50 * p6_15[n_test:]

    gb4_15 = SparseHistGBDT(n_estimators=270, learning_rate=0.045, max_depth=4, max_leaves=15, min_child_weight=22.0, reg_lambda=6.0, colsample_bytree=0.70, subsample=0.80, n_bins=48, random_state=142)
    _, p4_15 = gb4_15.fit_predict(X_tr_gc_15k, y_train - 1, X_te_gc_15k[:5], X_te_gc_15k)
    test_gb_d4_B = 0.50 * p4_15[:n_test] + 0.50 * p4_15[n_test:]

    # [5/5] Dual-Plan Blending (45% Plan A + 55% Plan B) + Hybrid Arithmetic/Geometric Pooling
    print("[5/5] Combining Dual-Plan Families & Enforcing Exact 1700:1700:1700 Assignment...")
    cal = lambda p: calibrate_for_balanced_prior(p, train_priors)

    oof_sp_fam = 0.42 * cal(oof_ord_f2) + 0.34 * cal(oof_ord_f1f2) + 0.20 * cal(oof_sp_f2) + 0.04 * cal(oof_mlp)
    test_sp_A = 0.42 * cal(test_ord_f2_A) + 0.34 * cal(test_ord_f1f2_A) + 0.20 * cal(test_sp_f2_A) + 0.04 * cal(test_mlp_A)
    test_sp_B = 0.42 * cal(test_ord_f2_B) + 0.34 * cal(test_ord_f1f2_B) + 0.20 * cal(test_sp_f2_B) + 0.04 * cal(test_mlp_B)
    test_sp_fam = 0.45 * test_sp_A + 0.55 * test_sp_B

    oof_gb_fam = 0.45 * cal(oof_gb_ord) + 0.27 * cal(oof_gb_d5) + 0.14 * cal(oof_gb_d6) + 0.14 * cal(oof_gb_d4)
    test_gb_A = 0.45 * cal(test_gb_ord_A) + 0.27 * cal(test_gb_d5_A) + 0.14 * cal(test_gb_d6_A) + 0.14 * cal(test_gb_d4_A)
    test_gb_B = 0.45 * cal(test_gb_ord_B) + 0.27 * cal(test_gb_d5_B) + 0.14 * cal(test_gb_d6_B) + 0.14 * cal(test_gb_d4_B)
    test_gb_fam = 0.45 * test_gb_A + 0.55 * test_gb_B

    w_gb = 0.52
    oof_lin = w_gb * oof_gb_fam + (1.0 - w_gb) * oof_sp_fam
    oof_geo = (np.clip(oof_gb_fam, 1e-6, 1.0) ** w_gb) * (np.clip(oof_sp_fam, 1e-6, 1.0) ** (1.0 - w_gb))
    oof_geo /= oof_geo.sum(axis=1, keepdims=True)
    oof_final = 0.50 * oof_lin + 0.50 * oof_geo

    test_lin = w_gb * test_gb_fam + (1.0 - w_gb) * test_sp_fam
    test_geo = (np.clip(test_gb_fam, 1e-6, 1.0) ** w_gb) * (np.clip(test_sp_fam, 1e-6, 1.0) ** (1.0 - w_gb))
    test_geo /= test_geo.sum(axis=1, keepdims=True)
    test_final = 0.50 * test_lin + 0.50 * test_geo

    bacc, recs = balanced_accuracy(y_train, np.argmax(oof_final, axis=1) + 1)
    print(f"      Final V6 Dual-Plan Hybrid OOF Balanced Accuracy: {bacc:.5f} | Recalls: {[round(r, 4) for r in recs]}")

    final_test_preds = enforce_exact_cardinality(test_final, target_counts=(1700, 1700, 1700))
    sub_df = pd.DataFrame({
        "id": test_df["id"].values if "id" in test_df.columns else np.arange(n_test),
        "label": final_test_preds.astype(int)
    })
    sub_df.to_csv("submission.csv", index=False)
    print(f"      Saved submission.csv (class counts: {sub_df['label'].value_counts().to_dict()})")
    print(f"      Total pipeline runtime: {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()