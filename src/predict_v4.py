# -*- coding: utf-8 -*-
"""
7xxx-series Al alloy mechanical-property prediction (v4, consistent with the paper).

The v4 ensemble is a leakage-free equal-weight average of six base models
(RF, ExtraTrees, XGBoost, LightGBM, GBRT, SVR). This script fits that ensemble on
the full v4 dataset (data/al7xxx_dataset_v4.csv) and predicts yield strength (YS),
ultimate tensile strength (UTS) and elongation (EL).

Usage
  Batch:      python predict_v4.py --input my_alloys.csv --output predictions.csv
  Interactive: python predict_v4.py
The input CSV needs the as-measured composition (wt.%) and heat-treatment columns;
see data/lab_data_template.csv. Missing columns are auto-filled with safe defaults.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recompute_v4 import features, models, ENSEMBLE, TGT, COMP_ALL, PROC

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# raw fields a user supplies; everything else is derived inside features()
RAW_COLS = COMP_ALL + PROC + ["temper", "product_form", "special_process"]
DEFAULTS = {**{c: 0.0 for c in COMP_ALL},
            **{c: np.nan for c in PROC},
            "temper": "unknown", "product_form": "unknown", "special_process": ""}


def complete_columns(df):
    """Add any missing raw columns with safe defaults so features() never breaks."""
    df = df.copy()
    for c, v in DEFAULTS.items():
        if c not in df.columns:
            df[c] = v
    for c in COMP_ALL + PROC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fit_full(df):
    """Fit the six-model ensemble on the full dataset for each target."""
    Xd, cols = features(df, "phys")
    X = np.nan_to_num(Xd[cols].values.astype(float), nan=0.0)
    Y = df[TGT].values.astype(float)
    sx = StandardScaler().fit(X)
    Xs = sx.transform(X)
    fitted, scalers_y = {}, {}
    factory = models()
    for ti in range(3):
        avail = ~np.isnan(Y[:, ti])
        sy = StandardScaler().fit(Y[avail, ti].reshape(-1, 1))
        ys = sy.transform(Y[avail, ti].reshape(-1, 1)).ravel()
        ms = {}
        for name in ENSEMBLE:
            m = factory[name]()
            m.fit(Xs[avail], ys)
            ms[name] = m
        fitted[ti] = ms
        scalers_y[ti] = sy
    return sx, fitted, scalers_y, cols


def predict_new(df_new, sx, fitted, scalers_y, cols):
    """Predict YS/UTS/EL for new records already containing the raw columns."""
    Xd, _ = features(df_new, "phys")
    for c in cols:
        if c not in Xd.columns:
            Xd[c] = 0.0
    X = np.nan_to_num(Xd[cols].values.astype(float), nan=0.0)
    Xs = sx.transform(X)
    out = np.zeros((len(df_new), 3))
    for ti in range(3):
        preds = []
        for name in ENSEMBLE:
            p = fitted[ti][name].predict(Xs)
            preds.append(scalers_y[ti].inverse_transform(p.reshape(-1, 1)).ravel())
        out[:, ti] = np.mean(preds, axis=0)
    return out


def interactive_record():
    print("Enter composition in wt.% and heat-treatment parameters.")
    print("Press Enter to use 0 for an element or to skip a process value.\n")
    rec = {}
    for c in ["Zn", "Mg", "Cu", "Zr", "Mn", "Cr", "Fe", "Si", "Ti"]:
        v = input(f"  {c} (wt.%): ").strip()
        rec[c] = float(v) if v else 0.0
    for c in ["solution_temp_C", "solution_time_h", "aging_temp_C",
              "aging_time_h", "pre_deformation_pct"]:
        v = input(f"  {c}: ").strip()
        rec[c] = float(v) if v else np.nan
    rec["temper"] = input("  temper (e.g. peak-aged (T6) / over-aged (T7) / unknown): ").strip() or "unknown"
    rec["product_form"] = input("  product_form (extrusion/forging/cast ingot/unknown): ").strip() or "unknown"
    rec["special_process"] = input("  special_process notes (optional): ").strip()
    return pd.DataFrame([rec])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="CSV with new alloy records")
    ap.add_argument("--output", default="predictions.csv", help="output CSV")
    args = ap.parse_args()

    print("Loading v4 dataset and fitting the ensemble (this takes ~10-30 s)...")
    df = pd.read_csv(DATA / "al7xxx_dataset_v4.csv", encoding="utf-8-sig")
    sx, fitted, scalers_y, cols = fit_full(df)
    print("Ensemble fitted on", len(df), "records.")

    if args.input:
        new = complete_columns(pd.read_csv(args.input))
    else:
        new = complete_columns(interactive_record())

    pred = predict_new(new, sx, fitted, scalers_y, cols)
    res = new.copy()
    res["YS_pred_MPa"] = np.round(pred[:, 0], 1)
    res["UTS_pred_MPa"] = np.round(pred[:, 1], 1)
    res["EL_pred_pct"] = np.round(pred[:, 2], 2)
    keep = ["YS_pred_MPa", "UTS_pred_MPa", "EL_pred_pct"]
    out_cols = ([c for c in new.columns if c in RAW_COLS] if not args.input
                else list(new.columns))
    res[out_cols + keep].to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\nPrediction result:")
    for i in range(len(res)):
        print(f"  YS={pred[i,0]:.1f} MPa  UTS={pred[i,1]:.1f} MPa  EL={pred[i,2]:.2f} %")
    print("\nSaved to", os.path.abspath(args.output))


if __name__ == "__main__":
    main()
