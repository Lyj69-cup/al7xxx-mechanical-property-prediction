# -*- coding: utf-8 -*-
"""Generate v4 figures (English, 300 dpi) from OOF predictions."""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RES = str(ROOT / "results")
OUT = RES
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#444444", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.7,
    "figure.dpi": 300, "savefig.dpi": 300,
})
C_ET, C_FU, C_LO = "#4C72B0", "#55A868", "#DD8452"
C_YS, C_UTS, C_EL = "#4C72B0", "#DD8452", "#55A868"

res = json.load(open(os.path.join(RES, "recompute_v4_results.json"), encoding="utf-8"))
rand = pd.read_csv(os.path.join(RES, "oof_random_v4.csv"))
loso = pd.read_csv(os.path.join(RES, "oof_loso_v4.csv"))

# ---------------- Fig 1: random vs LOSO R2 ----------------
targets = ["YS_MPa", "UTS_MPa", "EL_pct"]
tlabels = ["Yield strength", "Tensile strength", "Elongation"]
et = [res["random_phys"]["ET"][t]["R2_mean"] for t in targets]
fu = [res["random_phys"]["Fusion"][t]["R2_mean"] for t in targets]
lo = [res["loso_pooled"]["Fusion"]["small_sources"][t]["R2"] for t in targets]
x = np.arange(3); w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 4.3))
ax.bar(x - w, et, w, label="Random 5-fold, ET", color=C_ET)
ax.bar(x,     fu, w, label="Random 5-fold, Fusion", color=C_FU)
ax.bar(x + w, lo, w, label="LOSO, Fusion (engineering)", color=C_LO)
ax.axhline(0, color="#333333", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(tlabels)
ax.set_ylabel(r"$R^2$"); ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
for i, v in enumerate(et): ax.text(i-w, v+0.02, f"{v:.3f}", ha="center", fontsize=8)
for i, v in enumerate(fu): ax.text(i,   v+0.02, f"{v:.3f}", ha="center", fontsize=8)
for i, v in enumerate(lo): ax.text(i+w, v+0.02, f"{v:.3f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Fig2_random_vs_LOSO.png")); plt.close(fig)

# ---------------- Fig 2: parity (random OOF) ----------------
fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
for ax, t, lab, col in zip(axes, targets, tlabels, [C_YS, C_UTS, C_EL]):
    d = rand[[t+"_pred", t]].dropna()
    yt, yp = d[t].values, d[t+"_pred"].values
    r2 = 1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)
    rmse = np.sqrt(np.mean((yt-yp)**2))
    ax.scatter(yt, yp, s=22, alpha=0.65, color=col, edgecolor="white", linewidth=0.4)
    lo_, hi_ = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo_, hi_], [lo_, hi_], "--", color="#333333", lw=1.1)
    unit = "MPa" if t != "EL_pct" else "%"
    ax.set_xlabel(f"Measured {lab.lower()} ({unit})")
    ax.set_ylabel(f"Predicted {lab.lower()} ({unit})")
    ax.text(0.05, 0.93, f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.2f} {unit}",
            transform=ax.transAxes, va="top", fontsize=9.5,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.85))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Fig1_parity_random.png")); plt.close(fig)

# ---------------- Fig 3: feature importance ----------------
imp = res["feature_importance_top15"]
items = sorted(imp.items(), key=lambda kv: kv[1])
names = [k for k, _ in items]; vals = [v for _, v in items]
fig, ax = plt.subplots(figsize=(7.4, 5.6))
ax.barh(range(len(names)), vals, color="#4C72B0", alpha=0.85)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
ax.set_xlabel("Mean impurity-based importance (ExtraTrees, random folds)")
for i, v in enumerate(vals): ax.text(v+0.002, i, f"{v:.3f}", va="center", fontsize=7.5)
ax.set_xlim(0, max(vals)*1.12)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Fig4_feature_importance.png")); plt.close(fig)

# ---------------- Fig 4: per-source MAE (engineering) ----------------
small = loso[loso.source != "BHAT2023"].copy()
sources = ["XU2015", "XU2016", "XU2018", "LI2020", "PMC9571267"]
slabel = ["XU2015", "XU2016", "XU2018", "LI2020", "PMC9571267"]
mae = {t: [] for t in targets}
for s in sources:
    d = small[small.source == s]
    for t in targets:
        dd = d[[t+"_pred", t]].dropna()
        mae[t].append(np.mean(np.abs(dd[t]-dd[t+"_pred"])) if len(dd) else np.nan)
x = np.arange(len(sources)); w = 0.26
fig, ax = plt.subplots(figsize=(8.2, 4.5))
ax.bar(x-w, mae["YS_MPa"], w, label="Yield strength", color=C_YS)
ax.bar(x,   mae["UTS_MPa"], w, label="Tensile strength", color=C_UTS)
ax.bar(x+w, mae["EL_pct"], w, label="Elongation", color=C_EL)
ax.set_xticks(x); ax.set_xticklabels(slabel, fontsize=9)
ax.set_ylabel("MAE (MPa for strength; % for elongation)")
ax.legend(frameon=False, fontsize=9.5)
for i, t in enumerate(targets):
    for j, v in enumerate(mae[t]):
        if not np.isnan(v):
            ax.text(j+(i-1)*w, v+0.6, f"{v:.1f}", ha="center", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "Fig3_per_source_MAE.png")); plt.close(fig)

print("Figures written to", OUT)
for f in sorted(os.listdir(OUT)):
    print(" ", f, os.path.getsize(os.path.join(OUT, f)))
