"""
生成论文图表:
图1: 特征重要性柱状图 (前15个特征)
图2: 预测值vs真实值散点图 (3个子图)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False
rcParams['font.size'] = 12
rcParams['axes.labelsize'] = 13
rcParams['axes.titlesize'] = 14
rcParams['xtick.labelsize'] = 11
rcParams['ytick.labelsize'] = 11
rcParams['legend.fontsize'] = 11

PROJECT_DIR = r"C:\al7xxx"

# ============================================================
# 图1: 特征重要性柱状图
# ============================================================
print("生成图1: 特征重要性柱状图...")

fi = pd.read_csv(os.path.join(PROJECT_DIR, "final_feature_importance.csv"))
fi_top15 = fi.head(15).iloc[::-1]  # 反转，让最重要的在最上面

fig, ax = plt.subplots(figsize=(10, 7))

bars = ax.barh(fi_top15['feature'], fi_top15['importance'],
               color='#2E86AB', edgecolor='#1B4F72', linewidth=0.5, height=0.7)

# 在柱子上标注数值
for bar, val in zip(bars, fi_top15['importance']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left', fontsize=10, color='#1B4F72')

ax.set_xlabel('Feature Importance (Gini Importance)', fontsize=13)
ax.set_ylabel('Feature', fontsize=13)
ax.set_title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, fi_top15['importance'].max() * 1.15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig1_path = os.path.join(PROJECT_DIR, "Fig1_feature_importance.png")
plt.savefig(fig1_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  已保存: {fig1_path}")

# ============================================================
# 图2: 预测值vs真实值散点图 (3个子图)
# ============================================================
print("生成图2: 预测值vs真实值散点图...")

pred_df = pd.read_csv(os.path.join(PROJECT_DIR, "final_predictions.csv"))

targets = [
    ("yield_strength", "Yield Strength (MPa)", "#E74C3C"),
    ("tensile_strength", "Tensile Strength (MPa)", "#2E86AB"),
    ("elongation", "Elongation (%)", "#27AE60"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

for idx, (col, label, color) in enumerate(targets):
    ax = axes[idx]
    true_vals = pred_df[f"true_{col}"].values
    pred_vals = pred_df[f"pred_{col}"].values

    # 计算R²
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    ss_tot = np.sum((true_vals - np.mean(true_vals)) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    mae = np.mean(np.abs(true_vals - pred_vals))

    # 散点图
    ax.scatter(true_vals, pred_vals, c=color, alpha=0.6, s=50,
               edgecolors='white', linewidth=0.5, zorder=3)

    # y=x参考线
    min_val = min(true_vals.min(), pred_vals.min())
    max_val = max(true_vals.max(), pred_vals.max())
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin],
            [min_val - margin, max_val + margin],
            'k--', linewidth=1.5, alpha=0.7, label='Ideal Prediction (y=x)', zorder=2)

    # 标注R², RMSE, MAE
    textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}'
    props = dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor=color)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)

    ax.set_xlabel(f'True {label}', fontsize=12)
    ax.set_ylabel(f'Predicted {label}', fontsize=12)
    ax.set_title(f'({chr(97+idx)}) {label.split("(")[0].strip()}', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    ax.set_aspect('equal', adjustable='box')

plt.tight_layout()
fig2_path = os.path.join(PROJECT_DIR, "Fig2_prediction_scatter.png")
plt.savefig(fig2_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"  已保存: {fig2_path}")

# ============================================================
# 额外: 生成英文标注版本 (投稿用)
# ============================================================
print("生成英文标注版本...")

# 图1英文版
fig, ax = plt.subplots(figsize=(10, 7))
# 特征名英文映射
feat_map = {
    "total_aging_time": "Total Aging Time",
    "Fe_Si_total": "Fe+Si Impurity Total",
    "Fe": "Fe Content",
    "aging_temp_time": "Aging Temp×Time",
    "Zn_Mg_product": "Zn×Mg Product",
    "Zn_Mg_Cu_total": "Zn+Mg+Cu Total",
    "Al": "Al Content",
    "solution_temp_time": "Solution Temp×Time",
    "aging_time": "Aging Time",
    "aging_time_log": "log(Aging Time)",
    "Ti": "Ti Content",
    "alloy_agingtemp": "(Zn+Mg+Cu)×Aging Temp",
    "Zn_aging_interaction": "Zn×Aging Temp",
    "Zn": "Zn Content",
    "solution_temp": "Solution Temperature",
    "Mg_solution_interaction": "Mg×Solution Temp",
    "Zr_Mn_Cr_total": "Zr+Mn+Cr Total",
    "Mg_Cu_ratio": "Mg/Cu Ratio",
    "Zn_Mg_ratio": "Zn/Mg Ratio",
    "Mg": "Mg Content",
    "Cu": "Cu Content",
    "Cu_aging_interaction": "Cu×Aging Temp",
    "has_pre_aging": "Two-Stage Aging (0/1)",
    "has_deformation": "Pre-Deformation (0/1)",
    "Zr": "Zr Content",
    "Mn": "Mn Content",
    "Cr": "Cr Content",
    "Si": "Si Content",
    "homogenization_temp": "Homogenization Temp",
    "homogenization_time": "Homogenization Time",
    "quenching_rate": "Quenching Rate",
    "pre_aging_temp": "Pre-Aging Temp",
    "pre_aging_time": "Pre-Aging Time",
    "deformation_ratio": "Deformation Ratio",
    "solution_time": "Solution Time",
}
fi_top15_en = fi_top15.copy()
fi_top15_en['feature_en'] = fi_top15_en['feature'].map(feat_map).fillna(fi_top15_en['feature'])

bars = ax.barh(fi_top15_en['feature_en'], fi_top15_en['importance'],
               color='#2E86AB', edgecolor='#1B4F72', linewidth=0.5, height=0.7)
for bar, val in zip(bars, fi_top15_en['importance']):
    ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', ha='left', fontsize=10, color='#1B4F72')
ax.set_xlabel('Feature Importance (Gini Importance)', fontsize=13)
ax.set_ylabel('Feature', fontsize=13)
ax.set_title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, fi_top15_en['importance'].max() * 1.15)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
fig1_en = os.path.join(PROJECT_DIR, "Fig1_feature_importance_EN.png")
plt.savefig(fig1_en, dpi=300, bbox_inches='tight')
plt.close()
print(f"  已保存: {fig1_en}")

# 图2英文版 (已经是英文标注了，直接复制)
import shutil
fig2_en = os.path.join(PROJECT_DIR, "Fig2_prediction_scatter_EN.png")
shutil.copy(fig2_path, fig2_en)
print(f"  已保存: {fig2_en}")

print("\n所有图表生成完成！")
print(f"中文版: {fig1_path}, {fig2_path}")
print(f"英文版: {fig1_en}, {fig2_en}")
