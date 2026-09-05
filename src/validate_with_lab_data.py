"""
用课题组实验数据验证训练好的模型
使用方法:
1. 把你的课题组数据整理成CSV，列名见下方模板
2. 运行: D:\python\python.exe C:\al7xxx\validate_with_lab_data.py
3. 结果保存在 C:\al7xxx\validation_results\
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
rcParams['axes.unicode_minus'] = False

PROJECT_DIR = r"C:\al7xxx"
RESULT_DIR = os.path.join(PROJECT_DIR, "validation_results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ============================================================
# 第一步: 加载模型
# ============================================================
print("=" * 70)
print("  课题组数据外部验证")
print("=" * 70)

model_path = os.path.join(PROJECT_DIR, "final_model.pkl")
if not os.path.exists(model_path):
    print(f"\n错误: 模型文件不存在: {model_path}")
    print("请先运行: D:\\python\\python.exe C:\\al7xxx\\train_and_save_model.py")
    sys.exit(1)

with open(model_path, "rb") as f:
    saved = pickle.load(f)

models = saved["models"]
scaler_X = saved["scaler_X"]
scaler_y = saved["scaler_y"]
scaler_meta_r = saved["scaler_meta_ridge"]
meta_ridge = saved["meta_ridge"]
scaler_meta_b = saved["scaler_meta_bayes"]
meta_bayes = saved["meta_bayes"]
comp_cols = saved["comp_cols"]
proc_cols = saved["proc_cols"]
target_cols = saved["target_cols"]
extra_cols = saved["extra_cols"]
all_cols = saved["all_cols"]
model_names = saved["model_names"]

print(f"\n已加载模型: {', '.join(model_names)}")
print(f"特征数: {len(all_cols)}")
print(f"目标: {', '.join(target_cols)}")

# ============================================================
# 第二步: 特征工程 (与训练时完全一致)
# ============================================================
def feature_engineering(data):
    d = data.copy()
    d["Zn_Mg_ratio"] = d["Zn"] / (d["Mg"] + 0.01)
    d["Zn_Mg_Cu_total"] = d["Zn"] + d["Mg"] + d["Cu"]
    d["Mg_Cu_ratio"] = d["Mg"] / (d["Cu"] + 0.01)
    d["Zr_Mn_Cr_total"] = d["Zr"] + d["Mn"] + d["Cr"]
    d["Fe_Si_total"] = d["Fe"] + d["Si"]
    d["Zn_Mg_product"] = d["Zn"] * d["Mg"]
    d["total_aging_time"] = d["aging_time"] + d["pre_aging_time"]
    d["aging_temp_time"] = d["aging_temp"] * d["aging_time"] / 1000
    d["solution_temp_time"] = d["solution_temp"] * d["solution_time"] / 100
    d["has_pre_aging"] = (d["pre_aging_time"] > 0).astype(int)
    d["has_deformation"] = (d["deformation_ratio"] > 0).astype(int)
    d["aging_time_log"] = np.log1p(d["aging_time"])
    d["Zn_aging_interaction"] = d["Zn"] * d["aging_temp"] / 100
    d["Mg_solution_interaction"] = d["Mg"] * d["solution_temp"] / 100
    d["alloy_agingtemp"] = (d["Zn"] + d["Mg"] + d["Cu"]) * d["aging_temp"] / 100
    return d

# ============================================================
# 第三步: 加载课题组数据
# ============================================================
# 优先查找用户数据
lab_data_path = os.path.join(PROJECT_DIR, "lab_validation_data.csv")
template_path = os.path.join(PROJECT_DIR, "lab_data_template.csv")

if not os.path.exists(lab_data_path):
    print(f"\n未找到课题组数据文件: {lab_data_path}")
    print("正在生成数据模板...")
    
    # 生成模板
    template = pd.DataFrame(columns=[
        "sample_id", "Zn", "Mg", "Cu", "Zr", "Mn", "Cr", "Fe", "Si", "Ti", "Al",
        "homogenization_temp", "homogenization_time", "solution_temp", "solution_time",
        "quenching_rate", "aging_temp", "aging_time", "pre_aging_temp", "pre_aging_time",
        "deformation_ratio", "yield_strength", "tensile_strength", "elongation"
    ])
    # 添加示例行
    example = {
        "sample_id": "example_001",
        "Zn": 8.0, "Mg": 2.2, "Cu": 1.5, "Zr": 0.1, "Mn": 0.15, "Cr": 0.18,
        "Fe": 0.15, "Si": 0.1, "Ti": 0.05, "Al": 87.57,
        "homogenization_temp": 470, "homogenization_time": 24,
        "solution_temp": 475, "solution_time": 1, "quenching_rate": 50,
        "aging_temp": 120, "aging_time": 24, "pre_aging_temp": 0, "pre_aging_time": 0,
        "deformation_ratio": 0,
        "yield_strength": 550, "tensile_strength": 600, "elongation": 12.0
    }
    template = pd.concat([template, pd.DataFrame([example])], ignore_index=True)
    template.to_csv(template_path, index=False, encoding="utf-8-sig")
    print(f"模板已生成: {template_path}")
    print("\n请按以下步骤操作:")
    print("1. 打开模板文件，替换example行为你的真实实验数据")
    print("2. 可以添加多行，每行一个样品")
    print("3. 保存为 lab_validation_data.csv 放在 C:\\al7xxx\\ 目录下")
    print("4. 重新运行此脚本")
    print("\n必填列说明:")
    print("  成分(10列): Zn, Mg, Cu, Zr, Mn, Cr, Fe, Si, Ti, Al (单位: wt%)")
    print("  工艺(10列): homogenization_temp(℃), homogenization_time(h),")
    print("             solution_temp(℃), solution_time(h), quenching_rate(℃/s),")
    print("             aging_temp(℃), aging_time(h), pre_aging_temp(℃),")
    print("             pre_aging_time(h), deformation_ratio(%)")
    print("  性能(3列): yield_strength(MPa), tensile_strength(MPa), elongation(%)")
    print("  其他: sample_id (样品编号，可选)")
    sys.exit(0)

print(f"\n加载课题组数据: {lab_data_path}")
lab_df = pd.read_csv(lab_data_path)
print(f"数据量: {len(lab_df)} 条")

# 检查必要列
required_cols = comp_cols + proc_cols
missing = [c for c in required_cols if c not in lab_df.columns]
if missing:
    print(f"\n错误: 缺少以下必要列: {missing}")
    print("请参考模板文件: lab_data_template.csv")
    sys.exit(1)

# 检查是否有实测值（用于对比）
has_true = all(c in lab_df.columns for c in target_cols)
if has_true:
    print("检测到实测性能值，将进行预测精度评估")
else:
    print("未检测到实测性能值，仅输出预测结果")

# ============================================================
# 第四步: 预测
# ============================================================
print("\n进行特征工程和预测...")
lab_fe = feature_engineering(lab_df)
X_lab = lab_fe[all_cols].values
X_lab = np.nan_to_num(X_lab, nan=0.0, posinf=1e6, neginf=-1e6)
X_lab_scaled = scaler_X.transform(X_lab)

# 基模型预测
base_preds = {}
for name, model in models.items():
    p = model.predict(X_lab_scaled)
    base_preds[name] = scaler_y.inverse_transform(p)

# 简单平均
avg_pred = np.mean([base_preds[n] for n in model_names], axis=0)

# 分目标加权 (用训练集的权重，这里简化为等权)
weighted_pred = avg_pred.copy()

# Stacking
meta_X_lab = np.hstack([base_preds[n] for n in model_names])
stacking_r_pred = meta_ridge.predict(scaler_meta_r.transform(meta_X_lab))
stacking_b_pred = meta_bayes.predict(scaler_meta_b.transform(meta_X_lab))

# 超级融合 (最终模型)
super_pred = np.mean([avg_pred, weighted_pred, stacking_r_pred, stacking_b_pred], axis=0)

# ============================================================
# 第五步: 保存预测结果
# ============================================================
result_df = lab_df.copy()
for i, tgt in enumerate(target_cols):
    result_df[f"pred_{tgt}"] = super_pred[:, i]
    if has_true:
        result_df[f"error_{tgt}"] = super_pred[:, i] - lab_df[tgt].values
        result_df[f"abs_error_{tgt}"] = np.abs(super_pred[:, i] - lab_df[tgt].values)
        result_df[f"rel_error_{tgt}"] = np.abs(super_pred[:, i] - lab_df[tgt].values) / lab_df[tgt].values * 100

pred_csv = os.path.join(RESULT_DIR, "lab_predictions.csv")
result_df.to_csv(pred_csv, index=False, encoding="utf-8-sig")
print(f"\n预测结果已保存: {pred_csv}")

# ============================================================
# 第六步: 如果有实测值，计算评估指标
# ============================================================
if has_true:
    y_true = lab_df[target_cols].values
    print("\n" + "=" * 70)
    print("  外部验证结果 (课题组数据)")
    print("=" * 70)
    print(f"\n{'指标':<12s} {'屈服强度':>12s} {'抗拉强度':>12s} {'延伸率':>12s} {'平均':>12s}")
    print("-" * 64)
    
    metrics = {}
    for i, tgt in enumerate(target_cols):
        r2 = r2_score(y_true[:, i], super_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_true[:, i], super_pred[:, i]))
        mae = mean_absolute_error(y_true[:, i], super_pred[:, i])
        mape = np.mean(np.abs((y_true[:, i] - super_pred[:, i]) / y_true[:, i])) * 100
        metrics[tgt] = {"R2": r2, "RMSE": rmse, "MAE": mae, "MAPE": mape}
    
    for m_name, m_key in [("R²", "R2"), ("RMSE", "RMSE"), ("MAE", "MAE"), ("MAPE(%)", "MAPE")]:
        row = f"{m_name:<12s}"
        vals = []
        for tgt in target_cols:
            v = metrics[tgt][m_key]
            vals.append(v)
            row += f" {v:>12.4f}" if m_key == "R2" else f" {v:>12.2f}"
        row += f" {np.mean(vals):>12.4f}" if m_key == "R2" else f" {np.mean(vals):>12.2f}"
        print(row)
    
    # 保存评估指标
    with open(os.path.join(RESULT_DIR, "validation_metrics.txt"), "w", encoding="utf-8") as f:
        f.write("课题组数据外部验证结果\n")
        f.write(f"验证样本数: {len(lab_df)}\n\n")
        f.write(f"{'指标':<12s} {'屈服强度':>12s} {'抗拉强度':>12s} {'延伸率':>12s} {'平均':>12s}\n")
        f.write("-" * 64 + "\n")
        for m_name, m_key in [("R²", "R2"), ("RMSE", "RMSE"), ("MAE", "MAE"), ("MAPE(%)", "MAPE")]:
            row = f"{m_name:<12s}"
            vals = []
            for tgt in target_cols:
                v = metrics[tgt][m_key]
                vals.append(v)
                row += f" {v:>12.4f}" if m_key == "R2" else f" {v:>12.2f}"
            row += f" {np.mean(vals):>12.4f}" if m_key == "R2" else f" {np.mean(vals):>12.2f}"
            f.write(row + "\n")
    
    # ============================================================
    # 第七步: 生成验证图
    # ============================================================
    print("\n生成验证图表...")
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    colors = ["#E74C3C", "#2E86AB", "#27AE60"]
    labels = ["Yield Strength (MPa)", "Tensile Strength (MPa)", "Elongation (%)"]
    
    for idx, (tgt, label, color) in enumerate(zip(target_cols, labels, colors)):
        ax = axes[idx]
        true_vals = y_true[:, idx]
        pred_vals = super_pred[:, idx]
        
        ax.scatter(true_vals, pred_vals, c=color, alpha=0.7, s=80,
                   edgecolors='white', linewidth=1, zorder=3)
        
        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        margin = (max_val - min_val) * 0.08
        ax.plot([min_val - margin, max_val + margin],
                [min_val - margin, max_val + margin],
                'k--', linewidth=1.5, alpha=0.7, label='Ideal (y=x)', zorder=2)
        
        r2 = metrics[tgt]["R2"]
        rmse = metrics[tgt]["RMSE"]
        mae = metrics[tgt]["MAE"]
        textstr = f'R² = {r2:.3f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}\nN = {len(lab_df)}'
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
    fig_path = os.path.join(RESULT_DIR, "lab_validation_scatter.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"验证图已保存: {fig_path}")
    
    # 误差柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(lab_df))
    width = 0.25
    
    for i, (tgt, color) in enumerate(zip(target_cols, colors)):
        errors = np.abs(super_pred[:, i] - y_true[:, i])
        if tgt == "elongation":
            errors = errors  # 延伸率单位是%
        ax.bar(x + i * width, errors, width, label=tgt.replace('_', ' ').title(), color=color, alpha=0.8)
    
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Prediction Error per Sample (External Validation)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    if 'sample_id' in lab_df.columns:
        ax.set_xticklabels(lab_df['sample_id'], rotation=45, ha='right')
    else:
        ax.set_xticklabels([f'S{i+1}' for i in range(len(lab_df))], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    err_path = os.path.join(RESULT_DIR, "lab_validation_error_bar.png")
    plt.savefig(err_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"误差图已保存: {err_path}")
    
    print("\n" + "=" * 70)
    print("  验证完成！")
    print("=" * 70)
    print(f"\n结果文件保存在: {RESULT_DIR}")
    print(f"  1. lab_predictions.csv - 每条样品的预测值和误差")
    print(f"  2. validation_metrics.txt - 汇总评估指标")
    print(f"  3. lab_validation_scatter.png - 预测值vs真实值散点图")
    print(f"  4. lab_validation_error_bar.png - 各样品误差柱状图")
    print("\n建议: 将验证结果补充到论文的3.5节或新增'External Validation'小节")
    print("这将大幅提升论文的说服力和接收概率！")

else:
    print("\n" + "=" * 70)
    print("  预测完成（无实测值对比）")
    print("=" * 70)
    print(f"\n预测结果已保存: {pred_csv}")
    print("\n预测值预览:")
    for i, tgt in enumerate(target_cols):
        print(f"  {tgt}: {super_pred[:, i]}")
