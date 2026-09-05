"""
用全部300条数据训练最终模型并保存
用于后续课题组数据的外部验证
"""
import os, sys, json, warnings, pickle
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

PROJECT_DIR = r"C:\al7xxx"

print("=" * 70)
print("  训练最终模型 (用全部300条数据) 并保存")
print("=" * 70)

# 加载数据
df = pd.read_csv(os.path.join(PROJECT_DIR, "al7xxx_real_dataset_v2.csv"))
print(f"\n数据量: {len(df)} 条, 来源: {df['source'].nunique()} 个")

comp_cols = ["Zn", "Mg", "Cu", "Zr", "Mn", "Cr", "Fe", "Si", "Ti", "Al"]
proc_cols = ["homogenization_temp", "homogenization_time", "solution_temp",
             "solution_time", "quenching_rate", "aging_temp", "aging_time",
             "pre_aging_temp", "pre_aging_time", "deformation_ratio"]
target_cols = ["yield_strength", "tensile_strength", "elongation"]

# 特征工程 (与train_final.py完全一致)
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

extra_cols = [
    "Zn_Mg_ratio", "Zn_Mg_Cu_total", "Mg_Cu_ratio", "Zr_Mn_Cr_total",
    "Fe_Si_total", "Zn_Mg_product", "total_aging_time", "aging_temp_time",
    "solution_temp_time", "has_pre_aging", "has_deformation", "aging_time_log",
    "Zn_aging_interaction", "Mg_solution_interaction", "alloy_agingtemp"
]
all_cols = comp_cols + proc_cols + extra_cols

d = feature_engineering(df)
X = d[all_cols].values
y = d[target_cols].values
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
print(f"特征数: {len(all_cols)} (原始20 + 衍生15)")
print(f"X shape: {X.shape}, y shape: {y.shape}")

# 标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 训练6个基模型
print("\n训练6个基学习器...")
models = {}

models["RF"] = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=500, max_features=0.5, random_state=42, n_jobs=-1))
models["RF"].fit(X_scaled, y_scaled)
print("  [1/6] Random Forest 完成")

models["ET"] = MultiOutputRegressor(ExtraTreesRegressor(
    n_estimators=500, max_features=0.5, random_state=42, n_jobs=-1))
models["ET"].fit(X_scaled, y_scaled)
print("  [2/6] Extra Trees 完成")

models["XGB"] = MultiOutputRegressor(xgb.XGBRegressor(
    n_estimators=500, max_depth=5, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.6, min_child_weight=3,
    reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=0))
models["XGB"].fit(X_scaled, y_scaled)
print("  [3/6] XGBoost 完成")

models["LGB"] = MultiOutputRegressor(lgb.LGBMRegressor(
    n_estimators=500, max_depth=7, learning_rate=0.03,
    subsample=0.8, colsample_bytree=0.6, min_child_samples=8,
    reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=-1))
models["LGB"].fit(X_scaled, y_scaled)
print("  [4/6] LightGBM 完成")

models["GBRT"] = MultiOutputRegressor(GradientBoostingRegressor(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42))
models["GBRT"].fit(X_scaled, y_scaled)
print("  [5/6] GBRT 完成")

models["SVR"] = MultiOutputRegressor(SVR(kernel='rbf', C=200, gamma='scale', epsilon=0.05))
models["SVR"].fit(X_scaled, y_scaled)
print("  [6/6] SVR 完成")

# 计算各模型在训练集上的预测（用于Stacking元学习器训练）
base_preds = {}
for name, model in models.items():
    p = model.predict(X_scaled)
    base_preds[name] = scaler_y.inverse_transform(p)

# 简单平均
avg_pred = np.mean([base_preds[n] for n in models], axis=0)

# 分目标加权 (基于训练集R²)
weighted_pred = np.zeros_like(y)
for i, tgt in enumerate(target_cols):
    w = []
    for n in models:
        r2 = r2_score(y[:, i], base_preds[n][:, i])
        w.append(max(r2, 0.01))
    w = np.array(w) / sum(w)
    weighted_pred[:, i] = sum(base_preds[n][:, i] * w[j] for j, n in enumerate(models))

# Stacking元特征
meta_X = np.hstack([base_preds[n] for n in models])

# 训练Stacking-Ridge元学习器
scaler_meta_r = StandardScaler()
meta_X_r = scaler_meta_r.fit_transform(meta_X)
meta_ridge = MultiOutputRegressor(Ridge(alpha=0.5))
meta_ridge.fit(meta_X_r, y)
stacking_r_pred = meta_ridge.predict(meta_X_r)
print("\nStacking-Ridge 元学习器训练完成")

# 训练Stacking-Bayes元学习器
scaler_meta_b = StandardScaler()
meta_X_b = scaler_meta_b.fit_transform(meta_X)
meta_bayes = MultiOutputRegressor(BayesianRidge())
meta_bayes.fit(meta_X_b, y)
stacking_b_pred = meta_bayes.predict(meta_X_b)
print("Stacking-Bayes 元学习器训练完成")

# 超级融合
super_pred = np.mean([avg_pred, weighted_pred, stacking_r_pred, stacking_b_pred], axis=0)

# 在训练集上评估（确认模型正常）
print("\n训练集上的性能 (用于 sanity check):")
for i, tgt in enumerate(target_cols):
    r2 = r2_score(y[:, i], super_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y[:, i], super_pred[:, i]))
    mae = mean_absolute_error(y[:, i], super_pred[:, i])
    print(f"  {tgt}: R²={r2:.4f}, RMSE={rmse:.2f}, MAE={mae:.2f}")

# 保存模型
save_data = {
    "models": models,
    "scaler_X": scaler_X,
    "scaler_y": scaler_y,
    "scaler_meta_ridge": scaler_meta_r,
    "meta_ridge": meta_ridge,
    "scaler_meta_bayes": scaler_meta_b,
    "meta_bayes": meta_bayes,
    "comp_cols": comp_cols,
    "proc_cols": proc_cols,
    "target_cols": target_cols,
    "extra_cols": extra_cols,
    "all_cols": all_cols,
    "model_names": list(models.keys()),
}

save_path = os.path.join(PROJECT_DIR, "final_model.pkl")
with open(save_path, "wb") as f:
    pickle.dump(save_data, f)
print(f"\n模型已保存: {save_path}")
print(f"文件大小: {os.path.getsize(save_path)/1024/1024:.1f} MB")
print("\n训练完成！可以运行 validate_with_lab_data.py 进行课题组数据验证")
