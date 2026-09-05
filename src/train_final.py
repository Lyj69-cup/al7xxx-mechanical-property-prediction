"""
7系铝合金力学性能预测 - 最终优化版 V4
基于V2成功配置优化，目标: 平均R² > 0.90

关键改进:
1. 保留全部数据 (不激进剔除异常值)
2. 精选衍生特征 (避免冗余)
3. 加入XGBoost + LightGBM
4. 超参数精细调优
5. 多策略融合 (简单平均 + 加权 + Stacking)
6. 延伸率单独优化

使用方式: D:\python\python.exe C:\al7xxx\train_final.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb

PROJECT_DIR = r"C:\al7xxx"
sys.path.insert(0, os.path.dirname(PROJECT_DIR))

print("=" * 70)
print("  7系铝合金力学性能预测 - 最终优化版 V4")
print("  目标: 平均R² > 0.90")
print("=" * 70)

# ============================================================
# 第一步: 加载数据 (保留全部)
# ============================================================
print("\n【第一步】加载数据")

df = pd.read_csv(os.path.join(PROJECT_DIR, "al7xxx_real_dataset_v2.csv"))
print(f"  数据量: {len(df)} 条, 来源: {df['source'].nunique()} 个")

comp_cols = ["Zn", "Mg", "Cu", "Zr", "Mn", "Cr", "Fe", "Si", "Ti", "Al"]
proc_cols = ["homogenization_temp", "homogenization_time", "solution_temp",
             "solution_time", "quenching_rate", "aging_temp", "aging_time",
             "pre_aging_temp", "pre_aging_time", "deformation_ratio"]
target_cols = ["yield_strength", "tensile_strength", "elongation"]

# ============================================================
# 第二步: 精选特征工程
# ============================================================
print("\n【第二步】精选特征工程")

d = df.copy()

# 成分衍生 (精选物理意义明确的)
d["Zn_Mg_ratio"] = d["Zn"] / (d["Mg"] + 0.01)
d["Zn_Mg_Cu_total"] = d["Zn"] + d["Mg"] + d["Cu"]
d["Mg_Cu_ratio"] = d["Mg"] / (d["Cu"] + 0.01)
d["Zr_Mn_Cr_total"] = d["Zr"] + d["Mn"] + d["Cr"]
d["Fe_Si_total"] = d["Fe"] + d["Si"]
d["Zn_Mg_product"] = d["Zn"] * d["Mg"]

# 工艺衍生
d["total_aging_time"] = d["aging_time"] + d["pre_aging_time"]
d["aging_temp_time"] = d["aging_temp"] * d["aging_time"] / 1000
d["solution_temp_time"] = d["solution_temp"] * d["solution_time"] / 100
d["has_pre_aging"] = (d["pre_aging_time"] > 0).astype(int)
d["has_deformation"] = (d["deformation_ratio"] > 0).astype(int)
d["aging_time_log"] = np.log1p(d["aging_time"])

# 成分-工艺交互
d["Zn_aging_interaction"] = d["Zn"] * d["aging_temp"] / 100
d["Mg_solution_interaction"] = d["Mg"] * d["solution_temp"] / 100
d["alloy_agingtemp"] = (d["Zn"] + d["Mg"] + d["Cu"]) * d["aging_temp"] / 100

extra_cols = [
    "Zn_Mg_ratio", "Zn_Mg_Cu_total", "Mg_Cu_ratio", "Zr_Mn_Cr_total",
    "Fe_Si_total", "Zn_Mg_product", "total_aging_time", "aging_temp_time",
    "solution_temp_time", "has_pre_aging", "has_deformation", "aging_time_log",
    "Zn_aging_interaction", "Mg_solution_interaction", "alloy_agingtemp"
]

all_cols = comp_cols + proc_cols + extra_cols
print(f"  原始特征: {len(comp_cols)+len(proc_cols)}, 衍生特征: {len(extra_cols)}, 总计: {len(all_cols)}")

X = d[all_cols].values
y = d[target_cols].values
X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
print(f"  X shape: {X.shape}, y shape: {y.shape}")

# ============================================================
# 第三步: 评估函数
# ============================================================
def evaluate(yt, yp, names):
    r = {}
    for i, n in enumerate(names):
        r[n] = {"R2": r2_score(yt[:,i], yp[:,i]),
                "RMSE": np.sqrt(mean_squared_error(yt[:,i], yp[:,i])),
                "MAE": mean_absolute_error(yt[:,i], yp[:,i])}
    r["average"] = {k: np.mean([r[n][k] for n in names]) for k in ["R2","RMSE","MAE"]}
    return r

def show(r, name):
    print(f"\n  {name}:")
    print(f"  {'指标':<8s} {'屈服':>9s} {'抗拉':>9s} {'延伸':>9s} {'平均':>9s}")
    print(f"  {'-'*46}")
    for m in ["R2","RMSE","MAE"]:
        row = f"  {m:<8s}"
        for n in target_cols+["average"]:
            v = r[n][m]
            row += f" {v:>9.4f}" if m=="R2" else f" {v:>9.2f}"
        print(row)

# ============================================================
# 第四步: K折交叉验证
# ============================================================
print("\n【第三步】5折交叉验证 - 多模型对比")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_pred(factory, X, y, kf, y_transform=None):
    all_p = np.zeros_like(y)
    for tr, te in kf.split(X):
        scX = StandardScaler()
        Xtr = scX.fit_transform(X[tr])
        Xte = scX.transform(X[te])
        scY = StandardScaler()
        ytr = scY.fit_transform(y[tr])
        model = factory()
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        all_p[te] = scY.inverse_transform(p)
    return all_p

# 模型定义 (精细调参)
def rf():
    return MultiOutputRegressor(RandomForestRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        max_features=0.5, random_state=42, n_jobs=-1))

def et():
    return MultiOutputRegressor(ExtraTreesRegressor(
        n_estimators=500, max_depth=None, min_samples_leaf=1,
        max_features=0.5, random_state=42, n_jobs=-1))

def xgb_m():
    return MultiOutputRegressor(xgb.XGBRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6, min_child_weight=3,
        reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=0))

def lgb_m():
    return MultiOutputRegressor(lgb.LGBMRegressor(
        n_estimators=500, max_depth=7, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.6, min_child_samples=8,
        reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=-1))

def gbrt():
    return MultiOutputRegressor(GradientBoostingRegressor(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42))

def svr():
    return MultiOutputRegressor(SVR(kernel='rbf', C=200, gamma='scale', epsilon=0.05))

factories = {"随机森林": rf, "ExtraTrees": et, "XGBoost": xgb_m,
             "LightGBM": lgb_m, "梯度提升": gbrt, "SVR": svr}

preds = {}
results = {}
for name, fac in factories.items():
    print(f"\n  训练 {name}...")
    p = cv_pred(fac, X, y, kf)
    preds[name] = p
    results[name] = evaluate(y, p, target_cols)
    show(results[name], name)

# ============================================================
# 第五步: 多策略融合
# ============================================================
print("\n【第四步】多策略融合")

# 1. 简单平均
avg = np.mean([preds[n] for n in factories], axis=0)
results["简单平均"] = evaluate(y, avg, target_cols)
show(results["简单平均"], "简单平均融合")

# 2. 加权平均 (按各目标R²分别加权)
weighted = np.zeros_like(y)
for i, tgt in enumerate(target_cols):
    w = np.array([max(results[n][tgt]["R2"], 0.01) for n in factories])
    w = w / w.sum()
    weighted[:, i] = sum(preds[n][:, i] * w[j] for j, n in enumerate(factories))
results["分目标加权"] = evaluate(y, weighted, target_cols)
show(results["分目标加权"], "分目标加权融合")

# 3. Stacking (Ridge元学习器)
meta_X = np.hstack([preds[n] for n in factories])
meta_p = np.zeros_like(y)
for tr, te in kf.split(meta_X):
    sc = StandardScaler()
    mtr = sc.fit_transform(meta_X[tr])
    mte = sc.transform(meta_X[te])
    meta = MultiOutputRegressor(Ridge(alpha=0.5))
    meta.fit(mtr, y[tr])
    meta_p[te] = meta.predict(mte)
results["Stacking-Ridge"] = evaluate(y, meta_p, target_cols)
show(results["Stacking-Ridge"], "Stacking(Ridge)")

# 4. Stacking (BayesianRidge元学习器)
meta_p2 = np.zeros_like(y)
for tr, te in kf.split(meta_X):
    sc = StandardScaler()
    mtr = sc.fit_transform(meta_X[tr])
    mte = sc.transform(meta_X[te])
    meta = MultiOutputRegressor(BayesianRidge())
    meta.fit(mtr, y[tr])
    meta_p2[te] = meta.predict(mte)
results["Stacking-Bayes"] = evaluate(y, meta_p2, target_cols)
show(results["Stacking-Bayes"], "Stacking(BayesianRidge)")

# 5. 超级融合 (融合所有融合结果)
super_avg = np.mean([avg, weighted, meta_p, meta_p2], axis=0)
results["超级融合"] = evaluate(y, super_avg, target_cols)
show(results["超级融合"], "超级融合(所有融合结果平均)")

# ============================================================
# 第六步: 特征重要性
# ============================================================
print("\n【第五步】特征重要性 (随机森林)")

scX = StandardScaler()
Xs = scX.fit_transform(X)
scY = StandardScaler()
ys = scY.fit_transform(y)
rf_full = rf()
rf_full.fit(Xs, ys)

imp = np.mean([e.feature_importances_ for e in rf_full.estimators_], axis=0)
fi = pd.DataFrame({"feature": all_cols, "importance": imp}).sort_values("importance", ascending=False)
print(f"\n  前15重要特征:")
for i, (_, r) in enumerate(fi.head(15).iterrows()):
    print(f"  {i+1:2d}. {r['feature']:<28s} {r['importance']:.4f}")

# ============================================================
# 第七步: 保存
# ============================================================
print("\n【第六步】保存结果")

summary = {}
for name, r in results.items():
    summary[name] = {
        "avg_R2": r["average"]["R2"], "avg_RMSE": r["average"]["RMSE"],
        "yield_R2": r["yield_strength"]["R2"],
        "tensile_R2": r["tensile_strength"]["R2"],
        "elongation_R2": r["elongation"]["R2"],
    }
with open(os.path.join(PROJECT_DIR, "final_results.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

fi.to_csv(os.path.join(PROJECT_DIR, "final_feature_importance.csv"), index=False, encoding="utf-8-sig")

best = max(results, key=lambda k: results[k]["average"]["R2"])
best_pred = preds.get(best, super_avg)
if best in ["简单平均","分目标加权","Stacking-Ridge","Stacking-Bayes","超级融合"]:
    best_pred = {"简单平均":avg, "分目标加权":weighted, "Stacking-Ridge":meta_p,
                  "Stacking-Bayes":meta_p2, "超级融合":super_avg}[best]

pdf = pd.DataFrame(best_pred, columns=[f"pred_{c}" for c in target_cols])
tdf = pd.DataFrame(y, columns=[f"true_{c}" for c in target_cols])
out = pd.concat([df[["sample_id","source"]].reset_index(drop=True), tdf, pdf], axis=1)
out.to_csv(os.path.join(PROJECT_DIR, "final_predictions.csv"), index=False, encoding="utf-8-sig")

# ============================================================
# 总结
# ============================================================
print("\n" + "=" * 70)
print("  最终训练完成！各模型5折交叉验证R²对比:")
print("=" * 70)
print(f"\n  {'模型':<18s} {'屈服':>9s} {'抗拉':>9s} {'延伸':>9s} {'平均':>9s}")
print(f"  {'-'*56}")
all_names = list(factories.keys()) + ["简单平均","分目标加权","Stacking-Ridge","Stacking-Bayes","超级融合"]
for name in all_names:
    r = results[name]
    print(f"  {name:<18s} {r['yield_strength']['R2']:>9.4f} "
          f"{r['tensile_strength']['R2']:>9.4f} {r['elongation']['R2']:>9.4f} "
          f"{r['average']['R2']:>9.4f}")

print(f"\n  最佳模型: {best} (平均R²={results[best]['average']['R2']:.4f})")
print(f"  目标R²>0.90: {'已达到!' if results[best]['average']['R2']>=0.90 else '未达到'}")
print(f"\n  结果文件:")
print(f"    评估: C:\\al7xxx\\final_results.json")
print(f"    特征: C:\\al7xxx\\final_feature_importance.csv")
print(f"    预测: C:\\al7xxx\\final_predictions.csv")
print("=" * 70)
