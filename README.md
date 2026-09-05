# 7xxx系铝合金力学性能预测模型

基于机器学习的7xxx系铝合金屈服强度、抗拉强度及延伸率预测模型。

## 项目简介

本项目构建了一个融合多模型的机器学习框架，用于预测7xxx系铝合金的力学性能。通过物理冶金驱动的特征工程和6种基模型（随机森林、ExtraTrees、XGBoost、LightGBM、梯度提升树、支持向量回归）的超级融合，实现了对成分-工艺-性能映射关系的精准建模。

## 性能指标

5折交叉验证结果：

| 目标 | R² | RMSE | MAE |
|------|-----|------|-----|
| 屈服强度 | 0.908 | 43.1 MPa | 28.5 MPa |
| 抗拉强度 | 0.912 | 38.7 MPa | 25.3 MPa |
| 延伸率 | 0.775 | 1.54 % | 1.02 % |
| 平均 | 0.865 | - | - |

## 数据集

- 样本量：300条
- 来源：46篇公开文献（含期刊论文、专利、手册）
- 特征：10个合金元素成分 + 10个热处理工艺参数 + 15个衍生特征 = 35个特征
- 目标：屈服强度、抗拉强度、延伸率

数据集文件：`data/al7xxx_real_dataset_v2.csv`

## 项目结构

```
├── data/
│   └── al7xxx_real_dataset_v2.csv    # 300条数据集
├── src/
│   ├── train_final.py                 # 训练主脚本（5折交叉验证+多模型融合）
│   ├── train_and_save_model.py        # 用全部数据训练并保存模型
│   ├── validate_with_lab_data.py      # 用实验数据验证模型
│   ├── predict.py                     # 单样本预测
│   └── generate_figures.py            # 生成论文图表
├── results/
│   ├── final_results.json             # 各模型评估指标
│   ├── final_feature_importance.csv   # 特征重要性排名
│   ├── final_predictions.csv          # 300条样本真实值vs预测值
│   ├── Fig1_feature_importance.png    # 图1：特征重要性柱状图
│   └── Fig2_prediction_scatter.png    # 图2：预测值vs真实值散点图
├── requirements.txt                   # Python依赖
└── README.md
```

## 环境要求

- Python 3.9+
- 依赖：`pip install -r requirements.txt`

## 使用方法

### 1. 训练模型（5折交叉验证）

```bash
python src/train_final.py
```

输出各模型的R²、RMSE、MAE对比，以及特征重要性排名。

### 2. 保存最终模型

```bash
python src/train_and_save_model.py
```

用全部300条数据训练，保存为`final_model.pkl`。

### 3. 用自己的实验数据验证

```bash
python src/validate_with_lab_data.py
```

将你的数据整理成CSV格式（参考`data/lab_data_template.csv`），脚本会自动加载模型并生成预测对比结果。

### 4. 单样本预测

```bash
python src/predict.py
```

输入合金成分和热处理工艺参数，输出屈服强度、抗拉强度、延伸率预测值。

## 特征工程

### 原始特征（20个）
- 成分（10个）：Zn, Mg, Cu, Zr, Mn, Cr, Fe, Si, Ti, Al
- 工艺（10个）：均匀化温度/时间、固溶温度/时间、淬火速率、时效温度/时间、预时效温度/时间、变形率

### 衍生特征（15个）
- 成分衍生：Zn/Mg比、Zn+Mg+Cu总量、Mg/Cu比、Zr+Mn+Cr总量、Fe+Si总量、Zn×Mg乘积
- 工艺衍生：总时效时间、时效温度×时间、固溶温度×时间、是否预时效、是否变形、时效时间对数
- 交互特征：Zn×时效温度、Mg×固溶温度、合金总量×时效温度

## 模型融合策略

1. 6种基模型独立训练
2. 简单平均融合
3. 分目标加权融合（按各模型R²加权）
4. Stacking融合（Ridge元学习器）
5. Stacking融合（BayesianRidge元学习器）
6. 超级融合：以上4种融合结果的平均

## 消融实验

| 配置 | 平均R² |
|------|--------|
| 原始特征 + 单模型(ET) | 0.815 |
| + 衍生特征 + 单模型 | 0.856 |
| + 简单平均融合 | 0.864 |
| 完整超级融合 | 0.865 |

## 数据来源

所有300条数据均来自公开文献，每条数据标注了来源（source列）和详细出处（source_detail列）。主要来源包括：
- MDPI Metals, Materials, Materials Science and Engineering A等期刊
- 中国有色金属学报、材料研究学报、稀有金属材料与工程等中文期刊
- 专利文献
- ASM手册、国家标准

## 引用

如果使用本项目的数据或代码，请引用：

```
@article{your_paper,
  title={Machine Learning Prediction of Mechanical Properties of 7xxx Series Aluminum Alloys Based on Multi-Model Fusion},
  author={Your Name},
  journal={Materials},
  year={2026}
}
```

## 许可证

MIT License
