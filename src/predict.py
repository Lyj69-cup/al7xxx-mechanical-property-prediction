"""
预测脚本
加载训练好的模型，对新的铝合金成分+热处理工艺参数进行力学性能预测。

使用方式:
  from predict import Predictor
  predictor = Predictor("checkpoints/best_model.pt")
  result = predictor.predict_single(composition_dict, process_dict)
  print(result)
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Union

from .config import Config, ModelConfig, DataConfig
from .models.fusion_model import FusionModel
from .data.dataset import DataPreprocessor


class Predictor:
    """力学性能预测器"""

    def __init__(self, checkpoint_path: str,
                 preprocessor_path: Optional[str] = None,
                 device: str = "auto"):
        """
        Args:
            checkpoint_path: 模型检查点路径 (.pt)
            preprocessor_path: 预处理器保存路径 (.pkl)，为None则从checkpoint同目录查找
            device: 设备
        """
        # 加载检查点
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device,
                                 weights_only=False)
        self.config = checkpoint.get("config", Config())
        # 兼容完整Config和仅ModelConfig两种情况
        if hasattr(self.config, 'model'):
            model_cfg = self.config.model
            data_cfg = self.config.data
        else:
            model_cfg = self.config if hasattr(self.config, 'comp_dim') else ModelConfig()
            data_cfg = DataConfig()

        # 构建模型
        self.model = FusionModel(model_cfg)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()

        # 加载预处理器
        if preprocessor_path is None:
            preprocessor_path = os.path.join(
                os.path.dirname(checkpoint_path), "preprocessor.pkl")
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, "rb") as f:
                self.preprocessor = pickle.load(f)
            print(f"[Predictor] 预处理器已加载: {preprocessor_path}")
        else:
            print(f"[Predictor] 警告: 未找到预处理器 {preprocessor_path}，"
                  f"将使用配置中的列名，但归一化参数缺失。")
            self.preprocessor = DataPreprocessor(
                data_cfg.composition_features, data_cfg.process_features,
                data_cfg.target_features)

        self.comp_cols = self.preprocessor.comp_cols
        self.proc_cols = self.preprocessor.proc_cols
        self.target_cols = self.preprocessor.target_cols
        print(f"[Predictor] 模型已加载: {checkpoint_path}")
        print(f"[Predictor] 成分特征({len(self.comp_cols)}): {self.comp_cols}")
        print(f"[Predictor] 工艺特征({len(self.proc_cols)}): {self.proc_cols}")
        print(f"[Predictor] 预测目标({len(self.target_cols)}): {self.target_cols}")

    def _prepare_input(self, composition: Union[Dict, pd.Series, np.ndarray],
                       process: Union[Dict, pd.Series, np.ndarray]
                       ) -> tuple:
        """将输入转为归一化后的numpy数组"""
        # 成分
        if isinstance(composition, dict):
            comp_arr = np.array([composition.get(c, 0.0) for c in self.comp_cols],
                                dtype=np.float32)
        elif isinstance(composition, pd.Series):
            comp_arr = composition.reindex(self.comp_cols).fillna(0.0).values.astype(np.float32)
        else:
            comp_arr = np.asarray(composition, dtype=np.float32).flatten()

        # 工艺
        if isinstance(process, dict):
            proc_arr = np.array([process.get(c, 0.0) for c in self.proc_cols],
                                dtype=np.float32)
        elif isinstance(process, pd.Series):
            proc_arr = process.reindex(self.proc_cols).fillna(0.0).values.astype(np.float32)
        else:
            proc_arr = np.asarray(process, dtype=np.float32).flatten()

        # 归一化（构造单行DataFrame复用preprocessor，目标列填占位0）
        df = pd.DataFrame([comp_arr.tolist() + proc_arr.tolist() + [0.0] * len(self.target_cols)],
                          columns=self.comp_cols + self.proc_cols + self.target_cols)
        comp_n, proc_n, _ = self.preprocessor.transform(df)
        return comp_n[0], proc_n[0]

    @torch.no_grad()
    def predict_single(self, composition: Union[Dict, pd.Series, np.ndarray],
                       process: Union[Dict, pd.Series, np.ndarray],
                       return_normalized: bool = False) -> Dict[str, float]:
        """
        单样本预测

        Args:
            composition: 合金成分（dict/Series/array），键为元素名如 Zn, Mg, Cu...
            process: 热处理工艺参数（dict/Series/array）
            return_normalized: 是否返回归一化值

        Returns:
            预测结果字典，如 {"yield_strength": 520.3, "tensile_strength": 580.1, "elongation": 12.5}
        """
        comp_n, proc_n = self._prepare_input(composition, process)
        comp_t = torch.from_numpy(comp_n).unsqueeze(0).to(self.device)
        proc_t = torch.from_numpy(proc_n).unsqueeze(0).unsqueeze(-1).to(self.device)

        pred_norm = self.model(comp_t, proc_t).cpu().numpy()[0]

        if return_normalized:
            return {name: float(v) for name, v in zip(self.target_cols, pred_norm)}

        # 反归一化
        pred_original = self.preprocessor.inverse_transform_target(
            pred_norm.reshape(1, -1))[0]
        return {name: float(v) for name, v in zip(self.target_cols, pred_original)}

    @torch.no_grad()
    def predict_batch(self, compositions: List[Union[Dict, np.ndarray]],
                      processes: List[Union[Dict, np.ndarray]],
                      batch_size: int = 64) -> pd.DataFrame:
        """
        批量预测

        Args:
            compositions: 成分列表
            processes: 工艺列表
            batch_size: 批大小

        Returns:
            DataFrame，包含预测的三个性能指标
        """
        assert len(compositions) == len(processes)
        all_preds = []
        for i in range(0, len(compositions), batch_size):
            batch_comp = []
            batch_proc = []
            for comp, proc in zip(compositions[i:i+batch_size],
                                   processes[i:i+batch_size]):
                c_n, p_n = self._prepare_input(comp, proc)
                batch_comp.append(c_n)
                batch_proc.append(p_n)
            comp_t = torch.from_numpy(np.stack(batch_comp)).to(self.device)
            proc_t = torch.from_numpy(np.stack(batch_proc)).unsqueeze(-1).to(self.device)
            preds = self.model(comp_t, proc_t).cpu().numpy()
            all_preds.append(preds)

        all_preds = np.concatenate(all_preds, axis=0)
        pred_original = self.preprocessor.inverse_transform_target(all_preds)
        return pd.DataFrame(pred_original, columns=self.target_cols)

    def predict_from_csv(self, csv_path: str,
                         output_path: Optional[str] = None) -> pd.DataFrame:
        """
        从CSV文件批量预测

        Args:
            csv_path: 输入CSV路径，需包含成分和工艺列
            output_path: 输出CSV路径，为None则不保存

        Returns:
            包含预测结果的DataFrame
        """
        df = pd.read_csv(csv_path)
        compositions = df[self.comp_cols].to_dict("records")
        processes = df[self.proc_cols].to_dict("records")
        preds = self.predict_batch(compositions, processes)

        result = pd.concat([df.reset_index(drop=True), preds], axis=1)
        if output_path:
            result.to_csv(output_path, index=False, encoding="utf-8-sig")
            print(f"预测结果已保存: {output_path}")
        return result
