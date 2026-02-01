# ==============================================================================
# src/analysis/shap_interpreter.py
# Role: Explainable AI (XAI) Engine (v6.5 - Platinum Robustness)
# Function: Latent Preference Attribution with 1/Var Weighting.
# Fix: Resolved 'y' attribute error & established high-precision data contract.
# ==============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import logging
import os
import re

# 引入项目统一绘图引擎
from src.utils.plotting import DWTSPlotter


class ShapInterpreter:
    """
    SHAP 解释引擎：
    量化非线性特征对粉丝投票份额的边际贡献。
    """

    def __init__(self, df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("SHAP_INTERPRETER")
        self.fig_dir = fig_dir
        self.plotter = DWTSPlotter(output_dir=fig_dir)

        # 1. 物理级数据清洗 (防御性编程的核心)
        # 强制处理类似 '[1.15E-1]' 的异常字符串
        self.df = self._sanitize_data(df_platinum)

        # 2. 自动定位技术基准列
        self.tech_col = 'week_avg_score'
        for col in ['score_z', 'week_z_sum']:
            if col in self.df.columns:
                self.tech_col = col
                break

        # 3. 【关键点】：统一变量命名契约
        self.X, self.y_fan, self.y_judge, self.weights = self._prepare_data_matrix()

        os.makedirs(self.fig_dir, exist_ok=True)

    def _sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """解决 XGBoost 序列化导致的 string-to-float 报错"""
        df_clean = df.copy()
        target_cols = ['est_fan_vote_mu', 'est_fan_vote_sigma', 'score_z', 'week_avg_score', 'week_z_sum']
        for col in target_cols:
            if col in df_clean.columns:
                # 移除 [], (), 多余空格
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[\[\]\s]', '', regex=True)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)
        return df_clean

    def _prepare_data_matrix(self):
        """构造回归设计矩阵"""
        # 提取特征 (对齐 FeatureFactory)
        num_feats = ['celebrity_age_during_season', 'partner_alpha', 'score_delta']
        for f in num_feats:
            if f not in self.df.columns: self.df[f] = 0.0

        X_cat = pd.get_dummies(self.df[['celebrity_industry']].fillna('Other'), prefix='ind', dtype=int)
        X = pd.concat([self.df[num_feats], X_cat], axis=1)

        # 清除 XGBoost 禁止的列名特殊符号
        X.columns = [re.sub(r'[\[\]<>]', '', c) for c in X.columns]

        # 统一目标变量命名 (供 run_dual_shap_analysis 使用)
        y_fan = self.df['est_fan_vote_mu']
        y_judge = self.df[self.tech_col]

        # 逆方差加权 (WLS Logic)
        sigma = self.df['est_fan_vote_sigma'].replace(0, 1.0)
        weights = 1.0 / (sigma ** 2 + 1e-6)
        weights /= weights.mean()

        return X, y_fan, y_judge, weights

    def run_dual_shap_analysis(self):
        """
        [主算法] 执行双路 SHAP 因果审计。
        """
        self.logger.info(">>> 启动双路 SHAP 因果解析...")

        try:
            # 1. 粉丝偏好解释 (Target: y_fan)
            # 设置极简配置，防止底层解析器崩掉
            model_f = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                base_score=float(self.y_fan.mean())
            )
            model_f.fit(self.X, self.y_fan, sample_weight=self.weights)

            # 使用较慢但最稳健的 KernelExplainer 绕过 XGBoost 底层 JSON 解析 Bug
            self.logger.info("使用 Kernel 降级模式，执行 23 核并行 SHAP 估算...")
            background = shap.sample(self.X, 20)  # 压缩背景集以加速
            explainer_f = shap.KernelExplainer(model_f.predict, background)

            # 仅解释前 100 个样本以获取全局特征分布 (MCM 效率优先)
            X_test = self.X.head(100)
            shap_values_f = explainer_f.shap_values(X_test)

            # 2. 评委准则解释 (Target: y_judge)
            model_j = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
            model_j.fit(self.X, self.y_judge)
            explainer_j = shap.KernelExplainer(model_j.predict, background)
            shap_values_j = explainer_j.shap_values(X_test)

            return X_test, shap_values_j, shap_values_f

        except Exception as e:
            self.logger.error(f"SHAP 内核计算中断: {e}", exc_info=True)
            return self.X, None, None

    def plot_global_importance(self, shap_f):
        """绘制全局特征贡献图"""
        if shap_f is None: return
        plt.figure(figsize=(10, 6))
        # 确保传入的是数组格式
        shap.summary_plot(shap_f, self.X.head(len(shap_f)), show=False)
        plt.title("Key Latent Determinants of Fan Vote (SHAP)", fontsize=14)
        self.plotter.save_figure("task3_shap_global_beeswarm.png")