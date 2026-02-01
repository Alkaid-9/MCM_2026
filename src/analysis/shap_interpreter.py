# ==============================================================================
# src/analysis/shap_interpreter.py
# Role: Explainable AI (XAI) Engine (v6.7 - Init Logic Fix)
# Function: Latent Preference Attribution via Weighted SHAP.
# Fix: Resolved AttributeError by reordering __init__ logic.
# Standard: Top-Tier Econometrics / Explainable AI (XAI).
# ==============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import re
from typing import Tuple, Any

# 引入项目统一绘图引擎
from src.utils.plotting import DWTSPlotter


class ShapInterpreter:
    """
    SHAP 解释引擎：
    量化非线性特征对粉丝投票份额的边际贡献。
    采用 KernelExplainer + Wrapper 模式，确保兼容性与鲁棒性。
    """

    def __init__(self, df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("SHAP_INTERPRETER")
        self.fig_dir = fig_dir
        self.plotter = DWTSPlotter(output_dir=fig_dir)

        # 1. [修复] 先定位技术基准列 (Metadata Inspection)
        # 必须在数据清洗前确定使用哪一列作为 Judge Score，否则清洗函数会报错
        self.tech_col = 'week_avg_score'
        for col in ['score_z', 'week_z_sum']:
            if col in df_platinum.columns:
                self.tech_col = col
                break

        # 2. [修复] 再执行数据清洗 (Data Sanitization)
        # 此时 self.tech_col 已定义，_sanitize_data 可以安全调用
        self.df = self._sanitize_data(df_platinum)

        self.X = None
        self.y_fan = None
        self.y_judge = None
        self.weights = None

        os.makedirs(self.fig_dir, exist_ok=True)

    def _sanitize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据消毒：移除可能导致序列化失败的特殊字符"""
        df_clean = df.copy()
        # 目标列必须是数值
        # 这里使用了 self.tech_col，所以必须保证它在调用前已初始化
        target_cols = ['est_fan_vote_mu', 'est_fan_vote_sigma', self.tech_col]

        for col in target_cols:
            if col in df_clean.columns:
                # 强制转数值，无法转换的变为 0
                # 处理类似 '[0.5]' 这种可能残留的字符串格式
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].astype(str).str.replace(r'[\[\]\s]', '', regex=True)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0)

        return df_clean

    def _prepare_data_matrix(self):
        """构造回归设计矩阵"""
        # 提取特征 (对齐 FeatureFactory)
        num_feats = ['celebrity_age_during_season', 'partner_alpha', 'score_delta']

        # 构造数值特征
        X_num = pd.DataFrame()
        for f in num_feats:
            if f in self.df.columns:
                X_num[f] = self.df[f].fillna(self.df[f].median())
            else:
                X_num[f] = 0.0

        # 构造类别特征 (One-Hot)
        if 'celebrity_industry' in self.df.columns:
            ind_clean = self.df['celebrity_industry'].fillna('Other')
            X_cat = pd.get_dummies(ind_clean, prefix='ind', dtype=int)
        else:
            X_cat = pd.DataFrame()

        # 合并
        X = pd.concat([X_num, X_cat], axis=1)

        # 清除 XGBoost 禁止的列名特殊符号 (<, [, ])
        X.columns = [re.sub(r'[\[\]<>]', '', c) for c in X.columns]

        self.X = X
        self.y_fan = self.df['est_fan_vote_mu']
        self.y_judge = self.df[self.tech_col]

        # 逆方差加权 (WLS Logic): 不确定性越大的样本，权重越低
        # 物理意义：Task 1 反演结果越确定的样本，在归因时话语权越大
        sigma = self.df['est_fan_vote_sigma'].replace(0, 1.0)
        self.weights = 1.0 / (sigma ** 2 + 1e-6)
        self.weights /= self.weights.mean()  # 归一化权重

    def run_dual_shap_analysis(self) -> Tuple[Any, Any, Any]:
        """
        [主算法] 执行双路 SHAP 因果审计。
        使用 Wrapper 模式解决 SHAP 库与 XGBoost 的版本兼容性问题。
        """
        self.logger.info(">>> 启动双路 SHAP 因果解析 (Wrapper Patch Applied)...")

        try:
            self._prepare_data_matrix()

            # 1. 准备数据矩阵 (Pure NumPy 模式，断开 DataFrame 元数据关联)
            X_arr = self.X.values.astype(np.float64)
            y_fan_arr = self.y_fan.values
            y_judge_arr = self.y_judge.values

            # 2. 训练代理模型 (Surrogate Models)
            # 粉丝模型 (加权回归)
            model_f = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                n_jobs=4,
                random_state=2026
            )
            model_f.fit(self.X, y_fan_arr, sample_weight=self.weights)

            # 评委模型
            model_j = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                n_jobs=4,
                random_state=2026
            )
            model_j.fit(self.X, y_judge_arr)

            # --- [关键魔法] 预测函数包装器 ---
            # SHAP 库在检测到原生 XGB 模型时会尝试读取 feature_names_in_，导致版本冲突。
            # 传入一个纯 Python 函数可以绕过此检查。
            def f_wrapper(x):
                return model_f.predict(x)

            def j_wrapper(x):
                return model_j.predict(x)

            # 3. 初始化 KernelExplainer
            self.logger.info("初始化 SHAP Kernel (使用背景聚类以加速)...")
            # 使用 K-means 聚类减少背景样本量，提高计算速度
            if len(X_arr) > 20:
                background = shap.kmeans(X_arr, 20).data
            else:
                background = X_arr

            # 传入包装函数而非模型对象
            explainer_f = shap.KernelExplainer(f_wrapper, background)
            explainer_j = shap.KernelExplainer(j_wrapper, background)

            # 4. 计算 SHAP 值 (仅计算前 100 个样本作为代表，保证 MCM 竞赛效率)
            limit = min(100, len(X_arr))
            X_test_arr = X_arr[:limit]
            X_test_df = self.X.iloc[:limit]  # 保留列名用于绘图

            self.logger.info(f"正在计算 SHAP Values (N={limit})...")
            # silent=True 防止进度条刷屏
            shap_values_f = explainer_f.shap_values(X_test_arr, silent=True)
            shap_values_j = explainer_j.shap_values(X_test_arr, silent=True)

            return X_test_df, shap_values_j, shap_values_f

        except Exception as e:
            self.logger.error(f"SHAP 内核计算中断: {e}", exc_info=True)
            # 返回空值但不中断流水线 (Fail-Soft)
            return self.X, None, None

    def plot_global_importance(self, shap_values):
        """
        绘制全局特征贡献图 (Beeswarm Proxy)。
        """
        if shap_values is None or self.X is None: return

        try:
            plt.figure(figsize=(10, 6))
            # 为了兼容性，使用 summary_plot 绘制
            # 注意：shap_values 此时是一个 numpy array
            # 我们需要传入对应的特征矩阵(DataFrame)来显示列名
            limit = len(shap_values)
            shap.summary_plot(shap_values, self.X.iloc[:limit], show=False, max_display=12)

            plt.title("Latent Determinants of Fan Preference (Non-linear SHAP)", fontsize=14)
            plt.tight_layout()

            self.plotter.save_figure("task3_shap_global_beeswarm.png")
            self.logger.info("SHAP 全局重要性图已生成。")
        except Exception as e:
            self.logger.warning(f"SHAP 绘图失败: {e}")

    def plot_age_dependence_contrast(self, shap_j, shap_f):
        """
        【学术杀手锏】对比年龄(Age)在评委侧和观众侧的异质性影响曲线。
        """
        if shap_j is None or shap_f is None: return
        if 'celebrity_age_during_season' not in self.X.columns: return

        try:
            age_col = 'celebrity_age_during_season'
            age_idx = self.X.columns.get_loc(age_col)
            limit = len(shap_j)
            age_values = self.X[age_col].iloc[:limit].values

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Judge Plot
            ax1.scatter(age_values, shap_j[:, age_idx], color='#ff7f0e', alpha=0.6)
            # 添加趋势线
            sns.regplot(x=age_values, y=shap_j[:, age_idx], scatter=False, lowess=True,
                        ax=ax1, color='black', line_kws={'linewidth': 1.5})
            ax1.set_title("Judge Sensitivity to Age (Technical Merit)")
            ax1.set_xlabel("Celebrity Age")
            ax1.set_ylabel("SHAP Value (Score Impact)")
            ax1.grid(True, alpha=0.3)

            # Fan Plot
            ax2.scatter(age_values, shap_f[:, age_idx], color='#1f77b4', alpha=0.6)
            sns.regplot(x=age_values, y=shap_f[:, age_idx], scatter=False, lowess=True,
                        ax=ax2, color='black', line_kws={'linewidth': 1.5})
            ax2.set_title("Fan Sensitivity to Age (Popularity)")
            ax2.set_xlabel("Celebrity Age")
            ax2.set_ylabel("SHAP Value (Vote Impact)")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            self.plotter.save_figure("task3_age_heterogeneity_dependence.png")
            self.logger.info("年龄异质性曲线已生成。")
        except Exception as e:
            self.logger.warning(f"年龄依赖图绘制失败: {e}")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("ShapInterpreter module loaded. Ready for integration.")