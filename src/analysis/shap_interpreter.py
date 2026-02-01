# ==============================================================================
# src/analysis/shap_interpreter.py
# Role: Explainable AI (XAI) Engine - Task 3 Non-linear Analysis
# Function: Computing SHAP values for both Judge and Fan paths to find non-linearities.
# Physics: Decomposing the "Black Box" of success into marginal contributions (Shapley Values).
# Standard: High-DPI Visualization / Industrial Robustness / Academic Color Standards.
# ==============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import logging
import os
import re
import warnings

# 引入项目统一绘图引擎
from src.utils.plotting import DWTSPlotter


class ShapInterpreter:
    """
    SHAP 解释引擎：
    利用博弈论沙普利值 (Shapley Values) 量化每个特征对得分的边际贡献。

    [学术价值]:
    本模块旨在揭示线性模型无法捕捉的非线性阈值效应。
    例如：在评委侧，年龄可能存在一个“体能临界点”；在观众侧，年龄可能呈现“养成系”与“情怀系”的双峰分布。
    """

    def __init__(self, df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("SHAP_INTERPRETER")
        # 仅针对有效推断样本进行 XAI 分析
        self.df = df_platinum.dropna(subset=['est_fan_vote_mu', 'week_avg_score']).copy()
        self.fig_dir = fig_dir
        # 实例化绘图器，确保色盘统一
        self.plotter = DWTSPlotter(output_dir=fig_dir)
        os.makedirs(self.fig_dir, exist_ok=True)

    def _robust_clean(self, series: pd.Series) -> pd.Series:
        """
        【极致增强版防火墙】
        物理意义：专门处理类似 '[7.838039E0]' 这种被误识别为 object 的科学计数法碎片。
        """

        def force_float(x):
            if isinstance(x, (int, float)):
                return float(x)
            if x is None or pd.isna(x):
                return np.nan

            s = str(x)
            # 正则提取：匹配可选负号 + 数字 + 可选点 + 尾随数字 + 可选科学计数法后缀
            match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    return np.nan
            return np.nan

        return series.apply(force_float)

    def _prepare_feature_matrix(self):
        """
        构造回归特征矩阵 (X) 并进行清洗。
        """
        self.logger.info("构造并清洗特征矩阵...")

        # 核心因素：年龄、舞伴能力、表现进步度、赛程进度
        num_feats = ['celebrity_age_during_season', 'partner_alpha', 'score_delta', 'week_num']

        for col in num_feats:
            if col in self.df.columns:
                self.df[col] = self._robust_clean(self.df[col])
                # 使用中位数填充缺失，确保 XGBoost 矩阵对齐
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
            else:
                self.df[col] = 0.0

        X_num = self.df[num_feats]

        # 处理类别型因素 (行业背景)
        if 'industry_group' not in self.df.columns:
            # 兼容性处理：如果上游未分组，则寻找原始行业列
            if 'celebrity_industry' in self.df.columns:
                self.df['industry_group'] = self.df['celebrity_industry']
            else:
                self.df['industry_group'] = 'Unknown'

        X_cat = pd.get_dummies(self.df['industry_group'], prefix='ind', dtype=int)

        # 拼接全特征空间
        X = pd.concat([X_num, X_cat], axis=1)

        # [关键工程点] 清洗列名，SHAP 与 XGBoost 不支持 [, ] 或 < 符号
        clean_cols = [re.sub(r'[\[\]<]', '', str(col)) for col in X.columns]
        X.columns = clean_cols

        return X

    def run_dual_shap_analysis(self):
        """
        核心逻辑：计算评委路径和粉丝路径的 SHAP 值。
        [深度重构]: 采用置换解释器 (Permutation Explainer) 方案。
        物理意义：通过“黑盒扰动”绕过 XGBoost 3.x 序列化 Bug，确保学术归因的绝对鲁棒性。
        """
        self.logger.info(">>> 启动双路 SHAP 非线性归因 (Permutation-Stable Mode)...")

        try:
            # 1. 因子矩阵预处理
            X_df = self._prepare_feature_matrix()
            if X_df.empty or len(X_df) < 30:
                self.logger.warning("有效样本不足，跳过归因环节。")
                return None, None, None

            # 强制压实数据位宽，对齐 XGBoost 3.x 内存布局
            X_clean = X_df.astype(np.float32)

            # 2. 目标变量净化 (防污染逻辑)
            # y_j: 评委分; y_f: 估计得票率
            y_j = self._robust_clean(self.df['week_avg_score']).fillna(7.0).values.astype(np.float32)
            y_f = self._robust_clean(self.df['est_fan_vote_mu']).fillna(0.1).values.astype(np.float32)

            # 3. 训练代理模型 (Proxy Model)
            # 显式指定 base_score 为纯浮点数，避开 XGBoost 内部的 JSON 数组 Bug
            params = {
                'n_estimators': 80,
                'max_depth': 4,
                'learning_rate': 0.1,
                'tree_method': 'exact',  # 3.x 版本 exact 更稳定
                'base_score': 0.5,
                'n_jobs': 1  # 避免多进程嵌套导致的信号量死锁
            }

            self.logger.info("正在拟合非线性代理树...")
            model_j = xgb.XGBRegressor(**params).fit(X_clean, y_j)
            model_f = xgb.XGBRegressor(**params).fit(X_clean, y_f)

            import shap

            # 4. [架构师核心补丁] 建立健壮的解释器工厂
            def get_robust_explanation(model, data):
                """
                [架构师级补丁]: 解决 SHAP 0.49+ 在高维特征下的 max_evals 校验崩溃问题。
                物理意义：利用 Monte Carlo 置换抽样计算特征的边际贡献（Shapley Values）。
                """
                try:
                    # 1. 强制数据流脱敏：将 DataFrame 降维为纯 Numpy 矩阵，位宽压缩至 float32
                    X_array = data.values.astype(np.float32)

                    # 2. 建立独立掩码场 (Masker)
                    # max_samples=100 是精度与速度的“帕累托最优”点，足以代表数据背景分布
                    masker = shap.maskers.Independent(X_array, max_samples=100)

                    # 3. 实例化置换解释器 (Permutation Explainer)
                    # 注入 model.predict，将 XGBoost 视为一个标准的可调用黑盒
                    explainer = shap.Explainer(model.predict, masker)

                    # 4. 执行边际贡献推断
                    # [核心修复]: 将 max_evals 设为 500。
                    # 理由：SHAP 的 ExactExplainer 需要至少 2^k 次运算。
                    # 你的特征维数 k 约在 8-10 之间，500 次评价足以确保算法闭环。
                    shap_values = explainer(
                        X_array,
                        max_evals=500,
                        batch_size=50  # 向量化分批处理，提速显著
                    )

                    # 5. 元数据还原：将 SHAP 内部生成的 values 重新挂载特征名称，适配后续绘图
                    # 这能确保你在调用 self.plotter 时，坐标轴上的名称是人类可读的
                    if hasattr(shap_values, "feature_names"):
                        shap_values.feature_names = data.columns.tolist()

                    return shap_values

                except Exception as e:
                    self.logger.error(f"❌ 置换解释器内核异常: {str(e)}")
                    return None

            self.logger.info("执行非线性贡献度分解 (Sampling-based Inference)...")
            shap_values_j = get_robust_explanation(model_j, X_clean)
            shap_values_f = get_robust_explanation(model_f, X_clean)

            if shap_values_j is not None and shap_values_f is not None:
                self.logger.info("✅ SHAP 任务圆满完成。归因矩阵已生成。")
            else:
                self.logger.warning("归因矩阵生成异常，将由后序绘图组件执行空值保护。")

            return X_clean, shap_values_j, shap_values_f

        except Exception as e:
            self.logger.error(f"❌ Stage 4 归因引擎发生未捕获异常: {str(e)}", exc_info=True)
            return None, None, None

    def plot_global_importance(self, shap_values):
        """
        【工业级重构】手动构建特征重要性图。
        原理：直接提取 |SHAP| 均值，彻底摆脱 shap.plots 库的 API 陷阱。
        """
        if shap_values is None: return

        try:
            # 1. 提取原始数值 (核心解耦)
            # shap_values.values 是 (N_samples, N_features) 的矩阵
            # shap_values.feature_names 是特征列表
            vals = np.abs(shap_values.values).mean(axis=0)
            feature_names = shap_values.feature_names

            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': vals
            }).sort_values(by='importance', ascending=True).tail(12)

            # 2. 原生 Matplotlib 渲染
            plt.figure(figsize=(10, 6))
            colors = [self.plotter.colors['fan'] if 'ind' in f else self.plotter.colors['judge']
                      for f in importance_df['feature']]

            bars = plt.barh(importance_df['feature'], importance_df['importance'], color=colors, alpha=0.8)
            plt.grid(axis='x', linestyle=':', alpha=0.5)
            plt.title("Latent Determinants of Success: Mean |SHAP| Impact", fontsize=14, fontweight='bold')
            plt.xlabel("Average impact on model output magnitude (Normalized)", fontsize=12)

            plt.tight_layout()
            self.plotter.save_figure("task3_shap_global_importance.png")
            self.logger.info("✅ 全局重要性图已通过原生驱动生成。")
        except Exception as e:
            self.logger.error(f"SHAP 全局绘图失败: {e}")

    def plot_age_dependence_contrast(self, shap_j, shap_f):
        """
        【学术杀手锏】原生驱动的年龄依赖对比图。
        彻底删除所有对 shap.plots.scatter 的调用。
        """
        if shap_j is None or shap_f is None: return

        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
            target_col = "celebrity_age_during_season"

            # 内部渲染算子
            def render_scatter(ax, shap_obj, color, title, ylabel):
                # 提取原始数据
                # 注意：shap_obj 可能是 Explanation 对象
                x = shap_obj[:, target_col].data
                y = shap_obj[:, target_col].values

                # 散点渲染
                ax.scatter(x, y, color=color, alpha=0.4, s=35, edgecolors='white', linewidth=0.4)

                # 添加学术级趋势拟合 (LOWESS 平滑近似)
                # 使用多项式拟合作为稳健替代
                z = np.polyfit(x.astype(float), y.astype(float), 2)
                p = np.poly1d(z)
                xp = np.linspace(x.min(), x.max(), 100)
                ax.plot(xp, p(xp), color='black', linestyle='--', linewidth=2.5, label='Non-linear Trend')

                ax.set_title(title, fontsize=14, fontweight='bold')
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_xlabel("Celebrity Age", fontsize=12)
                ax.grid(True, alpha=0.2)

            # 执行双路渲染
            render_scatter(ax1, shap_j, self.plotter.colors['judge'],
                           "Expert Evaluative Heuristics", "SHAP Impact on Scores")
            render_scatter(ax2, shap_f, self.plotter.colors['fan'],
                           "Public Sentiment Heuristics", "SHAP Impact on Vote Share")

            plt.suptitle("Age Heterogeneity: Decoding Structural Bias between Expert and Populist Criteria",
                         fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            self.plotter.save_figure("task3_age_heterogeneity_dependence.png")
            self.logger.info("✅ 年龄依赖对比图已通过原生驱动生成（已绕过 API 陷阱）。")
        except Exception as e:
            self.logger.error(f"SHAP 依赖图绘制失败: {e}")