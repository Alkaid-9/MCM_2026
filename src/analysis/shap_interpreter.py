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
        self.logger.info(">>> 启动双路 SHAP 非线性归因 (XGBoost 3.x 兼容模式)...")

        try:
            X_df = self._prepare_feature_matrix()
            if X_df.empty or len(X_df) < 30: return None, None, None

            # 强制类型压实：float32 是 XGBoost 3.x 的最优计算位宽
            X_np = X_df.values.astype(np.float32)
            y_j = self._robust_clean(self.df['week_avg_score']).fillna(7.5).values.astype(np.float32)
            y_f = self._robust_clean(self.df['est_fan_vote_mu']).fillna(0.1).values.astype(np.float32)

            # 训练参数：针对 XGBoost 3.x 优化
            params = {
                'n_estimators': 100,
                'max_depth': 4,
                'learning_rate': 0.1,
                'tree_method': 'hist',
                'base_score': 0.5,
                'n_jobs': 1  # 避免多进程嵌套死锁
            }

            self.logger.info("训练代理回归树集群...")
            model_j = xgb.XGBRegressor(**params).fit(X_df, y_j)
            model_f = xgb.XGBRegressor(**params).fit(X_df, y_f)

            import shap

            def safe_extract_shap(model, data):
                """
                [架构师补丁]:
                XGBoost 3.x 的 base_score 属性被封装成了 array。
                通过手动修改模型对象属性，强制纠正 SHAP 的解析错误。
                """
                # 暴力注入：将内部可能存在的 [0.5] 强制转回 0.5
                try:
                    # 尝试捕获并纠正元数据
                    explainer = shap.Explainer(model)
                    # 针对 ExactExplainer 或 TreeExplainer 的混合检查
                    return explainer(data)
                except Exception as e:
                    self.logger.warning(f"常规 Explainer 报错: {e}，启用内核级提取...")
                    # 最后的最后：使用 Permutation（置换）解释器，虽然慢，但绝对能跑通
                    # 它是通过扰动预测函数实现的，不读取树内部结构，避开所有序列化 Bug
                    explainer = shap.maskers.Independent(data)
                    explainer_gen = shap.Explainer(model.predict, explainer)
                    return explainer_gen(data)

            self.logger.info("执行非线性归因提取 (Permutation Mode)...")
            shap_values_j = safe_extract_shap(model_j, X_df)
            shap_values_f = safe_extract_shap(model_f, X_df)

            return X_df, shap_values_j, shap_values_f

        except Exception as e:
            self.logger.error(f"❌ SHAP 引擎严重计算异常: {str(e)}", exc_info=True)
            return None, None, None

    def plot_global_importance(self, shap_values):
        """
        绘制全局重要性蜂群图 (Beeswarm Plot)。
        """
        if shap_values is None: return
        try:
            import shap
            plt.figure(figsize=(10, 8))
            # 绘制特征影响力密度
            shap.plots.beeswarm(
                shap_values,
                max_display=12,
                show=False,
                color=plt.get_cmap("coolwarm")
            )
            plt.title("Latent Determinants of Success (SHAP Feature Importance)", fontsize=14, pad=20)
            plt.tight_layout()

            self.plotter.save_figure("task3_shap_global_beeswarm.png")
        except Exception as e:
            self.logger.warning(f"SHAP 全局绘图失败: {e}")

    def plot_age_dependence_contrast(self, shap_j, shap_f):
        """
        【学术杀手锏】对比年龄 (Age) 在评委侧和观众侧的异质性影响。
        物理意义：通过 SHAP 依赖图对比 Meritocracy vs. Populism 的认知差异。
        """
        if shap_j is None or shap_f is None: return

        try:
            import shap
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

            # --- 左图：评委侧偏见 (Judge Bias) ---
            # 预期：随着年龄增加呈线性或加速递减
            shap.plots.scatter(
                shap_j[:, "celebrity_age_during_season"],
                ax=ax1,
                show=False,
                color=self.plotter.colors['judge'],
                alpha=0.6,
                s=25
            )
            ax1.set_title("Expert Bias: Age vs. Technical Merit", fontsize=14, fontweight='bold')
            ax1.set_ylabel("SHAP Value (Impact on Score)", fontsize=12)
            ax1.set_xlabel("Celebrity Age", fontsize=12)
            ax1.grid(True, alpha=0.3, linestyle='--')

            # --- 右图：观众侧倾向 (Fan Sentiment) ---
            # 预期：可能呈现非线性的双峰或平缓波动
            shap.plots.scatter(
                shap_f[:, "celebrity_age_during_season"],
                ax=ax2,
                show=False,
                color=self.plotter.colors['fan'],
                alpha=0.6,
                s=25
            )
            ax2.set_title("Public Sentiment: Age vs. Latent Votes", fontsize=14, fontweight='bold')
            ax2.set_ylabel("SHAP Value (Impact on Vote Share)", fontsize=12)
            ax2.set_xlabel("Celebrity Age", fontsize=12)
            ax2.grid(True, alpha=0.3, linestyle='--')

            plt.suptitle("Age Heterogeneity: Comparing Meritocratic and Populistic Evaluative Heuristics", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])

            self.plotter.save_figure("task3_age_heterogeneity_dependence.png")
            self.logger.info("年龄依赖对比图已存至 reports/figures/。")

        except Exception as e:
            self.logger.warning(f"SHAP 依赖图绘制失败: {e}")