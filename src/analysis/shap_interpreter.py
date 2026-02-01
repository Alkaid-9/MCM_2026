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
    线性模型 (LMM) 只能告诉我们“年龄越大越吃亏”，但 SHAP 可以告诉我们
    “年龄超过 50 岁后，扣分速度会突然加快” (非线性阈值效应)。
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
        【脏数据防火墙】
        修复上游 ETL 可能残留的 List-like 字符串 (e.g. '[-1.03E-1]' 或 "['0.5']").
        物理意义：确保进入 XGBoost 的全是纯净的 float64。
        """

        def clean_val(x):
            if isinstance(x, (int, float)):
                return x
            if isinstance(x, str):
                # 移除方括号和引号，取第一个值
                clean = x.replace('[', '').replace(']', '').replace("'", "").replace('"', "")
                try:
                    # 处理逗号分隔的情况 '0.1, 0.2'
                    return float(clean.split(',')[0])
                except ValueError:
                    return np.nan
            return np.nan

        return series.apply(clean_val)

    def _prepare_feature_matrix(self):
        """
        构造回归特征矩阵 (X) 并进行清洗。
        """
        self.logger.info("构造并清洗特征矩阵...")

        # 1. 核心数值特征
        # Partner Alpha: 舞伴红利
        # Score Delta: 进步幅度 (成长剧本)
        # Week Num: 赛程进度
        num_feats = ['celebrity_age_during_season', 'partner_alpha', 'score_delta', 'week_num']

        # 强力清洗数值列
        for col in num_feats:
            if col in self.df.columns:
                self.df[col] = self._robust_clean(self.df[col])
                # 中位数填充缺失 (鲁棒性处理)
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
            else:
                self.df[col] = 0.0

        X_num = self.df[num_feats]

        # 2. 类别型特征 (One-hot 编码)
        # 确保 industry_group 存在
        if 'industry_group' not in self.df.columns:
            self.df['industry_group'] = 'Unknown'

        X_cat = pd.get_dummies(self.df['industry_group'], prefix='ind', dtype=int)

        # 3. 拼接
        X = pd.concat([X_num, X_cat], axis=1)

        # 4. [关键] 清洗列名，防止 XGBoost 报错 (列名不能包含 [, ], <)
        clean_cols = [re.sub(r'[\[\]<]', '', col) for col in X.columns]
        X.columns = clean_cols

        return X

    def run_dual_shap_analysis(self):
        """
        核心逻辑：计算评委路径 (Judge) 和粉丝路径 (Fan) 的 SHAP 值。
        """
        self.logger.info(">>> 启动双路 SHAP 非线性归因分析...")

        try:
            X = self._prepare_feature_matrix()
            if X.empty or len(X) < 50:
                self.logger.warning("样本量不足 (<50)，跳过 SHAP 分析以防过拟合。")
                return None, None, None

            # 目标 A: 评委技术分 (Ground Truth)
            y_j = self._robust_clean(self.df['week_avg_score'])
            # 目标 B: 粉丝投票分 (Latent Variable from Task 1)
            y_f = self._robust_clean(self.df['est_fan_vote_mu'])

            # 训练配置 (注重解释性而非极致预测)
            params = {
                'n_estimators': 150,
                'max_depth': 4,  # 浅树，防止过拟合
                'learning_rate': 0.05,
                'n_jobs': 1,  # 限制单核防止并行死锁
                'random_state': 2026,
                'tree_method': 'hist'
            }

            self.logger.info("训练 XGBoost 代理模型...")
            model_j = xgb.XGBRegressor(**params).fit(X, y_j)
            model_f = xgb.XGBRegressor(**params).fit(X, y_f)

            # 计算 SHAP
            import shap
            # 使用 TreeExplainer，它比 KernelExplainer 快得多且稳定
            explainer_j = shap.TreeExplainer(model_j)
            explainer_f = shap.TreeExplainer(model_f)

            shap_j = explainer_j(X)
            shap_f = explainer_f(X)

            self.logger.info("SHAP 值计算成功。")
            return X, shap_j, shap_f

        except Exception as e:
            self.logger.error(f"SHAP 引擎严重计算异常: {str(e)}", exc_info=True)
            # 熔断保护：返回 None，让流水线继续运行
            return None, None, None

    def plot_global_importance(self, shap_values):
        """
        绘制全局重要性 (Beeswarm Plot)。
        """
        if shap_values is None: return
        try:
            import shap
            plt.figure(figsize=(10, 8))
            # Beeswarm plot 展示特征密度
            shap.plots.beeswarm(
                shap_values,
                max_display=12,
                show=False,
                color=plt.get_cmap("coolwarm")
            )
            plt.title("Latent Determinants of Fan Preference (Non-linear Impact)", fontsize=14, pad=20)
            plt.tight_layout()

            self.plotter.save_figure("task3_shap_global_beeswarm.png")
        except Exception as e:
            self.logger.warning(f"SHAP 全局绘图失败: {e}")

    def plot_age_dependence_contrast(self, shap_j, shap_f):
        """
        【学术杀手锏】对比年龄 (Age) 在评委侧和观众侧的异质性影响曲线。
        物理意义：揭示“实力派”与“养成系/情怀系”的生存差异。
        """
        if shap_j is None or shap_f is None: return

        try:
            import shap
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # --- 左图：评委侧 (Meritocracy) ---
            # 通常是线性递减 (偏见：老了跳不动)
            # 强制使用项目标准色：橙色
            shap.plots.scatter(
                shap_j[:, "celebrity_age_during_season"],
                ax=ax1,
                show=False,
                color=self.plotter.colors['judge'],
                alpha=0.6,
                s=20
            )
            ax1.set_title("Judge Bias: Age vs. Technical Merit", fontsize=14, fontweight='bold')
            ax1.set_ylabel("SHAP Value (Impact on Score)", fontsize=12)
            ax1.set_xlabel("Celebrity Age", fontsize=12)
            ax1.grid(True, alpha=0.3, linestyle='--')

            # --- 右图：观众侧 (Populism) ---
            # 可能是 U 型分布 (喜欢小鲜肉 + 敬重老戏骨)
            # 强制使用项目标准色：蓝色
            shap.plots.scatter(
                shap_f[:, "celebrity_age_during_season"],
                ax=ax2,
                show=False,
                color=self.plotter.colors['fan'],
                alpha=0.6,
                s=20
            )
            ax2.set_title("Fan Sentiment: Age vs. Popularity", fontsize=14, fontweight='bold')
            ax2.set_ylabel("SHAP Value (Impact on Votes)", fontsize=12)
            ax2.set_xlabel("Celebrity Age", fontsize=12)
            ax2.grid(True, alpha=0.3, linestyle='--')

            plt.suptitle("Heterogeneous Effects of Age on Success: Linearity vs. Non-linearity", fontsize=16)
            plt.tight_layout()

            self.plotter.save_figure("task3_age_heterogeneity_dependence.png")
            self.logger.info("年龄异质性对比图已生成。")

        except Exception as e:
            self.logger.warning(f"SHAP 依赖图绘制失败: {e}")