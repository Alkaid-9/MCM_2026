# ==============================================================================
# src/analysis/shap_interpreter.py
# Role: Explainable AI (XAI) Engine - Task 3 Non-linear Analysis
# Function: Computing SHAP values for both Judge and Fan paths
# Viz: Global Beeswarm plots & Feature Interaction Dependence plots
# Fix: Added strict string-cleaning firewall for "[-1.23E-1]" artifacts
# ==============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import re
from src.utils.plotting import DWTSPlotter


class ShapInterpreter:
    """
    SHAP 解释引擎：
    利用博弈论沙普利值 (Shapley Values) 量化每个特征对得分的边际贡献。
    重点揭示：线性模型无法捕捉的非线性阈值效应（例如：年龄大到一定程度扣分加速）。
    """

    def __init__(self, df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("SHAP_INTERPRETER")
        # 仅针对有效推断样本进行 XAI 分析
        self.df = df_platinum.dropna(subset=['est_fan_vote_mu', 'week_avg_score']).copy()
        self.fig_dir = fig_dir
        self.plotter = DWTSPlotter(output_dir=fig_dir)
        os.makedirs(self.fig_dir, exist_ok=True)

    def _robust_clean(self, series: pd.Series) -> pd.Series:
        """
        【脏数据防火墙】
        修复上游 ETL 可能残留的 List-like 字符串 (e.g. '[-1.03E-1]')。
        """

        def clean_val(x):
            if isinstance(x, (int, float)):
                return x
            if isinstance(x, str):
                # 移除方括号，取第一个值
                clean = x.replace('[', '').replace(']', '').replace("'", "")
                try:
                    # 处理逗号分隔的情况 '0.1, 0.2'
                    return float(clean.split(',')[0])
                except ValueError:
                    return np.nan
            return np.nan

        return series.apply(clean_val)

    def _prepare_feature_matrix(self):
        """构造回归特征矩阵 (X) 并进行清洗"""
        self.logger.info("构造并清洗特征矩阵...")

        # 1. 核心数值特征
        num_feats = ['celebrity_age_during_season', 'partner_alpha', 'score_delta', 'week_num']

        # 强力清洗数值列
        for col in num_feats:
            if col in self.df.columns:
                self.df[col] = self._robust_clean(self.df[col])
                # 中位数填充缺失
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

        # 4. [关键] 清洗列名，防止 XGBoost 报错 (不能包含 [ ] <)
        clean_cols = [re.sub(r'[\[\]<]', '', col) for col in X.columns]
        X.columns = clean_cols

        return X

    def run_dual_shap_analysis(self):
        """
        核心逻辑：计算评委路径和粉丝路径的 SHAP 值。
        """
        self.logger.info(">>> 启动双路 SHAP 非线性归因分析...")

        try:
            X = self._prepare_feature_matrix()
            if X.empty or len(X) < 50:
                self.logger.warning("样本量不足 (<50)，跳过 SHAP 分析。")
                return None, None, None

            # 目标 A: 评委技术分 (使用 Robust Z-Score 如果有，否则用 Raw)
            y_j = self._robust_clean(self.df['week_avg_score'])

            # 目标 B: 粉丝投票分 (posterior_mu)
            y_f = self._robust_clean(self.df['est_fan_vote_mu'])

            # 训练配置
            params = {
                'n_estimators': 150,
                'max_depth': 4,
                'learning_rate': 0.05,
                'n_jobs': 1,  # 限制单核防止死锁
                'random_state': 2026,
                'tree_method': 'hist'  # 加速
            }

            # 训练两个代理模型
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
            # 熔断保护：返回 None，流水线继续运行
            return None, None, None

    def plot_global_importance(self, shap_values):
        """绘制全局重要性 (Beeswarm)"""
        if shap_values is None: return

        try:
            import shap
            plt.figure(figsize=(10, 8))
            # Beeswarm plot 展示特征密度
            shap.plots.beeswarm(shap_values, max_display=12, show=False, color=plt.get_cmap("coolwarm"))

            plt.title("Latent Determinants of Fan Preference (Non-linear Impact)", fontsize=14, pad=20)
            # 调整布局以适应长标签
            plt.tight_layout()
            self.plotter.save_figure("task3_shap_global_beeswarm.png")
        except Exception as e:
            self.logger.warning(f"绘图失败: {e}")

    def plot_age_dependence_contrast(self, shap_j, shap_f):
        """
        【学术杀手锏】对比年龄(Age)在评委侧和观众侧的异质性影响曲线。
        物理意义：揭示“实力派”与“养成系/情怀系”的生存差异。
        """
        if shap_j is None or shap_f is None: return

        try:
            import shap
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # 评委侧：通常是线性递减 (偏见：老了跳不动)
            shap.plots.scatter(shap_j[:, "celebrity_age_during_season"], ax=ax1, show=False, color='#ff7f0e', alpha=0.6)
            ax1.set_title("Judge Bias: Age vs. Technical Merit", fontsize=14)
            ax1.set_ylabel("SHAP Value (Score Impact)", fontsize=12)
            ax1.set_xlabel("Age", fontsize=12)
            ax1.grid(True, alpha=0.3)

            # 观众侧：可能是 U 型分布 (喜欢小鲜肉 + 敬重老戏骨)
            shap.plots.scatter(shap_f[:, "celebrity_age_during_season"], ax=ax2, show=False, color='#1f77b4', alpha=0.6)
            ax2.set_title("Fan Sentiment: Age vs. Popularity", fontsize=14)
            ax2.set_ylabel("SHAP Value (Vote Impact)", fontsize=12)
            ax2.set_xlabel("Age", fontsize=12)
            ax2.grid(True, alpha=0.3)

            plt.suptitle("Heterogeneous Effects of Age on Success", fontsize=16)
            plt.tight_layout()
            self.plotter.save_figure("task3_age_heterogeneity_dependence.png")

        except Exception as e:
            self.logger.warning(f"SHAP 依赖图绘制失败: {e}")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 构造脏数据 Mock
    df_mock = pd.DataFrame({
        'week_avg_score': [20, 25, 30, '[-1.03E-1]', 22],  # 包含脏字符串
        'est_fan_vote_mu': [0.1, 0.2, 0.3, 0.4, 0.5],
        'celebrity_age_during_season': [20, 30, 40, 50, 60],
        'partner_alpha': [1.0, 1.2, 0.8, 1.5, 1.0],
        'score_delta': [0, 1, -1, 2, 0],
        'week_num': [1, 2, 3, 4, 5],
        'industry_group': ['A', 'B', 'A', 'B', 'C']
    })

    interpreter = ShapInterpreter(df_mock)
    X, s_j, s_f = interpreter.run_dual_shap_analysis()

    if s_j is not None:
        print("[PASS] 脏数据清洗与 SHAP 计算成功！")
        interpreter.plot_age_dependence_contrast(s_j, s_f)
    else:
        print("[FAIL] 计算失败")