# ==============================================================================
# src/analysis/shap_interpreter.py
# Role: Explainable AI (XAI) Engine - Task 3 Non-linear Analysis
# Function: Computing SHAP values for both Judge and Fan paths
# Viz: Global Beeswarm plots & Feature Interaction Dependence plots
# ==============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os


class ShapInterpreter:
    """
    SHAP 解释引擎：
    利用沙普利值量化每个特征对估计得票数的边际贡献，揭示‘人设’与‘技术’的非线性博弈。
    """

    def __init__(self, df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("SHAP_INTERPRETER")
        # 仅针对有效推断样本进行 XAI 分析
        self.df = df_platinum.dropna(subset=['est_fan_vote_mu', 'week_avg_score']).copy()
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)
        # 学术风格设置
        plt.rcParams['font.family'] = 'serif'
        sns.set_context("paper", font_scale=1.4)

    def _prepare_feature_matrix(self):
        """构造回归特征矩阵 (X)"""
        # 选取 Task 3 关注的核心特征
        num_feats = ['celebrity_age_during_season', 'partner_alpha', 'score_delta']
        cat_feats = ['industry_group']

        # 1. 数值型特征
        X_num = self.df[num_feats].fillna(self.df[num_feats].median())

        # 2. 类别型特征 (One-hot 编码)
        X_cat = pd.get_dummies(self.df[cat_feats], prefix='ind', dtype=int)

        # 3. 拼接
        X = pd.concat([X_num, X_cat], axis=1)
        return X

    def run_dual_shap_analysis(self):
        """
        核心逻辑：计算评委路径和粉丝路径的 SHAP 值。
        """
        self.logger.info(">>> 启动双路 SHAP 非线性归因分析...")
        X = self._prepare_feature_matrix()

        # 目标 A: 评委技术分 (z_score)
        y_j = self.df['score_z']
        # 目标 B: 粉丝投票分 (posterior_mu)
        y_f = self.df['est_fan_vote_mu']

        # 训练两个代理模型 (XGBoost)
        # 注意：此处 base_score 强制转换以解决之前的序列化 Bug
        model_j = xgb.XGBRegressor(n_estimators=100, max_depth=3, base_score=float(y_j.mean())).fit(X, y_j)
        model_f = xgb.XGBRegressor(n_estimators=100, max_depth=3, base_score=float(y_f.mean())).fit(X, y_f)

        # 计算 SHAP
        explainer_j = shap.TreeExplainer(model_j)
        explainer_f = shap.TreeExplainer(model_f)

        # 封装为 Explanation 对象，适配新版绘图接口
        shap_j = explainer_j(X)
        shap_f = explainer_f(X)

        return X, shap_j, shap_f

    def plot_global_importance(self, shap_f):
        """绘制粉丝得票的全局重要性 (Beeswarm)"""
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(shap_f, max_display=12, show=False)
        plt.title("Latent Determinants of Fan Preference (SHAP Global)", fontsize=14)

        path = os.path.join(self.fig_dir, "shap_global_beeswarm.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"SHAP 全局图已保存: {path}")

    def plot_age_dependence_contrast(self, shap_j, shap_f):
        """
        【学术杀手锏】对比年龄(Age)在评委侧和观众侧的异质性影响曲线。
        物理意义：揭示“实力派”与“养成系/情怀系”的生存差异。
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 评委侧：通常是线性递减
        shap.plots.scatter(shap_j[:, "celebrity_age_during_season"], ax=ax1, show=False, color='#ff7f0e')
        ax1.set_title("Age vs. Technical Merit (Judge Side)")
        ax1.set_ylabel("SHAP Value (Impact on Score)")

        # 观众侧：可能是 U 型分布
        shap.plots.scatter(shap_f[:, "celebrity_age_during_season"], ax=ax2, show=False, color='#1f77b4')
        ax2.set_title("Age vs. Public Sentiment (Fan Side)")
        ax2.set_ylabel("SHAP Value (Impact on Vote)")

        plt.tight_layout()
        path = os.path.join(self.fig_dir, "age_heterogeneity_dependence.png")
        plt.savefig(path, dpi=300)
        plt.close()
        self.logger.info(f"年龄异质性曲线已保存: {path}")


# ------------------------------------------------------------------------------
#量化‘非线性贡献’
# ------------------------------------------------------------------------------
def extract_top_interactions(shap_explanation):
    """
    寻找交互作用最强的特征对。
    用于回答 Task 3 中复杂的‘things impact’。
    """
    # 逻辑：分析 SHAP Interaction Values 矩阵
    pass