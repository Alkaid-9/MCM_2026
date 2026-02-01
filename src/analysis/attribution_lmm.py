# ==============================================================================
# src/analysis/attribution_lmm.py
# Role: Causal Attribution Engine - Hierarchical Modeling (Task 3)
# Function: Partitioning variance between Celebrity traits and Partner effects
# Method: Linear Mixed-Effects Model (LMM) & ICC Decomposition
# ==============================================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import logging
import warnings
import os


class LMMAttributionEngine:
    """
    归因引擎：利用混合效应模型解决‘名师带高徒’的内生性问题。
    """

    def __init__(self, df_platinum: pd.DataFrame, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("LMM_ENGINE")
        # 预过滤：仅针对有反演票数的有效观测进行因果推断
        self.df = df_platinum.dropna(subset=['est_fan_vote_mu', 'week_avg_score']).copy()
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)
        self._prepare_data()

    def _prepare_data(self):
        """标准化特征，使 Beta 系数在不同量纲间可比"""
        scaler = StandardScaler()
        # 目标变量标准化：后验票数权重 与 原始技术分
        self.df['z_fan_vote'] = scaler.fit_transform(self.df[['est_fan_vote_mu']])
        self.df['z_judge_score'] = scaler.fit_transform(self.df[['week_avg_score']])

        # 自变量标准化：年龄、动量等
        self.df['z_age'] = scaler.fit_transform(self.df[['celebrity_age_during_season']].fillna(35))
        self.df['z_momentum'] = scaler.fit_transform(self.df[['score_delta']].fillna(0))

        # 行业类别归一化
        self.df['industry'] = self.df['industry_group'].fillna('Other')
        self.logger.info(f"LMM 预处理完成。分析样本数: {len(self.df)}")

    def run_dual_path_lmm(self):
        """
        核心算法：双路径 LMM 拟合。
        路径 J (Judge): 技术分的影响因子。
        路径 F (Fan):   投票分的影响因子。
        """
        self.logger.info(">>> 正在执行双路径 LMM 方差分解...")

        # 混合效应公式：固定效应 (Age + Industry) + 随机效应 (Partner)
        # 物理意义：(1|ballroom_partner) 表示我们承认舞伴有偏置，并试图将其残差化
        formula = " ~ z_age + z_momentum + C(industry)"

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                # 1. 拟合评委偏好模型
                model_j = smf.mixedlm("z_judge_score" + formula, self.df,
                                      groups=self.df["ballroom_partner"]).fit(method='nm')

                # 2. 拟合粉丝偏好模型
                model_f = smf.mixedlm("z_fan_vote" + formula, self.df,
                                      groups=self.df["ballroom_partner"]).fit(method='nm')

            self.logger.info("LMM 拟合成功。开始计算方差贡献率...")
            return model_j, model_f
        except Exception as e:
            self.logger.error(f"LMM 拟合崩溃: {e}")
            return None, None

    def calculate_icc(self, model):
        """
        计算 ICC (组内相关系数)。
        物理意义：量化‘舞伴’这个因素解释了多少百分比的得分差异。
        """
        if model is None: return 0.0
        var_random = float(model.cov_re.iloc[0, 0])  # 舞伴带来的方差
        var_resid = float(model.scale)  # 剩余无法解释的方差
        icc = var_random / (var_random + var_resid)
        return icc

    def plot_coefficient_butterfly(self, model_j, model_f):
        """
        绘制蝴蝶图 (Butterfly Plot)：直观对比评委 vs 观众的偏好鸿沟。
        """
        # 提取固定效应系数
        params_j = model_j.params.drop(['Intercept', 'Group Var'], errors='ignore')
        params_f = model_f.params.drop(['Intercept', 'Group Var'], errors='ignore')

        # 对齐特征名
        features = [f.replace("C(industry)[T.", "").replace("]", "") for f in params_j.index]

        plot_df = pd.DataFrame({
            'Feature': features,
            'Judge_Beta': params_j.values,
            'Fan_Beta': params_f.values
        }).sort_values('Fan_Beta')

        plt.figure(figsize=(10, 8))
        y_pos = np.arange(len(plot_df))

        plt.barh(y_pos + 0.2, plot_df['Fan_Beta'], 0.4, label='Public Sentiment', color='#1f77b4', alpha=0.8)
        plt.barh(y_pos - 0.2, plot_df['Judge_Beta'], 0.4, label='Expert Quality', color='#ff7f0e', alpha=0.8)

        plt.yticks(y_pos, plot_df['Feature'])
        plt.axvline(0, color='black', lw=1, ls='--')
        plt.title("Meritocracy vs. Populism: Beta Coefficient Comparison", fontsize=14)
        plt.xlabel("Standardized Impact (Impact per StdDev)")
        plt.legend()
        plt.tight_layout()

        path = os.path.join(self.fig_dir, "lmm_butterfly_contrast.png")
        plt.savefig(path, dpi=300)
        plt.close()
        self.logger.info(f"蝴蝶图已生成: {path}")


# ------------------------------------------------------------------------------
# 量化‘认知背离度’ (Task 3 的终极指标)
# ------------------------------------------------------------------------------
def compute_preference_divergence(model_j, model_f):
    """计算评委与观众偏好向量的余弦相似度"""
    v_j = model_j.params.drop(['Intercept', 'Group Var'], errors='ignore').values
    v_f = model_f.params.drop(['Intercept', 'Group Var'], errors='ignore').values
    cos_sim = np.dot(v_j, v_f) / (np.linalg.norm(v_j) * np.linalg.norm(v_f))
    return 1 - cos_sim  # 越大表示审美越分裂