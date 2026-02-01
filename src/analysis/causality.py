"""
Causal Attribution & SHAP Engine (Final Robust Version v5.6)
Role: Decomposing "Star Power" vs. "Meritocracy".
Function:
    - Preprocessing: Gender inference & Interaction term generation.
    - Linear Mixed-Effects Models (LMM) for coefficient butterfly plots.
    - XGBoost + SHAP for non-linear feature attribution (FIXED for SHAP ValueError).
Standard: Top-Tier Econometrics / Explainable AI (XAI).
"""

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import logging
import warnings

# 引入前置预处理器
from src.analysis.causality_prep import CausalityPreprocessor

# 设置学术绘图风格
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
sns.set_context("paper", font_scale=1.4)


class CausalityEngine:
    """
    因果归因引擎：
    量化明星属性与舞伴特征对比赛结果的贡献度。
    """

    def __init__(self, df_platinum: pd.DataFrame, figures_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("CAUSALITY_ENGINE")
        self.fig_dir = figures_dir
        os.makedirs(self.fig_dir, exist_ok=True)

        # --- 1. 调用预处理器 (Data Enrichment) ---
        # 补全性别、计算交互项、标准化数值
        self.logger.info("初始化因果预处理器...")
        preprocessor = CausalityPreprocessor(df_platinum)
        self.df = preprocessor.process()

    def run_lmm_comparison(self):
        """
        [逻辑 A]: 混合效应模型 (Mixed-Effects Analysis)
        目标：对比同一特征在“评委打分方程”和“粉丝投票方程”中的系数差异。

        Formula: Y ~ Age + Industry + Interaction(Male*PartnerAlpha) + (1|Partner)
        """
        self.logger.info(">>> 启动混合效应模型 (LMM) 对照实验...")

        # 定义回归方程
        # z_age, inter_male_x_partner 是在 prep 阶段生成的
        # C(celebrity_industry) 处理行业分类变量
        formula = " ~ z_celebrity_age_during_season + inter_male_x_partner + C(celebrity_industry)"

        with warnings.catch_warnings():
            # 忽略收敛警告，保证流水线不中断
            warnings.filterwarnings("ignore", category=sm.tools.sm_exceptions.ConvergenceWarning)

            try:
                # 1. 粉丝投票模型 (Dependent: z_fan_vote)
                # 随机效应：ballroom_partner (控制舞伴个体差异)
                model_fan = smf.mixedlm(
                    "z_fan_vote" + formula,
                    self.df,
                    groups=self.df["ballroom_partner"]
                ).fit(maxiter=2000, method='nm')  # Nelder-Mead 优化更稳健

                # 2. 评委打分模型 (Dependent: z_judge_score)
                model_judge = smf.mixedlm(
                    "z_judge_score" + formula,
                    self.df,
                    groups=self.df["ballroom_partner"]
                ).fit(maxiter=2000, method='nm')

                self.logger.info("LMM 模型拟合成功。")
                return model_fan, model_judge

            except Exception as e:
                self.logger.error(f"LMM 最终收敛失败 (Final Fail): {e}")
                self.logger.info("学术处理：将降级依赖 SHAP 结果，并在论文中解释 LMM 失败的原因(稀疏性)。")
                return None, None

    def run_shap_attribution(self):
        """
        [逻辑 B]: 基于 XGBoost + SHAP 的非线性归因
        核心修复：强制设定 base_score，解决 SHAP 的 ValueError。
        """
        self.logger.info(">>> 启动 XGBoost + SHAP 非线性归因 (FIXED)...")

        # 1. 准备特征矩阵 X
        # 数值特征 (已在 prep 中标准化为 z_*)
        num_feats = [
            'z_celebrity_age_during_season',
            'z_partner_alpha',
            'z_score_delta',
            'is_male',
            'inter_male_x_partner',
            'week_avg_score'  # 原始技术分也是重要特征
        ]

        # 行业 One-Hot
        X = pd.get_dummies(self.df[['celebrity_industry']], prefix='ind', dtype=int)

        # 合并数值特征
        # 注意：使用原始 df 中的列，prep 阶段已处理好填充
        for col in num_feats:
            if col in self.df.columns:
                X[col] = self.df[col]
            else:
                X[col] = 0.0

        # 目标变量：隐变量票数 (未标准化，保留物理意义)
        y = self.df['est_fan_vote_mu']

        # 2. 训练 XGBoost
        # [CRITICAL FIX]: base_score=y.mean() 防止 JSON 序列化错误
        model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=2026,
            n_jobs=-1,
            base_score=y.mean()
        )
        model.fit(X, y)

        # 3. 计算 SHAP 值
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            return X, shap_values
        except Exception as e:
            self.logger.error(f"SHAP 计算失败: {e}")
            return None, None

    def plot_comparative_impact(self, model_fan, model_judge):
        """
        绘制蝴蝶图 (Butterfly Chart)：直观对比评委与观众的偏好差异。
        """
        if model_fan is None or model_judge is None: return

        try:
            # 提取固定效应系数
            params_f = model_fan.params.drop("Intercept", errors='ignore')
            params_j = model_judge.params.drop("Intercept", errors='ignore')

            # 提取置信区间
            conf_f = model_fan.conf_int().drop("Intercept", errors='ignore')
            conf_j = model_judge.conf_int().drop("Intercept", errors='ignore')

            # 清洗特征名称
            factors = [idx.replace("C(celebrity_industry)[T.", "").replace("]", "") for idx in params_f.index]

            df_plot = pd.DataFrame({
                'Factor': factors,
                'Fan_Effect': params_f.values,
                'Judge_Effect': params_j.values,
                'Fan_Err': (conf_f[1] - conf_f[0]).values / 2,
                'Judge_Err': (conf_j[1] - conf_j[0]).values / 2
            })

            # 按粉丝影响排序
            df_plot['sort_key'] = df_plot['Fan_Effect'].abs()
            df_plot = df_plot.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)

            # 绘图
            fig, ax = plt.subplots(figsize=(12, 8))
            y = np.arange(len(df_plot))
            width = 0.35

            # 粉丝条形 (蓝色)
            ax.barh(y + width / 2, df_plot['Fan_Effect'], width, xerr=df_plot['Fan_Err'],
                    label='Fan Preference (Posterior)', color='#1f77b4', alpha=0.8, capsize=3)
            # 评委条形 (橙色)
            ax.barh(y - width / 2, df_plot['Judge_Effect'], width, xerr=df_plot['Judge_Err'],
                    label='Judge Preference (Technical)', color='#ff7f0e', alpha=0.8, capsize=3)

            ax.set_yticks(y)
            ax.set_yticklabels(df_plot['Factor'])
            ax.set_xlabel('Standardized Impact Size (Beta Coefficient)')
            ax.set_title('The Evaluation Gap: Meritocracy vs. Populism')
            ax.axvline(0, color='black', linewidth=0.8)
            ax.legend()
            ax.invert_yaxis()

            plt.tight_layout()
            plt.savefig(f"{self.fig_dir}/lmm_coefficient_contrast.png", dpi=300)
            plt.close()

            # 记录最大分歧点
            df_plot['Divergence'] = df_plot['Fan_Effect'] - df_plot['Judge_Effect']
            max_div = df_plot.loc[df_plot['Divergence'].abs().idxmax()]
            self.logger.info(f"最大偏好分歧点: {max_div['Factor']} (Diff: {max_div['Divergence']:.2f})")

        except Exception as e:
            self.logger.error(f"蝴蝶图绘制失败: {e}")

    def plot_shap_summary(self, X, shap_values):
        """
        绘制 SHAP 蜂群图 (Beeswarm Plot)。
        """
        if X is None or shap_values is None: return

        try:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, show=False, plot_type="dot", max_display=15)
            plt.title("Non-linear Drivers of Fan Votes (SHAP Attribution)")
            plt.tight_layout()
            plt.savefig(f"{self.fig_dir}/shap_beeswarm.png", dpi=300)
            plt.close()
        except Exception as e:
            self.logger.error(f"SHAP 绘图失败: {e}")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("CausalityEngine Ready.")