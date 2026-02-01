# ==============================================================================
# src/analysis/causality.py
# Role: Final Causal Attribution Engine (Mission Critical v5.0)
# Function: Decomposing "Star Power" vs. "Meritocracy"
# Strategy: Robust LMM + Hybrid Feature Importance (SHAP/Native Fallback)
# ==============================================================================

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
import logging
import warnings


# --- 0. 绘图环境防御性初始化 ---
def setup_robust_style():
    """配置学术绘图风格，自动降级字体以防止报错"""
    # 优先列表：Times -> DejaVu (Linux标配) -> Sans
    font_candidates = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif', 'Arial']
    system_fonts = {f.name for f in fm.fontManager.ttflist}

    selected_font = 'sans-serif'
    for f in font_candidates:
        if f in system_fonts:
            selected_font = f
            break

    plt.rcParams['font.family'] = selected_font if selected_font != 'sans-serif' else 'sans-serif'
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号乱码
    sns.set_context("paper", font_scale=1.4)
    return selected_font


class CausalityEngine:
    def __init__(self, df_platinum: pd.DataFrame, figures_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("CAUSALITY_ENGINE")
        self.df = df_platinum.copy()
        self.fig_dir = figures_dir
        os.makedirs(self.fig_dir, exist_ok=True)

        font = setup_robust_style()
        self.logger.info(f"绘图引擎就绪 (Font: {font})")

        self._preprocess_data()

    def _preprocess_data(self):
        """清洗与标准化：为回归分析准备数据"""
        # 1. 填补分类缺失
        self.df['celebrity_industry'] = self.df['celebrity_industry'].fillna('Other').astype(str)
        self.df['ballroom_partner'] = self.df['ballroom_partner'].fillna('Unknown').astype(str)

        # 2. 提取有效样本 (Task 1 反演结果 + 原始打分均存在)
        valid_mask = self.df['est_fan_vote_mu'].notna() & self.df['week_avg_score'].notna()
        self.df = self.df[valid_mask].copy()

        if len(self.df) < 10:
            self.logger.error("有效样本不足 10 条，归因分析将无法进行！")
            return

        # 3. Z-Score 标准化 (让 Beta 系数可比)
        scaler = StandardScaler()
        # 目标变量
        self.df['z_fan_vote'] = scaler.fit_transform(self.df[['est_fan_vote_mu']])
        self.df['z_judge_score'] = scaler.fit_transform(self.df[['week_avg_score']])
        # 自变量
        self.df['celebrity_age_during_season'] = self.df['celebrity_age_during_season'].fillna(
            self.df['celebrity_age_during_season'].median())
        self.df['z_age'] = scaler.fit_transform(self.df[['celebrity_age_during_season']])

    def run_lmm_comparison(self):
        """
        [模型 A] 混合效应模型 (LMM)
        对比：评委 vs 粉丝 对同一特征（年龄、行业）的敏感度差异。
        """
        self.logger.info(">>> 启动 LMM 对照实验 (Merit vs. Popularity)...")

        if len(self.df) < 50:
            self.logger.warning("样本量过少 (<50)，跳过 LMM。")
            return None, None

        # 公式：固定效应(年龄+行业) + 随机效应(舞伴)
        # 物理意义：舞伴带来的影响被视为随机截距，剩下的行业系数就是纯粹的观众偏好
        formula = " ~ z_age + C(celebrity_industry)"

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                # 1. 粉丝模型
                model_fan = smf.mixedlm("z_fan_vote" + formula, self.df,
                                        groups=self.df["ballroom_partner"]).fit(method='nm', maxiter=2000)
                # 2. 评委模型
                model_judge = smf.mixedlm("z_judge_score" + formula, self.df,
                                          groups=self.df["ballroom_partner"]).fit(method='nm', maxiter=2000)

            self.logger.info("LMM 双向拟合成功。")
            return model_fan, model_judge
        except Exception as e:
            self.logger.error(f"LMM 拟合失败: {str(e)}")
            return None, None

    def run_shap_attribution(self):
        """
        [模型 B] 特征重要性归因 (Hybrid Mode)
        策略：尝试 SHAP -> 失败则使用原生 Gain 重要性。绝不崩溃。
        """
        self.logger.info(">>> 启动特征归因分析...")

        # 1. 准备特征矩阵
        feats = ['celebrity_age_during_season', 'partner_alpha', 'score_delta', 'week_avg_score']
        X_num = self.df[feats].fillna(0)
        X_cat = pd.get_dummies(self.df[['celebrity_industry']], prefix='ind', dtype=int)
        X = pd.concat([X_num, X_cat], axis=1)
        y = self.df['est_fan_vote_mu']

        if len(X) == 0: return None, None

        # 2. 训练 XGBoost (作为 SHAP 的代理模型)
        model = xgb.XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            random_state=2026, n_jobs=1  # 限制单核防止死锁
        )
        model.fit(X, y)

        # 3. 尝试 SHAP (Plan A)
        try:
            import shap
            # 使用最通用的 KernelExplainer (慢但兼容性好)
            # 为了速度，只用 K-Means 聚类后的背景
            background = shap.kmeans(X, 10)
            explainer = shap.KernelExplainer(model.predict, background)

            # 只计算前 100 个样本用于绘图 (足够了)
            X_sample = X.iloc[:100]
            shap_values = explainer.shap_values(X_sample)

            # 成功则返回 SHAP 数据
            self.logger.info("SHAP (Kernel) 计算成功。")
            return X_sample, shap_values

        except Exception as e:
            self.logger.warning(f"SHAP 模块不可用或崩溃 ({e})，切换至原生特征重要性。")

            # --- Plan B: 原生 Feature Importance ---
            # 提取 Gain (信息增益)
            importance = pd.Series(model.feature_importances_, index=X.columns)
            return X, importance

    def plot_comparative_impact(self, model_fan, model_judge):
        """绘制 LMM 蝴蝶图"""
        if not model_fan or not model_judge: return

        try:
            # 提取参数，处理可能的索引不一致
            params_f = model_fan.params.drop("Intercept", errors='ignore')
            params_j = model_judge.params.drop("Intercept", errors='ignore')

            # 取交集
            common_idx = params_f.index.intersection(params_j.index)
            p_f = params_f[common_idx]
            p_j = params_j[common_idx]

            # 简化标签
            labels = [i.replace("C(celebrity_industry)[T.", "").replace("]", "") for i in common_idx]

            df_plot = pd.DataFrame({
                'Factor': labels, 'Fan': p_f.values, 'Judge': p_j.values
            }).sort_values('Fan')

            # 绘图
            fig, ax = plt.subplots(figsize=(10, 8))
            y = np.arange(len(df_plot))

            ax.barh(y + 0.2, df_plot['Fan'], 0.4, label='Fan Preference', color='#1f77b4')
            ax.barh(y - 0.2, df_plot['Judge'], 0.4, label='Judge Preference', color='#ff7f0e')

            ax.set_yticks(y)
            ax.set_yticklabels(df_plot['Factor'])
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.set_xlabel("Standardized Impact Coefficient (Beta)")
            ax.set_title("Meritocracy vs. Populism: Evaluation Criteria Gap")
            ax.legend()

            plt.tight_layout()
            plt.savefig(f"{self.fig_dir}/lmm_contrast.png", dpi=300)
            plt.close()
            self.logger.info("LMM 对比图已保存。")
        except Exception as e:
            self.logger.error(f"LMM 绘图失败: {e}")

    def plot_shap_summary(self, X, attrib_data):
        """绘制归因图 (兼容 SHAP 和 Native Importance)"""
        if attrib_data is None: return

        try:
            plt.figure(figsize=(10, 8))

            if isinstance(attrib_data, pd.Series):
                # Native Importance 模式
                top_20 = attrib_data.sort_values().tail(20)
                top_20.plot(kind='barh', color='teal', alpha=0.8)
                plt.xlabel("Relative Importance (Gain)")
                plt.title("Key Drivers of Fan Votes (XGBoost)")
            else:
                # SHAP 模式 (attrib_data is numpy array)
                import shap
                shap.summary_plot(attrib_data, X, show=False)
                plt.title("Non-linear Drivers of Fan Votes (SHAP)")

            plt.tight_layout()
            plt.savefig(f"{self.fig_dir}/feature_attribution_summary.png", dpi=300)
            plt.close()
            self.logger.info("特征归因图已保存。")
        except Exception as e:
            self.logger.error(f"归因绘图失败: {e}")