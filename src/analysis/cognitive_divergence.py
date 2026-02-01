# ==============================================================================
# src/analysis/cognitive_divergence.py
# Role: Evaluative Heuristics Auditor (The "Clash of Criteria")
# Function: Quantifying the Merit-Popularity Gap via Cosine Similarity
# Method: Vector Projection of Standardized Coefficients (Betas)
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os


class DivergenceAnalyzer:
    """
    认知背离分析器：
    利用线性代数方法，量化“技术评价体系”与“人气倾向体系”之间的鸿沟。
    """

    def __init__(self, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("DIVERGENCE_AUDIT")
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)
        # 学术绘图配置
        plt.rcParams['font.family'] = 'serif'
        sns.set_context("paper", font_scale=1.4)

    def calculate_dissonance_index(self, model_j, model_f):
        """
        【数学核心】计算认知失调指数 (Dissonance Index)。
        逻辑：将评委偏好和观众偏好视为高维特征空间中的两个向量。
        """
        if model_j is None or model_f is None:
            return None

        # 1. 提取标准化回归系数 (Beta Coefficients)
        # 剔除截距项和随机效应项，只关注‘业务特征’的权重
        beta_j = model_j.params.drop(['Intercept', 'Group Var'], errors='ignore')
        beta_f = model_f.params.drop(['Intercept', 'Group Var'], errors='ignore')

        # 2. 严格对齐特征空间 (Feature Space Alignment)
        common_features = beta_j.index.intersection(beta_f.index)
        v_j = beta_j[common_features].values
        v_f = beta_f[common_features].values

        # 3. 计算余弦相似度 (Cosine Similarity)
        # Cos(theta) = (A·B) / (||A||*||B||)
        dot_product = np.dot(v_j, v_f)
        norm_j = np.linalg.norm(v_j)
        norm_f = np.linalg.norm(v_f)

        cosine_sim = dot_product / (norm_j * norm_f + 1e-9)

        # 4. 定义认知失调指数 (Index = 1 - Similarity)
        # Index = 0: 完美同步; Index = 1: 完全无关; Index = 2: 极度对立
        dissonance_idx = 1 - cosine_sim

        self.logger.info(f">>> 审美对齐度 (Cosine Similarity): {cosine_sim:.4f}")
        self.logger.info(f">>> 认知失调指数 (Dissonance Index): {dissonance_idx:.4f}")

        return {
            'cosine_sim': cosine_sim,
            'dissonance_idx': dissonance_idx,
            'features': list(common_features),
            'v_judge': v_j,
            'v_fan': v_f
        }

    def plot_preference_radar(self, results_dict):
        """
        绘制雷达图 (Radar Chart)：直观展示评委与观众的“权力足迹”重合度。
        物理意义：展示系统性偏置发生在哪些维度。
        """
        if not results_dict: return

        labels = [f.replace("C(industry)[T.", "").replace("]", "") for f in results_dict['features']]
        v_j = results_dict['v_judge']
        v_f = results_dict['v_fan']

        # 归一化以便展示形状
        v_j_norm = v_j / (np.abs(v_j).max() + 1e-9)
        v_f_norm = v_f / (np.abs(v_f).max() + 1e-9)

        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # 闭合图形
        v_j_norm = np.concatenate((v_j_norm, [v_j_norm[0]]))
        v_f_norm = np.concatenate((v_f_norm, [v_f_norm[0]]))
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # 评委多边形
        ax.fill(angles, v_j_norm, color='#ff7f0e', alpha=0.25, label='Expert Consensus (Judges)')
        ax.plot(angles, v_j_norm, color='#ff7f0e', linewidth=2)

        # 观众多边形
        ax.fill(angles, v_f_norm, color='#1f77b4', alpha=0.25, label='Public Sentiment (Fans)')
        ax.plot(angles, v_f_norm, color='#1f77b4', linewidth=2)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)
        plt.title("Cognitive Divergence Radar: The Merit-Popularity Gap", y=1.1, fontsize=15)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        path = os.path.join(self.fig_dir, "cognitive_divergence_radar.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"雷达对比图已保存: {path}")


# ------------------------------------------------------------------------------
# 量化‘最大背离特征’
# ------------------------------------------------------------------------------
def identify_conflict_hotspot(results_dict):
    """
    寻找导致分歧最大的特征。
    用于回答 Task 3 中关于‘哪些因素影响方式不一致’的问题。
    """
    diff = np.abs(results_dict['v_judge'] - results_dict['v_fan'])
    hotspot_idx = np.argmax(diff)
    return results_dict['features'][hotspot_idx]