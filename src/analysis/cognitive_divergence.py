# ==============================================================================
# src/analysis/cognitive_divergence.py
# Role: Evaluative Heuristics Auditor (The "Clash of Criteria")
# Function: Quantifying the Merit-Popularity Gap via Cosine Similarity.
# Physics: Measuring the orthogonality between Expert and Public preference vectors.
# Standard: Industrial Robustness / Academic Visualization.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from src.utils.plotting import DWTSPlotter


class DivergenceAnalyzer:
    """
    认知背离分析器：
    利用线性代数方法，将 LMM 回归系数转化为高维偏好向量，
    计算其余弦距离以量化“认知失调” (Cognitive Dissonance)。
    """

    def __init__(self, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("DIVERGENCE_AUDIT")
        self.fig_dir = fig_dir
        # 复用全局绘图风格
        self.plotter = DWTSPlotter(output_dir=fig_dir)
        os.makedirs(self.fig_dir, exist_ok=True)

    def calculate_dissonance_index(self, model_j, model_f):
        """
        【数学核心】计算认知失调指数。
        Dissonance = 1 - CosineSimilarity(Beta_J, Beta_F)
        """
        if model_j is None or model_f is None:
            self.logger.warning("模型输入为空，无法计算认知失调。")
            return None

        self.logger.info("正在将回归系数映射为偏好向量...")

        # 1. 提取固定效应系数 (Beta Coefficients)
        # 排除 Intercept (截距) 和 Group Var (随机效应方差)
        beta_j = model_j.params.drop(['Intercept', 'Group Var'], errors='ignore')
        beta_f = model_f.params.drop(['Intercept', 'Group Var'], errors='ignore')

        # 2. 严格特征对齐 (Vector Alignment)
        # 确保两个向量在同一个基底上（处理某个模型可能剔除不显著特征的情况）
        common_features = beta_j.index.intersection(beta_f.index)

        if len(common_features) < 2:
            self.logger.warning("公共特征维度不足，无法构建向量空间。")
            return None

        v_j = beta_j[common_features].values
        v_f = beta_f[common_features].values

        # 3. 计算余弦相似度 (Cosine Similarity)
        # Formula: (A . B) / (||A|| * ||B||)
        dot_product = np.dot(v_j, v_f)
        norm_j = np.linalg.norm(v_j)
        norm_f = np.linalg.norm(v_f)

        # 防止零向量除法
        if norm_j == 0 or norm_f == 0:
            cosine_sim = 0.0
        else:
            cosine_sim = dot_product / (norm_j * norm_f)

        # 4. 定义失调指数 (Dissonance Index)
        # Range: [0, 2]. 0=Aligned, 1=Orthogonal (Unrelated), 2=Opposite
        dissonance_idx = 1 - cosine_sim

        # 5. 识别最大冲突点 (Hotspot Detection)
        # 寻找符号相反且模长最大的维度
        diff_vec = np.abs(v_j - v_f)
        # 增加符号惩罚：如果符号相反，差异加倍
        sign_conflict = np.sign(v_j) != np.sign(v_f)
        weighted_diff = diff_vec * (1 + sign_conflict.astype(int))

        max_conflict_idx = np.argmax(weighted_diff)
        conflict_feat = common_features[max_conflict_idx]

        # 清洗特征名用于展示
        conflict_feat_clean = conflict_feat.replace("C(celebrity_industry)[T.", "").replace("]", "")

        self.logger.info(f" 偏好对齐度 (Cosine Sim): {cosine_sim:.4f}")
        self.logger.info(f" 认知失调指数 (Dissonance): {dissonance_idx:.4f}")
        self.logger.info(f" 最大分歧维度: {conflict_feat_clean}")

        return {
            'cosine_sim': float(cosine_sim),
            'dissonance_idx': float(dissonance_idx),
            'conflict_feature': conflict_feat_clean,
            'features': list(common_features),
            'v_judge': v_j,
            'v_fan': v_f
        }

    def plot_preference_radar(self, metrics: dict):
        """
        绘制偏好雷达图 (Preference Projection Radar)。
        物理意义：直观展示评委和观众在不同维度上的“拉扯”。
        """
        if not metrics: return

        features = metrics['features']
        # 清洗标签
        labels = [f.replace("C(celebrity_industry)[T.", "").replace("]", "").replace("z_", "")
                  for f in features]

        v_j = metrics['v_judge']
        v_f = metrics['v_fan']

        # 归一化处理 (Scaling for Visualization)
        # 我们关注的是“相对形状”而非绝对大小，因此按绝对值最大值归一化
        max_val = max(np.max(np.abs(v_j)), np.max(np.abs(v_f))) + 1e-9
        v_j_norm = v_j / max_val
        v_f_norm = v_f / max_val

        # 构建雷达图坐标
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # 闭环
        v_j_plot = np.concatenate((v_j_norm, [v_j_norm[0]]))
        v_f_plot = np.concatenate((v_f_norm, [v_f_norm[0]]))
        angles += angles[:1]

        # 绘图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # 评委层 (Expert Layer)
        ax.plot(angles, v_j_plot, color=self.plotter.colors['judge'], linewidth=2, linestyle='-',
                label='Judge Preference')
        ax.fill(angles, v_j_plot, color=self.plotter.colors['judge'], alpha=0.1)

        # 观众层 (Populist Layer)
        ax.plot(angles, v_f_plot, color=self.plotter.colors['fan'], linewidth=2, linestyle='-', label='Fan Preference')
        ax.fill(angles, v_f_plot, color=self.plotter.colors['fan'], alpha=0.1)

        # 轴标签
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)

        # 移除径向刻度，避免视觉杂乱
        ax.set_yticklabels([])

        plt.title(f"Cognitive Divergence Radar\n(Dissonance Index: {metrics['dissonance_idx']:.2f})",
                  y=1.08, fontsize=14, fontweight='bold')

        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        save_path = os.path.join(self.fig_dir, "cognitive_divergence_radar.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"雷达图已保存: {save_path}")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Mock 数据测试绘图
    mock_metrics = {
        'dissonance_idx': 0.65,
        'features': ['Age', 'Industry_Music', 'Partner_Alpha', 'Momentum', 'Industry_Sport'],
        'v_judge': np.array([-0.8, 0.2, 0.5, 0.9, 0.4]),
        'v_fan': np.array([0.5, 0.8, 0.1, 0.2, 0.6])  # 故意制造冲突
    }
    analyzer = DivergenceAnalyzer()
    analyzer.plot_preference_radar(mock_metrics)