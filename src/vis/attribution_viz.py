"""
MCM 2026 Problem C: Causal Attribution Visualization Engine
Role: Visualizing the "Clash of Criteria" between Judges and Fans.
Key Outputs: Coefficient Butterfly Plots, SHAP Interaction Maps, Dissonance Radars.
Standard: High-DPI, Publication-Ready, Narrative-Driven.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import math

# 引入全局绘图风格配置
from src.utils.plotting import DWTSPlotter


class AttributionVisualizer:
    """
    归因可视化引擎：
    负责将 LMM 和 SHAP 的抽象数值转化为“审美分歧”的视觉证据。
    """

    def __init__(self, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("VIZ_ATTRIBUTION")
        self.fig_dir = fig_dir
        self.plotter = DWTSPlotter(output_dir=fig_dir)
        os.makedirs(self.fig_dir, exist_ok=True)

    def plot_lmm_butterfly(self, df_coeffs: pd.DataFrame):
        """
        【图表 1】系数蝴蝶图 (Butterfly Plot)
        物理意义：直观对比评委 vs 观众对同一特征（如年龄、舞伴、行业）的敏感度差异。
        输入 df_coeffs 需包含: 'Feature', 'Judge_Beta', 'Fan_Beta'
        """
        self.logger.info("绘制 LMM 系数蝴蝶图...")

        # 按粉丝系数绝对值排序，增强视觉冲击力
        df = df_coeffs.sort_values('Fan_Beta', ascending=True).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 8))

        y = np.arange(len(df))
        height = 0.35

        # 左翼：观众 (Fans)
        # 注意：为了做出“背对背”效果，通常一边取负，但这里我们用双条形图更直观
        # 我们采用交错条形图 (Staggered Bar)
        rects1 = ax.barh(y + height / 2, df['Fan_Beta'], height,
                         label='Public Sentiment (Fans)', color=self.plotter.colors['fan'], alpha=0.8)

        # 右翼：评委 (Judges)
        rects2 = ax.barh(y - height / 2, df['Judge_Beta'], height,
                         label='Expert Quality (Judges)', color=self.plotter.colors['judge'], alpha=0.8)

        # 装饰
        ax.set_yticks(y)
        ax.set_yticklabels(df['Feature'], fontsize=11)
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-')

        # 添加网格
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)

        # 标注显著性差异
        for i, (j_val, f_val) in enumerate(zip(df['Judge_Beta'], df['Fan_Beta'])):
            # 如果两者符号相反，或者差距极大，标注出来
            if np.sign(j_val) != np.sign(f_val) and abs(j_val) > 0.05:
                ax.text(max(j_val, f_val) + 0.1, i, "⚠ Dissonance",
                        fontsize=9, color='red', va='center')

        plt.title("The Evaluation Gap: Coefficient Contrast (LMM Fixed Effects)", fontsize=15, pad=15)
        plt.xlabel("Standardized Impact (Beta Coefficient)", fontsize=12)
        plt.legend(loc='lower right')

        self.plotter.save_figure("task3_lmm_butterfly.png")

    def plot_dissonance_radar(self, metrics: dict):
        """
        【图表 2】认知失调雷达图 (Cognitive Divergence Radar)
        物理意义：将评委和观众的偏好向量投影到极坐标，面积重合度越低，审美越割裂。
        """
        self.logger.info("绘制认知失调雷达图...")

        # 提取数据
        labels = metrics.get('features', [])
        # 简化标签名
        labels = [l.replace('C(industry_group)[T.', '').replace(']', '') for l in labels]

        v_j = np.array(metrics.get('v_judge', []))
        v_f = np.array(metrics.get('v_fan', []))

        if len(labels) < 3:
            self.logger.warning("特征数少于3，无法绘制雷达图。")
            return

        # 归一化到 [0, 1] 区间以便展示相对偏好强度
        # 使用 Softmax 或者 MinMax，这里用 Abs Max Scaling 保留方向性意义比较难，
        # 所以我们画的是“关注度权重” (绝对值)
        v_j_abs = np.abs(v_j)
        v_f_abs = np.abs(v_f)

        v_j_norm = v_j_abs / (v_j_abs.max() + 1e-9)
        v_f_norm = v_f_abs / (v_f_abs.max() + 1e-9)

        # 闭环
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        v_j_norm = np.concatenate((v_j_norm, [v_j_norm[0]]))
        v_f_norm = np.concatenate((v_f_norm, [v_f_norm[0]]))
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # 绘制评委
        ax.plot(angles, v_j_norm, color=self.plotter.colors['judge'], linewidth=2, label='Judge Focus')
        ax.fill(angles, v_j_norm, color=self.plotter.colors['judge'], alpha=0.25)

        # 绘制观众
        ax.plot(angles, v_f_norm, color=self.plotter.colors['fan'], linewidth=2, label='Fan Focus')
        ax.fill(angles, v_f_norm, color=self.plotter.colors['fan'], alpha=0.25)

        # 装饰
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)

        # 计算重合度 (IoU Proxy)
        intersection = np.minimum(v_j_norm, v_f_norm).sum()
        union = np.maximum(v_j_norm, v_f_norm).sum()
        iou = intersection / union

        plt.title(f"Preference Alignment Radar (Overlap Score: {iou:.2f})", y=1.08, fontsize=14)
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))

        self.plotter.save_figure("task3_dissonance_radar.png")

    def plot_shap_dependence(self, shap_values, X, feature_x='age', interaction_feature='industry'):
        """
        【图表 3】非线性交互依赖图 (SHAP Interaction Plot)
        物理意义：展示 Age 对票数的非线性影响（如 U 型曲线），以及行业如何调节这种影响。
        """
        self.logger.info(f"绘制 SHAP 依赖图 ({feature_x})...")

        try:
            import shap
        except ImportError:
            self.logger.warning("SHAP 库未安装，跳过依赖图绘制。")
            return

        plt.figure(figsize=(10, 7))

        # 这是一个 wrapper，SHAP 自带的绘图很难定制风格，我们尝试调整参数
        # 注意：这里传入的是 shap_values 对象
        shap.plots.scatter(shap_values[:, feature_x], color=shap_values[:, interaction_feature],
                           show=False, ax=plt.gca())

        plt.title(f"Non-Linear Effect of {feature_x.capitalize()} on Fan Support", fontsize=14)
        plt.ylabel(f"SHAP Value\n(Impact on Vote Share)", fontsize=12)
        plt.xlabel(f"{feature_x.capitalize()} (Standardized)", fontsize=12)
        plt.grid(True, alpha=0.2)

        self.plotter.save_figure(f"task3_shap_dependence_{feature_x}.png")

    def plot_icc_decomposition(self, icc_judge, icc_fan):
        """
        【图表 4】方差分解饼图 (Variance Decomposition)
        物理意义：回答“舞伴有多重要”。比较技术分和观众票中，舞伴效应（Partner Alpha）的占比。
        """
        self.logger.info("绘制 ICC 方差分解图...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        colors = ['#bdc3c7', self.plotter.colors['highlight']]  # 灰色 vs 高亮色

        # Judge ICC
        ax1.pie([1 - icc_judge, icc_judge], labels=['Idiosyncratic\n(Star Skill)', 'Partner Effect\n(Alpha)'],
                autopct='%1.1f%%', startangle=90, colors=colors, explode=(0, 0.05),
                textprops={'fontsize': 12})
        ax1.set_title("Judge Score Variance Sources", fontsize=14)

        # Fan ICC
        ax2.pie([1 - icc_fan, icc_fan], labels=['Idiosyncratic\n(Star Charisma)', 'Partner Effect\n(Halo)'],
                autopct='%1.1f%%', startangle=90, colors=colors, explode=(0, 0.05),
                textprops={'fontsize': 12})
        ax2.set_title("Fan Vote Variance Sources", fontsize=14)

        plt.suptitle("The 'Pro-Partner' Halo Effect: Merit vs. Popularity", fontsize=16)
        self.plotter.save_figure("task3_icc_decomposition.png")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    viz = AttributionVisualizer()

    # 1. Test Butterfly
    df_coeffs = pd.DataFrame({
        'Feature': ['Age', 'Male', 'Singer', 'Athlete'],
        'Judge_Beta': [-0.3, 0.1, 0.05, 0.4],
        'Fan_Beta': [0.2, -0.1, 0.5, 0.1]
    })
    viz.plot_lmm_butterfly(df_coeffs)

    # 2. Test Radar
    metrics = {
        'features': ['Age', 'Male', 'Singer', 'Athlete', 'Momentum'],
        'v_judge': [0.8, 0.2, 0.1, 0.9, 0.7],  # 重视技术指标
        'v_fan': [0.3, 0.4, 0.9, 0.5, 0.2]  # 重视娱乐指标
    }
    viz.plot_dissonance_radar(metrics)

    # 3. Test ICC
    viz.plot_icc_decomposition(0.15, 0.25)

    print("Attribution visuals generated.")