# ==============================================================================
# src/vis/attribution_viz.py
# Role: Causal Attribution Visualization Engine
# Function: Visualizing the "Clash of Criteria" between Judges and Fans.
# Key Outputs: Coefficient Butterfly Plots, Dissonance Radars, ICC Variance Pies.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
import math

# 引入全局绘图风格配置 (Single Source of Truth for Aesthetics)
from src.utils.plotting import DWTSPlotter


class AttributionVisualizer:
    """
    归因可视化引擎：
    负责将 LMM (线性混合模型) 和统计归因的抽象数值，
    转化为直观展示“审美分歧”与“系统性偏见”的视觉证据。
    """

    def __init__(self, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("VIZ_ATTRIBUTION")
        self.fig_dir = fig_dir
        # 复用全局绘图器，确保字体和配色统一
        self.plotter = DWTSPlotter(output_dir=fig_dir)
        os.makedirs(self.fig_dir, exist_ok=True)

    def plot_lmm_butterfly(self, df_coeffs: pd.DataFrame):
        """
        【图表 1】系数蝴蝶图 (Butterfly Plot)
        物理意义：直观对比评委 (Merit) vs 观众 (Populism) 对同一特征的敏感度差异。
        学术亮点：自动标记符号相反的特征 (Cognitive Dissonance)。

        :param df_coeffs: 包含 columns ['Feature', 'Judge_Beta', 'Fan_Beta']
        """
        self.logger.info("绘制 LMM 系数蝴蝶图 (Butterfly Plot)...")

        # 1. 数据预处理：按观众系数绝对值排序，形成漏斗状视觉流
        if df_coeffs.empty: return
        df = df_coeffs.copy()
        df['abs_fan'] = df['Fan_Beta'].abs()
        df = df.sort_values('abs_fan', ascending=True).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 8))
        y_pos = np.arange(len(df))
        height = 0.35

        # 2. 绘制双向条形图
        # 左翼 (或上层)：观众偏好 (蓝色/冷色)
        rects1 = ax.barh(y_pos + height / 2, df['Fan_Beta'], height,
                         label='Public Sentiment (Fans)',
                         color=self.plotter.colors['fan'], alpha=0.9)

        # 右翼 (或下层)：评委偏好 (橙色/暖色)
        rects2 = ax.barh(y_pos - height / 2, df['Judge_Beta'], height,
                         label='Expert Quality (Judges)',
                         color=self.plotter.colors['judge'], alpha=0.9)

        # 3. 装饰坐标轴
        ax.set_yticks(y_pos)
        # 清洗特征名称，去除公式残留
        clean_labels = [l.replace('C(industry)[T.', '').replace(']', '') for l in df['Feature']]
        ax.set_yticklabels(clean_labels, fontsize=11)

        # 添加零轴分割线
        ax.axvline(0, color='black', linewidth=1.2, linestyle='-')
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)

        # 4. 【高阶功能】自动标注认知失调 (Dissonance Markers)
        # 如果两者符号相反，且绝对值均超过阈值，视为显著冲突
        for i, (j_val, f_val) in enumerate(zip(df['Judge_Beta'], df['Fan_Beta'])):
            if np.sign(j_val) != np.sign(f_val) and (abs(j_val) > 0.02 or abs(f_val) > 0.02):
                # 在较长的一侧标注警告符号
                target_x = max(j_val, f_val) if j_val > 0 else min(j_val, f_val)
                offset = 0.05 if target_x > 0 else -0.15
                ax.text(target_x + offset, i, "⚠ Clash",
                        fontsize=9, color='#d62728', va='center', fontweight='bold')

        plt.title("The Evaluation Gap: Coefficient Contrast (LMM Fixed Effects)", fontsize=15, pad=15)
        plt.xlabel("Standardized Impact (Beta Coefficient)\n← Negative Correlation | Positive Correlation →",
                   fontsize=12)
        plt.legend(loc='lower right', frameon=True, framealpha=0.9)

        self.plotter.save_figure("task3_lmm_butterfly.png")

    def plot_dissonance_radar(self, metrics: dict):
        """
        【图表 2】认知失调雷达图 (Cognitive Divergence Radar)
        物理意义：将评委和观众的偏好向量投影到极坐标。
        面积重合度 (IoU) 越低，说明审美标准越割裂。
        """
        self.logger.info("绘制认知失调雷达图...")

        # 1. 提取并归一化向量
        labels = metrics.get('features', [])
        # 简化标签
        labels = [l.replace('C(industry_group)[T.', '').replace(']', '') for l in labels]
        v_j = np.array(metrics.get('v_judge', []))
        v_f = np.array(metrics.get('v_fan', []))

        if len(labels) < 3:
            self.logger.warning("特征维度不足 (<3)，跳过雷达图绘制。")
            return

        # 使用绝对值归一化 (关注度权重)
        v_j_abs = np.abs(v_j)
        v_f_abs = np.abs(v_f)
        v_j_norm = v_j_abs / (v_j_abs.max() + 1e-9)
        v_f_norm = v_f_abs / (v_f_abs.max() + 1e-9)

        # 2. 闭环处理
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        v_j_norm = np.concatenate((v_j_norm, [v_j_norm[0]]))
        v_f_norm = np.concatenate((v_f_norm, [v_f_norm[0]]))
        angles += angles[:1]

        # 3. 极坐标绘图
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        # 评委区域
        ax.plot(angles, v_j_norm, color=self.plotter.colors['judge'], linewidth=2, label='Expert Consensus')
        ax.fill(angles, v_j_norm, color=self.plotter.colors['judge'], alpha=0.25)

        # 观众区域
        ax.plot(angles, v_f_norm, color=self.plotter.colors['fan'], linewidth=2, label='Public Sentiment')
        ax.fill(angles, v_f_norm, color=self.plotter.colors['fan'], alpha=0.25)

        # 4. 装饰
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=11)
        # 移除径向标签，保持整洁
        ax.set_yticklabels([])

        # 计算简单的重合度 (IoU Proxy) 用于标题展示
        intersection = np.minimum(v_j_norm, v_f_norm).sum()
        union = np.maximum(v_j_norm, v_f_norm).sum()
        iou = intersection / (union + 1e-9)

        plt.title(f"Cognitive Divergence Radar\n(Preference Overlap Index: {iou:.2f})", y=1.08, fontsize=14,
                  fontweight='bold')
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

        self.plotter.save_figure("task3_dissonance_radar.png")

    def plot_icc_decomposition(self, icc_judge: float, icc_fan: float):
        """
        【图表 3】方差分解饼图 (Variance Decomposition)
        物理意义：回答“舞伴有多重要”。比较技术分和观众票中，舞伴效应 (Partner Alpha) 的占比。
        """
        self.logger.info("绘制 ICC 方差分解图...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        # 配色：灰色代表选手自身实力 (Residual)，高亮色代表舞伴加成 (Partner Effect)
        colors_j = ['#e0e0e0', self.plotter.colors['judge']]
        colors_f = ['#e0e0e0', self.plotter.colors['fan']]

        explode = (0, 0.05)  # 突出显示舞伴效应

        # Subplot 1: Judge Score Breakdown
        ax1.pie([1 - icc_judge, icc_judge],
                labels=['Idiosyncratic Skill\n(Star Only)', 'Pro-Partner Effect\n(Alpha)'],
                autopct='%1.1f%%', startangle=90, colors=colors_j, explode=explode,
                textprops={'fontsize': 12}, shadow=True)
        ax1.set_title("Drivers of Judge Scores", fontsize=14, fontweight='bold', color=self.plotter.colors['judge'])

        # Subplot 2: Fan Vote Breakdown
        ax2.pie([1 - icc_fan, icc_fan],
                labels=['Star Charisma\n(Fame)', 'Pro-Partner Effect\n(Halo)'],
                autopct='%1.1f%%', startangle=90, colors=colors_f, explode=explode,
                textprops={'fontsize': 12}, shadow=True)
        ax2.set_title("Drivers of Fan Votes", fontsize=14, fontweight='bold', color=self.plotter.colors['fan'])

        plt.suptitle("The 'Pro-Dancer' Halo Effect: Variance Decomposition (ICC)", fontsize=16, y=0.95)
        self.plotter.save_figure("task3_icc_decomposition.png")