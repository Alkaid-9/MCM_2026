# ==============================================================================
# src/vis/sensitivity.py
# Role: Mechanism Stability & Robustness Visualization (Figure 6)
# Function: Proving the "Low-Pass Filter" hypothesis via Monte Carlo stress tests.
# Aesthetics: Nature/Science Journal Style, High Contrast, Non-overlapping.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from src.utils.plotting import DWTSPlotter


class SensitivityVisualizer:
    """
    鲁棒性可视化专家：
    负责绘制 Figure 6，通过噪声注入实验实证不同裁决机制的结构稳定性。
    """

    def __init__(self, output_dir: str = "reports/figures/"):
        self.output_dir = output_dir
        self.plotter = DWTSPlotter(output_dir=output_dir)
        self.logger = logging.getLogger("VIZ_SENSITIVITY")
        os.makedirs(self.output_dir, exist_ok=True)

        # 全局字体强化
        plt.rcParams['font.weight'] = 'bold'
        plt.rcParams['axes.labelweight'] = 'bold'

    def plot_stability_curve(self, df_sens: pd.DataFrame):
        """
        【图表 6】机制稳定性曲线 (The "Low-Pass Filter" Proof)

        X轴: 噪声强度 sigma (粉丝投票的非理性程度)
        Y轴: 冠军翻转率 (Flip Rate / 系统失效概率)
        """
        self.logger.info("绘制机制稳定性对比曲线 (Figure 6)...")

        # 1. 设置画布 (8x6 比例，适合论文紧凑排版)
        fig, ax = plt.subplots(figsize=(8, 6))

        # 2. 绘制 Percent System 曲线 (高灵敏度/高风险)
        # 使用红色（DarkRed），虚线，暗示不稳定
        ax.plot(df_sens['noise_level'], df_sens['flip_rate_percent'],
                color='#B22222', linestyle='--', linewidth=2.5,
                marker='s', markersize=6, label='Percent System (Signal Amplifier)',
                zorder=3)

        # 3. 绘制 Rank System 曲线 (低灵敏度/稳健)
        # 使用深蓝（MidnightBlue），实线，暗示稳健
        ax.plot(df_sens['noise_level'], df_sens['flip_rate_rank'],
                color='#191970', linestyle='-', linewidth=3,
                marker='o', markersize=7, label='Rank System (Low-Pass Filter)',
                zorder=4)

        # 4. 填充 "Stability Gap" (鲁棒性溢价区域)
        ax.fill_between(df_sens['noise_level'],
                        df_sens['flip_rate_rank'],
                        df_sens['flip_rate_percent'],
                        color='#708090', alpha=0.15,
                        label='Robustness Gain Layer', zorder=1)

        # ==================================================================
        # 5. 核心学术标注 (防止重叠算法)
        # ==================================================================

        ax.annotate('Signal Amplification:\nHigh Volatility',
                    xy=(0.20, 0.45),  # 箭头指向 Percent 线上的一个具体点
                    xytext=(0.24, 0.46),  # 文字框稍微往上、往右提，避开线条
                    arrowprops=dict(
                        arrowstyle="->",  # 使用标准箭头样式
                        color='#B22222',  # 箭头颜色与 Percent 线一致，或者用 'black'
                        lw=2,  # 增加线宽确保可见
                        connectionstyle="arc3,rad=-0.1"  # 微微向外弯曲，避开数据点
                    ),
                    fontsize=10, fontweight='bold', color='#B22222',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#B22222", alpha=0.9))

        # B. 标注低通滤波效应 (调整了指向位置)
        ax.annotate('Noise Attenuation:\nStructural Resilience',
                    xy=(0.18, 0.13),  # 箭头指向 Rank 线
                    xytext=(0.19, 0.04),  # 文字框下移，利用底部空白区
                    arrowprops=dict(
                        arrowstyle="->",
                        color='#191970',  # 箭头颜色与 Rank 线一致
                        lw=2,
                        connectionstyle="arc3,rad=0.1"  # 微微向内弯曲
                    ),
                    fontsize=10, fontweight='bold', color='#191970',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#191970", alpha=0.8))
        # ==================================================================
        # 6. 坐标轴与排版美化
        # ==================================================================
        ax.set_title("Figure 6: Mechanism Stability Curve under Fan-Vote Noise",
                     fontsize=13, pad=15, fontweight='bold')
        ax.set_xlabel(r"Noise Intensity ($\sigma$ of Public Sentiment)", fontsize=11)
        ax.set_ylabel("Winner Flip Rate (Outcome Reversal Prob.)", fontsize=11)

        # Y轴显示为百分比
        import matplotlib.ticker as mtick
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

        # 限制坐标轴范围，确保视觉聚焦
        ax.set_ylim(-0.02, 0.60)
        ax.set_xlim(0, df_sens['noise_level'].max() * 1.05)

        # 网格辅助线
        ax.grid(True, linestyle='--', alpha=0.4, zorder=0)

        # 去除上方和右侧边框
        sns.despine()

        # 图例：放在左上角，分两行显示以节省宽度
        ax.legend(loc='upper left', frameon=True, framealpha=1.0, fontsize=9, ncol=1)

        # 7. 保存
        save_path = os.path.join(self.output_dir, "task2_stability_curve.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.info(f"机制稳定性曲线已生成: {save_path}")


# --- 单元测试 (Mock Data) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 1. 扩展数据点，使曲线更平滑，且范围覆盖坐标轴限制
    # noise 从 0 到 0.35，覆盖 xlim
    noise = np.linspace(0, 0.35, 20)

    # 2. 构造符合 "Signal Amplifier" (Percent) 的数据
    # 物理特征：线性高灵敏度，并在高噪时饱和
    # 目标：在 x=0.20 时，y 约为 0.45 (对齐你的箭头标注)
    # y = 2.25 * x
    flip_pct = 2.25 * noise
    # 增加一点随机扰动模拟 Monte Carlo 实验的真实感
    np.random.seed(42)
    flip_pct += np.random.normal(0, 0.01, len(noise))
    # 截断以适配 ylim (0.60)
    flip_pct = np.clip(flip_pct, 0, 0.58)

    # 3. 构造符合 "Low-Pass Filter" (Rank) 的数据
    # 物理特征：低噪声下有"死区" (Deadband)，高噪声下缓慢上升
    # 目标：在 x=0.18 时，y 约为 0.13 (对齐你的箭头标注)
    # y = 0.8 * (x - 0.02)
    flip_rank = 0.8 * (noise - 0.02)
    flip_rank = np.where(flip_rank < 0, 0, flip_rank)  # 模拟滤波器的阈值效应
    flip_rank += np.random.normal(0, 0.005, len(noise))  # 较小的扰动
    flip_rank = np.clip(flip_rank, 0, 0.25)

    df_mock = pd.DataFrame({
        'noise_level': noise,
        'flip_rate_percent': flip_pct,
        'flip_rate_rank': flip_rank
    })

    print(f"Mock Data Generated: {len(df_mock)} points")

    # 4. 执行绘图
    viz = SensitivityVisualizer()
    viz.plot_stability_curve(df_mock)

    print("\n[Test Complete]")
    print("请检查生成的图片，确认箭头是否完美指向曲线：")
    print("reports/figures/task2_stability_curve.png")