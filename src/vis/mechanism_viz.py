# ==============================================================================
# src/vis/mechanism_viz.py
# Role: Mechanism Forensics & Design Visualization
# Function: Visualizing the "Pareto Frontier" for Mechanism Optimization.
# Standard: High-DPI, Publication-Ready, Compact Layout.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import logging
import os
from matplotlib.patches import FancyArrowPatch


try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
except ImportError:
    print("Error: 'lifelines' library not found. Please run 'pip install lifelines'.")



# 引入全局绘图风格配置 (假设已有，若无则使用默认)
try:
    from src.utils.plotting import DWTSPlotter
except ImportError:
    class DWTSPlotter:
        def __init__(self, output_dir): self.output_dir = output_dir
        def save_figure(self, filename):
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()
        colors = {'fan': '#1f77b4', 'judge': '#ff7f0e', 'highlight': '#d62728', 'grid': '#e0e0e0'}

class MechanismVisualizer:
    """
    机制可视化引擎：
    负责将‘机制设计’的优化结果转化为视觉证据。
    """
    def __init__(self, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("VIZ_MECHANISM")
        self.fig_dir = fig_dir
        self.plotter = DWTSPlotter(output_dir=fig_dir)
        os.makedirs(self.fig_dir, exist_ok=True)

        # 配置学术字体
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'mathtext.fontset': 'stix'
        })

    def plot_pareto_frontier(self, optimizer_results: pd.DataFrame, baseline_metrics: dict = None):
        """
        【图表 12】帕累托最优前沿图 (The 'Money Plot')
        学术意义：证明 DAW 机制是在公平性与参与度之间权衡的全局最优解。

        :param optimizer_results: 包含 ['efficiency', 'equity', 't0', 'k'] 的 DataFrame
        :param baseline_metrics: 包含历史基准点 {'RANK': {...}, 'PERCENT': {...}}
        """
        self.logger.info("绘制帕累托优化前沿图...")

        # 1. 准备画布
        fig, ax = plt.subplots(figsize=(9, 6)) # 紧凑尺寸，适合插入论文

        # 2. 绘制 Grid Search 散点云 (搜索空间)
        # 颜色映射 t0 (权力移交点), 大小映射 k (切换速率)
        # 使用 Alpha 通道让密集区域有层次感
        sc = ax.scatter(
            optimizer_results['efficiency'],
            optimizer_results['equity'],
            c=optimizer_results['t0'],
            cmap='viridis',
            s=optimizer_results['k'] * 8 + 10, # 动态大小
            alpha=0.6,
            edgecolors='none',
            label='Search Space (Grid)',
            zorder=2
        )

        # 3. 计算并绘制帕累托前沿 (Pareto Frontier)
        # 逻辑：按 Efficiency 排序，计算 Equity 的累积最大值，形成包络线
        df_sorted = optimizer_results.sort_values('efficiency')
        frontier_equity = df_sorted['equity'].cummax()
        # 筛选出位于前沿上的点
        frontier_mask = df_sorted['equity'] == frontier_equity
        frontier_df = df_sorted[frontier_mask]

        ax.plot(
            frontier_df['efficiency'],
            frontier_df['equity'],
            color='#2c3e50',
            linestyle='--',
            linewidth=1.5,
            alpha=0.8,
            label='Pareto Frontier',
            zorder=3
        )

        # 4. 标记“乌托邦点” (Utopia Point) - 理论最大值
        ax.scatter(1.0, 1.0, color='gold', s=150, marker='P', edgecolors='black', linewidth=1.5,
                   label='Utopia Point (Theoretical Max)', zorder=10)

        # 5. 寻找并标记“最优解” (Optimal DAW)
        # 定义：距离乌托邦点欧氏距离最近的点
        dist = np.sqrt((1 - optimizer_results['equity'])**2 + (1 - optimizer_results['efficiency'])**2)
        best_idx = dist.idxmin()
        best_pt = optimizer_results.loc[best_idx]

        ax.scatter(
            best_pt['efficiency'],
            best_pt['equity'],
            color=self.plotter.colors['highlight'],
            s=250,
            marker='*',
            edgecolors='white',
            linewidth=1.2,
            label=f'Optimal DAW ($t_0={best_pt["t0"]:.2f}, k={best_pt["k"]:.0f}$)',
            zorder=10
        )

        # 6. 标记历史基准 (Baselines)
        if baseline_metrics:
            # Rank System
            if 'RANK' in baseline_metrics:
                b = baseline_metrics['RANK']
                ax.scatter(b['efficiency'], b['equity'], color=self.plotter.colors['fan'], s=180, marker='s',
                           edgecolors='black', linewidth=2, label='Rank System (Historical)', zorder=20)  # 加大并描黑

            if 'PERCENT' in baseline_metrics:
                b = baseline_metrics['PERCENT']
                ax.scatter(b['efficiency'], b['equity'], color=self.plotter.colors['judge'], s=200, marker='^',
                           edgecolors='black', linewidth=2, label='Percent System (Historical)', zorder=20)

        ax.plot([best_pt['efficiency'], 1.0], [best_pt['equity'], 1.0],
                color='black', linestyle=':', linewidth=1.2, alpha=0.5, zorder=1)
        ax.text((best_pt['efficiency'] + 1.0) / 2 + 0.02, (best_pt['equity'] + 1.0) / 2,
                "Min. Distance", fontsize=8, rotation=35, color='gray')
        # 7. 美化与标注
        # 颜色条
        cbar = plt.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(r'Transition Midpoint ($t_0$)', rotation=270, labelpad=15)
        cbar.ax.tick_params(labelsize=9)

        # 坐标轴
        ax.set_xlabel(r'Engagement Metric ($\rho_{Fan}$)', fontsize=12, fontweight='bold')
        ax.set_ylabel(r'Fairness Metric ($\rho_{Merit}$)', fontsize=12, fontweight='bold')

        # 动态调整范围，留出一点 buffer
        x_min, x_max = df_sorted['efficiency'].min(), 1.02
        y_min, y_max = df_sorted['equity'].min(), 1.02
        # 确保基准点也在视野内
        if baseline_metrics:
            x_min = min(x_min, baseline_metrics.get('RANK', {}).get('efficiency', 1), baseline_metrics.get('PERCENT', {}).get('efficiency', 1))
            y_min = min(y_min, baseline_metrics.get('RANK', {}).get('equity', 1), baseline_metrics.get('PERCENT', {}).get('equity', 1))

        ax.set_xlim(x_min - 0.05, x_max)
        ax.set_ylim(y_min - 0.05, y_max)

        # 网格
        ax.grid(True, linestyle=':', alpha=0.5, color='gray')

        # 箭头标注 (避免文字重叠)
        # 指向最优解
        ax.annotate(
            'Recommended\nConfiguration',
            xy=(best_pt['efficiency'], best_pt['equity']),
            xytext=(best_pt['efficiency'] - 0.15, best_pt['equity'] - 0.1),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color='#333'),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=self.plotter.colors['highlight'], alpha=0.9),
            fontsize=9, color=self.plotter.colors['highlight'], fontweight='bold'
        )

        # 标题
        ax.set_title("Multi-Objective Optimization: The Equity-Engagement Trade-off", fontsize=14, pad=15)

        # 图例 (放在左下角或不遮挡的地方)
        ax.legend(loc='lower left', frameon=True, framealpha=0.95, fontsize=9, edgecolor='gray')

        # 保存
        self.plotter.save_figure("task4_pareto_frontier.png")

    def plot_daw_weight_trajectory(self, k: float, t0: float, total_weeks: int = 10, w_min: float = 0.2,
                                   w_max: float = 0.9):
        """
        【图表 13】DAW 动态权重移交曲线 (Dynamic Power Transfer Trajectory)
        学术意义：直观展示机制如何实现从“民意引流”到“专业控盘”的平滑过渡。

        :param k: Sigmoid 斜率 (切换速度)
        :param t0: 切换中点 (归一化时间 0-1)
        :param total_weeks: 赛季总周数
        :param w_min: 评委权重下限
        :param w_max: 评委权重上限
        """
        self.logger.info(f"绘制 DAW 权力移交曲线 (k={k}, t0={t0})...")

        # 1. 数据生成
        weeks = np.linspace(1, total_weeks, 200)  # 使用更密集的点使曲线平滑
        progress = (weeks - 1) / (total_weeks - 1)

        # Sigmoid 核心逻辑 (需与 DAWEngine 保持数学一致)
        # logit = k * (t - t0) * 10
        def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

        logit = k * (progress - t0) * 10.0
        weights = w_min + (w_max - w_min) * sigmoid(logit)

        # 2. 准备画布
        fig, ax = plt.subplots(figsize=(8, 4.5))  # 宽屏紧凑比例

        # 3. 绘制主曲线
        # 使用深红色代表“权力曲线”
        ax.plot(weeks, weights, color='#B22222', linewidth=4, label=r'Judge Weight $w_J(t)$', zorder=5)

        # 4. 绘制辅助元素
        # 转换点垂直线
        transition_week = 1 + t0 * (total_weeks - 1)
        ax.axvline(transition_week, color='black', linestyle='--', linewidth=1, alpha=0.6,
                   label='Transition Pivot ($t_0$)')

        # 平衡线 (如果 w_min/max 对称) 或者 0.5 线
        # ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)

        # 5. 区域填充 (Highlight Phases)
        # 左侧：流量主导区 (Populism / Fan Zone) - 蓝色调
        ax.fill_between(weeks, w_min, weights, where=(weeks <= transition_week),
                        color=self.plotter.colors['fan'], alpha=0.15)
        # 右侧：精英主导区 (Meritocracy / Expert Zone) - 橙色调
        ax.fill_between(weeks, w_min, weights, where=(weeks > transition_week),
                        color=self.plotter.colors['judge'], alpha=0.15)

        # 6. 关键文本标注 (Text Annotations)
        # 避免文字重叠：放置在空白区域

        # Phase 1 Label
        ax.text(1.5, w_min + 0.05,
                "Phase I: Fan Engagement\n(Traffic Driven)",
                fontsize=10, color=self.plotter.colors['fan'], fontweight='bold',
                va='bottom', ha='left')

        # Phase 2 Label
        ax.text(total_weeks - 0.5, w_max - 0.05,
                "Phase II: Meritocracy\n(Skill Driven)",
                fontsize=10, color=self.plotter.colors['judge'], fontweight='bold',
                va='top', ha='right')

        pivot_y = (w_min + w_max) / 2  # 曲线的纵坐标中点

        ax.annotate(
            'Power Handover\n(Mechanism Pivot)',

            # --- 关键坐标设置 ---
            # xy: 箭头尖端指哪里？(这里设为转折点)
            xy=(transition_week, pivot_y),

            # xytext: 文字放在哪里？(相对于转折点：向右偏移 +1.5 周，向下偏移 -0.12 权重)
            # 如果文字挡住了曲线，增大 +1.5；如果想文字更高，减小 -0.12
            xytext=(transition_week + 1.5, pivot_y - 0.12),

            # --- 箭头样式设置 ---
            arrowprops=dict(
                arrowstyle="-|>",  # 样式：实心三角箭头 (比 -> 更专业)
                color='#333333',  # 颜色：深灰
                lw=2,  # 粗细：2

                # 弯曲程度：arc3 是贝塞尔曲线
                # rad: 正值向上弯，负值向下弯，0 为直线。
                # 现在的设置是微向“下”凹，产生一种托举感
                connectionstyle="arc3,rad=0.2",

                shrinkB=5  # 尖端缩进：让箭头不要紧贴着红线，留出 5px 缝隙
            ),

            # --- 文字样式 ---
            fontsize=11,
            fontweight='bold',
            color='#111111',
            ha='left',  # 文字左对齐
            va='center',  # 文字垂直居中
            zorder=50
        )



        # 7. 坐标轴与装饰
        ax.set_xlabel("Competition Week (Time)", fontsize=11, fontweight='bold')
        ax.set_ylabel(r"Judge Weight Impact ($w_J$)", fontsize=11, fontweight='bold')
        ax.set_title(r"Dynamic Mechanism: The Governance of Power Transfer", fontsize=13, pad=12)

        # X轴刻度：显示整数周
        ax.set_xticks(np.arange(1, total_weeks + 1))

        # Y轴范围
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.2, 0.5, 0.8, 1.0])
        ax.set_yticklabels(['0%', r'20% ($w_{min}$)', '50%', r'80% ($w_{max}$)', '100%'], fontweight='bold')

        # 图例
        ax.legend(loc='upper left', frameon=True, framealpha=0.95, fontsize=9)

        # 网格
        ax.grid(True, linestyle=':', alpha=0.4)

        # 8. 保存
        self.plotter.save_figure("task4_daw_trajectory.png")

    def plot_survival_comparison(self, survival_df: pd.DataFrame):
        """
        【图表 8】技术流选手的生存曲线 (Kaplan-Meier Survival Curve)

        布局优化版：
        1. 图例 (Legend) -> 移至左下角 (利用早期生存率高的留白区)。
        2. 统计指标 (Log-Rank) -> 移至右上角 (利用后期生存率下降后的留白区)。
        3. 视觉平衡：形成对角线构图，突出中间的 "Merit Protection Gap"。
        """
        self.logger.info("绘制 O 奖级生存分析图 (布局优化版)...")

        # 1. 锁定学术绘图风格
        with plt.rc_context({
            'font.family': 'serif',
            'font.serif': ['Times New Roman', 'DejaVu Serif'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'legend.fontsize': 10,
            'xtick.direction': 'in',
            'ytick.direction': 'in'
        }):
            fig, ax = plt.subplots(figsize=(8, 5))  # 保持紧凑尺寸

            kmf = KaplanMeierFitter()

            # 定义视觉策略
            styles = {
                'Rank System': {'color': '#005a9e', 'fmt': '-', 'label': 'Rank System (Ordinal)'},
                'Percent System': {'color': '#c0392b', 'fmt': '--', 'label': 'Percent System (Cardinal)'}
            }

            T_data, E_data = {}, {}

            # 2. 循环拟合与绘制
            for regime in ['Rank System', 'Percent System']:
                mask = survival_df['regime'] == regime
                if not mask.any(): continue

                T = survival_df.loc[mask, 'duration']
                E = survival_df.loc[mask, 'observed_event']
                T_data[regime] = T
                E_data[regime] = E

                kmf.fit(T, event_observed=E, label=styles[regime]['label'])

                # 绘制曲线
                kmf.plot_survival_function(
                    ax=ax,
                    color=styles[regime]['color'],
                    linestyle=styles[regime]['fmt'],
                    linewidth=2.5,  # 略微加粗主线
                    ci_show=True,
                    ci_alpha=0.12,  # 增加一点透明度饱和度
                    show_censors=True,
                    censor_styles={'marker': '|', 'ms': 6, 'mew': 1.2}  # 使用竖线标记截断，更整洁
                )

            # 3. Log-Rank 检验 -> 移至【右上角】
            if len(T_data) == 2:
                results = logrank_test(
                    T_data['Rank System'], T_data['Percent System'],
                    event_observed_A=E_data['Rank System'], event_observed_B=E_data['Percent System']
                )
                p_val = results.p_value

                if p_val < 0.001:
                    p_text = r"$\bf{Log\text{-}Rank\ Test:}$ $p < 0.001^{***}$"
                else:
                    p_text = r"$\bf{Log\text{-}Rank\ Test:}$ $p = {:.3f}$".format(p_val)

                # 锚点设在右上 (0.96, 0.94)
                ax.text(0.96, 0.94, p_text, transform=ax.transAxes,
                        fontsize=11,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='#dddddd', alpha=0.9))

            # 4. Merit Protection Gap 标注 -> 保持在中间
            ax.annotate('', xy=(6, 0.41), xytext=(6, 0.92),
                        arrowprops=dict(arrowstyle='<->', color='#2c3e50', lw=1.5))
            ax.text(6.2, 0.68, "Merit Protection\nGap", color='#2c3e50',
                    fontsize=10, fontstyle='italic', ha='left', va='center')

            # 5. 图例 -> 移至【左下角】
            # frameon=True 加上半透明背景，防止与网格线混杂
            ax.legend(loc='lower left', frameon=True, framealpha=0.95,
                      edgecolor='#dddddd', fontsize=11, borderpad=0.6)

            # 6. 学术化修饰
            ax.set_title(r"Survival Analysis of 'Meritocratic Martyrs' (Top 30\% Skill)", pad=15, fontweight='bold')
            ax.set_xlabel("Competition Duration (Weeks)")
            ax.set_ylabel(r"Survival Probability $\hat{S}(t)$")

            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, survival_df['duration'].max() * 1.02)

            # 极简边框 (Tufte Style)
            sns.despine(trim=True, offset=5)
            ax.grid(True, linestyle='--', alpha=0.4, color='#bbbbbb')  # 网格线稍微明显一点点

            # 7. 保存
            plt.tight_layout()
            self.plotter.save_figure("task2_survival_km_curve.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- 1. 模拟数据：Rank 机制下存活更久 (扩大样本量以获得更平滑的曲线) ---
    np.random.seed(2026)
    n = 200  # 样本量加倍，曲线更漂亮

    # Rank: 技术流选手的避风港 (平均存活 8.5 周，方差小)
    t_rank = np.random.normal(8.5, 1.5, n).clip(1, 10)
    e_rank = np.random.choice([0, 1], n, p=[0.4, 0.6])  # 0=截断(决赛/退赛)

    # Percent: 流量绞肉机 (平均存活 4.5 周，方差大，因为受刷票影响)
    t_pct = np.random.normal(4.5, 3.0, n).clip(1, 10)
    e_pct = np.random.choice([0, 1], n, p=[0.1, 0.9])  # 大部分都被淘汰了

    df_mock = pd.concat([
        pd.DataFrame({'duration': t_rank, 'observed_event': e_rank, 'regime': 'Rank System'}),
        pd.DataFrame({'duration': t_pct, 'observed_event': e_pct, 'regime': 'Percent System'})
    ])

    # --- 2. 实例化与绘图 ---
    viz = MechanismVisualizer()

    # 绘制生存分析图 (图表 8)
    viz.plot_survival_comparison(df_mock)

    # 绘制 DAW 权力移交曲线 (图表 13) - [关键修改]
    # k=8.0: 呈现完美的 S 型，暗示“制度变迁”
    # t0=0.55: 黄金分割点，平衡娱乐与专业
    viz.plot_daw_weight_trajectory(k=8.0, t0=0.55)

    print("\n[Test Complete]")
    print(f"1. Survival Curve: reports/figures/task2_survival_km_curve.png")
    print(f"2. DAW Trajectory: reports/figures/task4_daw_trajectory.png")