# ==============================================================================
# src/vis/attribution_viz.py
# Role: Causal Attribution Visualization Engine
# Function: Visualizing the "Clash of Criteria" and Non-linear Drivers.
# Standard: High-DPI, Publication-Ready, No-Overlap Layout.
# ==============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import logging
import os

# 引入全局绘图风格
try:
    from src.utils.plotting import DWTSPlotter
except ImportError:
    # Fallback for standalone testing
    class DWTSPlotter:
        def __init__(self, output_dir):
            self.output_dir = output_dir
            self.colors = {'fan': '#1f77b4', 'judge': '#ff7f0e', 'neutral': '#7f7f7f'}

        def save_figure(self, filename):
            plt.savefig(os.path.join(self.output_dir, filename), dpi=300, bbox_inches='tight')
            plt.close()


class AttributionVisualizer:
    """
    归因可视化引擎：
    负责将 LMM 和 SHAP 的抽象统计结果转化为“社会学证据”。
    """

    def __init__(self, fig_dir: str = "reports/figures/"):
        self.logger = logging.getLogger("VIZ_ATTRIBUTION")
        self.fig_dir = fig_dir
        self.plotter = DWTSPlotter(output_dir=fig_dir)
        os.makedirs(self.fig_dir, exist_ok=True)

        # 配置学术字体 (Times New Roman)
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'mathtext.fontset': 'stix'
        })

    def plot_lmm_butterfly(self, df_coeffs: pd.DataFrame):
        """
        【图表 9】系数蝴蝶图 (Butterfly Bar Chart)
        学术意义：量化“认知失调”(Cognitive Dissonance)。
        展示精英 (Judges) 与大众 (Fans) 对同一特征权重的分歧。

        :param df_coeffs: DataFrame ['Feature', 'Judge_Beta', 'Fan_Beta', 'p_value']
        """
        self.logger.info("绘制 LMM 系数蝴蝶图...")

        # 1. 数据准备
        # 按粉丝系数排序，制造“漏斗”视觉效果
        df = df_coeffs.sort_values('Fan_Beta', ascending=True).reset_index(drop=True)
        y = np.arange(len(df))
        height = 0.4

        # 2. 准备画布
        fig, ax = plt.subplots(figsize=(10, 8))

        # 3. 绘制条形 (左侧 Fan 为负值显示，右侧 Judge 为正值)
        # 注意：Fan_Beta 取负值是为了向左延伸，标签后续修正
        rects_fan = ax.barh(y, -df['Fan_Beta'].abs(), height, label='Popularity (Fans)',
                            color=self.plotter.colors['fan'], alpha=0.8, edgecolor='white')
        rects_judge = ax.barh(y, df['Judge_Beta'].abs(), height, label='Meritocracy (Judges)',
                              color=self.plotter.colors['judge'], alpha=0.8, edgecolor='white')

        # 4. 中轴与装饰
        ax.axvline(0, color='black', linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(df['Feature'], fontsize=11)

        # 5. X轴刻度修正 (去除负号)
        ticks = ax.get_xticks()
        ax.set_xticklabels([f"{abs(t):.1f}" for t in ticks])
        ax.set_xlabel(r"Normalized Impact Magnitude ($|\beta|$)", fontsize=12, fontweight='bold')

        # 6. 标注“认知失调” (Dissonance Markers)
        # 逻辑：如果符号相反 (Sign Flip)，则标记冲突
        for i, row in df.iterrows():
            # 判断符号是否相反 (且系数足够大，忽略噪音)
            if (np.sign(row['Fan_Beta']) != np.sign(row['Judge_Beta'])) and \
                    (abs(row['Fan_Beta']) > 0.05) and (abs(row['Judge_Beta']) > 0.05):
                # 在条形图较短的一侧外标注
                target_x = max(abs(row['Fan_Beta']), abs(row['Judge_Beta'])) + 0.05
                ax.text(target_x, i, "⚡Clash", ha='left', va='center',
                        color='#d62728', fontsize=9, fontweight='bold', style='italic')

        # 7. 标题与图例
        ax.set_title("The Evaluation Gap: Elites vs. The Crowd", fontsize=16, pad=20)
        # 自定义图例位置
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=2, frameon=False)

        # 网格
        ax.grid(True, axis='x', linestyle=':', alpha=0.4)

        # 去除边框
        sns.despine(left=True, bottom=False)

        # 保存
        self.plotter.save_figure("task3_lmm_butterfly.png")

    def plot_shap_summary_proxy(self, shap_df: pd.DataFrame):
        """
        【图表 10】SHAP 蜂群图 (Beeswarm Proxy)
        学术意义：展示非线性驱动因素。线性模型只能看均值，SHAP 能看分布。

        :param shap_df: DataFrame ['Feature', 'SHAP_Value', 'Feature_Value_Norm']
                        其中 Feature_Value_Norm 是归一化到 0-1 的原始特征值 (用于颜色映射)
        """
        self.logger.info("绘制 SHAP 非线性蜂群图...")

        # 1. 准备画布
        fig, ax = plt.subplots(figsize=(10, 6))

        # 2. 绘制蜂群 (使用 Seaborn Stripplot)
        # jitter=True 让点分散开，形成“蜂群”效果
        # hue 映射特征值高低 (Blue->Red)
        sns.stripplot(
            data=shap_df,
            x='SHAP_Value',
            y='Feature',
            hue='Feature_Value_Norm',
            palette='coolwarm',
            jitter=0.25,
            size=4,
            alpha=0.7,
            ax=ax,
            edgecolor='none',
            zorder=2
        )

        # 3. 辅助线
        ax.axvline(0, color='black', linewidth=0.8, linestyle='-', zorder=1)

        # 4. 装饰
        ax.set_xlabel("SHAP Value (Impact on Fan Vote)", fontsize=12, fontweight='bold')
        ax.set_ylabel("")
        ax.set_title("Non-linear Drivers of Success: Feature Impact Distribution", fontsize=14, pad=15)

        # 5. 处理图例 (替换为 Colorbar)
        # 移除 seaborn 默认图例
        ax.get_legend().remove()

        # 手动添加 Colorbar
        norm = plt.Normalize(0, 1)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, aspect=30, pad=0.02)
        cbar.set_label("Feature Value (Low $\\to$ High)", rotation=270, labelpad=15)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low', 'High'])

        # 网格
        ax.grid(True, axis='x', linestyle=':', alpha=0.4)

        self.plotter.save_figure("task3_shap_beeswarm.png")

    def plot_icc_decomposition(self, icc_judge: float, icc_fan: float):
        """
        【图表 11】ICC 方差分解图 (Nested Donut Charts)
        学术意义：量化“舞伴效应”(Partner Effect) 的占比。

        :param icc_judge: 评委分模型的 ICC
        :param icc_fan: 粉丝票模型的 ICC
        """
        self.logger.info("绘制 ICC 方差分解图...")

        # 1. 准备画布 (左右两个子图)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        # 定义绘制单个甜甜圈的函数
        def draw_donut(ax, icc, color_main, title):
            # 数据: [舞伴效应, 选手自身/噪音]
            sizes = [icc, 1 - icc]
            labels = [f'Partner Halo\n{icc:.1%}', f'Star/Residual\n{1 - icc:.1%}']
            colors = [color_main, '#e0e0e0']  # 高亮舞伴效应，其他置灰
            explode = (0.05, 0)  # 炸开舞伴部分

            # 绘制饼图
            wedges, texts, autotexts = ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct='',  # 不使用默认百分比，手动控制位置
                startangle=90,
                pctdistance=0.85,
                explode=explode,
                wedgeprops=dict(width=0.4, edgecolor='w')  # width=0.4 变成甜甜圈
            )

            # 中心文字 (大号显示 ICC)
            ax.text(0, 0, f"ICC\n{icc:.2f}", ha='center', va='center', fontsize=14, fontweight='bold')

            # 优化标签样式
            for text in texts:
                text.set_fontsize(10)
                text.set_color('#333')

            ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        # 2. 绘制评委模型 (Judge)
        draw_donut(ax1, icc_judge, self.plotter.colors['judge'], "Judge Scores\n(Technical Metric)")

        # 3. 绘制粉丝模型 (Fan)
        draw_donut(ax2, icc_fan, self.plotter.colors['fan'], "Fan Votes\n(Popularity Metric)")

        # 4. 全局标题
        plt.suptitle("The 'Pro-Partner' Halo Effect: Variance Decomposition", fontsize=15, y=1.05)

        self.plotter.save_figure("task3_icc_decomposition.png")


# --- 单元测试 (生成 Mock 数据验证绘图) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    viz = AttributionVisualizer()

    # 1. Test Butterfly
    df_bf = pd.DataFrame({
        'Feature': ['Technique', 'Difficulty', 'Chemistry', 'Age', 'Reality Star', 'Country Music'],
        'Judge_Beta': [0.8, 0.7, 0.4, -0.2, -0.1, -0.3],
        'Fan_Beta': [0.2, 0.1, 0.5, 0.1, 0.6, 0.8]
    })
    viz.plot_lmm_butterfly(df_bf)

    # 2. Test SHAP Beeswarm
    # 模拟数据：Feature_Value_Norm 0-1, SHAP 随之变化
    n_samples = 300
    features = ['Age', 'Partner_Alpha', 'Technique']
    records = []
    for f in features:
        vals = np.random.rand(n_samples)  # Feature Values 0-1
        # 模拟不同关系：Age 是倒U型，Partner 是正相关
        if f == 'Age':
            shaps = -4 * (vals - 0.5) ** 2 + 0.5 + np.random.normal(0, 0.1, n_samples)
        elif f == 'Partner_Alpha':
            shaps = 2 * vals - 1 + np.random.normal(0, 0.1, n_samples)
        else:
            shaps = np.random.normal(0, 0.2, n_samples)

        for v, s in zip(vals, shaps):
            records.append({'Feature': f, 'Feature_Value_Norm': v, 'SHAP_Value': s})

    viz.plot_shap_summary_proxy(pd.DataFrame(records))

    # 3. Test ICC
    viz.plot_icc_decomposition(0.15, 0.08)  # Judge ICC=0.15, Fan ICC=0.08

    print("Test Complete. Check reports/figures/")