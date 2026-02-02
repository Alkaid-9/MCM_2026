# ==============================================================================
# src/vis/architecture_viz.py
# Role: System Architecture Visualization Engine (Refined for O-Prize)
# Function: Generating the Figure 1 BIO-Pareto Framework diagram.
# Aesthetics: High-Contrast, Bold Fonts, Compact Layout, No Overlaps.
# ==============================================================================

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging
import os


class ArchitectureVisualizer:
    """
    架构绘图引擎 (终极优化版)：
    解决重叠问题，增强字体粗细，紧凑排版，符合顶刊印刷标准。
    """

    def __init__(self, output_dir: str = "reports/figures/"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = logging.getLogger("VIZ_ARCH")

        # --- 学术级配色方案 (高对比度 / 深邃蓝系) ---
        self.colors = {
            'core_fill': '#F0F4F8',  # 极浅蓝 (内核背景)
            'core_stroke': '#003366',  # MidnightBlue (内核边框 - 极深)
            'data_fill': '#FFFFFF',  # 纯白 (数据流背景，突出内容)
            'data_stroke': '#4A6572',  # SlateGray (数据流边框 - 加深)
            'task_fill': '#FFF9F5',  # 暖白 (分析任务背景)
            'task_stroke': '#8B4513',  # SaddleBrown (分析任务边框 - 加深)
            'platinum_fill': '#E6E6FA',  # 薰衣草紫 (Platinum 特殊色)
            'platinum_stroke': '#483D8B',  # DarkSlateBlue
            'text_main': '#000000',  # 纯黑 (确保清晰度)
            'text_sub': '#333333',  # 深灰
            'arrow': '#2F4F4F'  # 深绿灰
        }

        # 字体配置 (强制粗体，大字号)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.sans-serif'] = ['Times New Roman', 'DejaVu Serif']
        plt.rcParams['font.weight'] = 'bold'

    def _draw_box(self, ax, xy, width, height, color_fill, color_stroke, label, sublabel=None, style='solid', lw=2.0,
                  zorder=10, fontsize=12):
        """绘制带圆角的方框 (增强版)"""
        box = patches.FancyBboxPatch(
            xy, width, height,
            boxstyle="round,pad=0.1,rounding_size=0.2",
            ec=color_stroke,
            fc=color_fill,
            lw=lw,
            linestyle=style,
            zorder=zorder,
            alpha=1.0  # 不透明，防止线条干扰
        )
        ax.add_patch(box)

        # 主标签 (居中)
        cx = xy[0] + width / 2
        cy = xy[1] + height / 2 + (0.15 if sublabel else 0)
        ax.text(cx, cy, label, ha='center', va='center', fontsize=fontsize, fontweight='bold',
                color=self.colors['text_main'], zorder=zorder + 1)

        # 副标签 (居中，下方)
        if sublabel:
            ax.text(cx, cy - 0.4, sublabel, ha='center', va='center', fontsize=fontsize - 3, style='italic',
                    fontweight='normal', color=self.colors['text_sub'], zorder=zorder + 1)

        return box

    def _draw_arrow(self, ax, start, end, style="->", rad=0.0, lw=2.0, label=None):
        """绘制粗壮的连接线"""
        arrow = patches.FancyArrowPatch(
            start, end,
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle=style,
            color=self.colors['arrow'],
            lw=lw,
            mutation_scale=15,  # 箭头更大
            zorder=5
        )
        ax.add_patch(arrow)

        # 如果有标签，画在箭头中间
        if label:
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            # 根据弧度调整标签位置
            if rad != 0: mid_x += 0.2
            ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=10,
                    style='italic', color='#333333', backgroundcolor='white', zorder=6)

    def draw_architecture_diagram(self):
        """
        主绘图逻辑：紧凑型布局，无重叠
        """
        self.logger.info("正在绘制优化版架构图...")

        # 调整画布比例，更宽一点以容纳文字
        fig, ax = plt.subplots(figsize=(13, 9))
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 10.5)
        ax.axis('off')

        # ======================================================================
        # 1. Phase I: Data Curation (Top Layer) - Y: 8.0 - 10.0
        # ======================================================================
        # 外框
        p1_rect = patches.FancyBboxPatch(
            (0.5, 7.8), 11.0, 2.0,
            boxstyle="round,pad=0.1", ec=self.colors['data_stroke'], fc='none',
            linestyle='--', lw=1.5, zorder=0
        )
        ax.add_patch(p1_rect)
        ax.text(0.8, 9.5, "Phase I: Data Curation Pipeline (ETL)", fontsize=13, fontweight='bold',
                color=self.colors['data_stroke'], backgroundcolor='white')

        # 内部模块 (Bronze -> Silver -> Gold)
        box_w, box_h = 2.4, 1.2
        y_p1 = 8.1
        self._draw_box(ax, (1.0, y_p1), box_w, box_h, '#F5F5F5', self.colors['data_stroke'], "Bronze Layer",
                       "Raw CSV\n(Unstructured)")
        self._draw_box(ax, (4.8, y_p1), box_w, box_h, '#E8E8E8', self.colors['data_stroke'], "Silver Layer",
                       "Robust Z-Score\n(Time-Aligned)")
        self._draw_box(ax, (8.6, y_p1), box_w, box_h, '#DCDCDC', self.colors['data_stroke'], "Gold Layer",
                       "Factor Library\n(Partner Alpha)")

        # 箭头 I
        self._draw_arrow(ax, (3.5, y_p1 + 0.6), (4.7, y_p1 + 0.6))
        self._draw_arrow(ax, (7.3, y_p1 + 0.6), (8.5, y_p1 + 0.6))

        # ======================================================================
        # 2. Phase II: BIO Engine (Middle Layer) - Y: 4.0 - 7.5
        # ======================================================================
        # 外框 (核心引擎，加重)
        p2_rect = patches.FancyBboxPatch(
            (2.0, 4.0), 8.0, 3.2,
            boxstyle="round,pad=0.1", ec=self.colors['core_stroke'], fc=self.colors['core_fill'],
            linestyle='-', lw=3.0, zorder=0
        )
        ax.add_patch(p2_rect)
        ax.text(2.3, 6.9, "Phase II: BIO Inference Engine (C++ Kernel)", fontsize=14, fontweight='bold',
                color=self.colors['core_stroke'])

        # MCMC Sampler (左侧核心)
        self._draw_box(ax, (2.5, 4.5), 3.0, 2.0, '#FFFFFF', self.colors['core_stroke'], "MCMC Sampler",
                       "Parallel HMC\n(23 Cores / OpenMP)", lw=2.5, fontsize=13)

        # Prior & Likelihood (右侧组件)
        self._draw_box(ax, (6.5, 5.6), 2.8, 0.8, '#FFFFFF', self.colors['core_stroke'], "Prior", "Zipf-Dirichlet Field")
        self._draw_box(ax, (6.5, 4.4), 2.8, 0.8, '#FFFFFF', self.colors['core_stroke'], "Likelihood",
                       "Quadratic Hinge Loss")

        # 内部箭头
        self._draw_arrow(ax, (6.4, 6.0), (5.6, 5.8), style="-")  # Prior -> Sampler
        self._draw_arrow(ax, (6.4, 4.8), (5.6, 5.0), style="-")  # Likelihood -> Sampler

        # Tech Stack 标注 (右侧空白处)
        ax.text(10.2, 6.5, "Technology Stack:", fontsize=10, fontweight='bold')
        ax.text(10.2, 6.1, "• Eigen3 (SIMD)", fontsize=10)
        ax.text(10.2, 5.7, "• PyBind11 (Zero-Copy)", fontsize=10)
        ax.text(10.2, 5.3, "• Numba (JIT)", fontsize=10)

        # 跨层箭头: Gold -> Prior (特征矩阵注入)
        self._draw_arrow(ax, (9.8, 8.0), (8.0, 6.5), rad=-0.3, label="Feature Matrix")

        # ======================================================================
        # 3. Phase III: Analytics & Design (Bottom Layer) - Y: 0.2 - 3.5
        # ======================================================================
        # 外框
        p3_rect = patches.FancyBboxPatch(
            (0.5, 0.2), 11.0, 3.2,
            boxstyle="round,pad=0.1", ec=self.colors['task_stroke'], fc='none',
            linestyle='--', lw=1.5, zorder=0
        )
        ax.add_patch(p3_rect)
        # 标题放左上角，避开 Platinum Layer
        ax.text(0.8, 3.1, "Phase III: Forensics, Attribution & Design", fontsize=13, fontweight='bold',
                color=self.colors['task_stroke'], backgroundcolor='white')

        # Platinum Layer (衔接点 - 位于 Phase III 内部顶端居中)
        # 关键修改：下移，避免和 Phase III 标题重叠
        plat_x, plat_y = 4.5, 2.3
        self._draw_box(ax, (plat_x, plat_y), 3.0, 1.0, self.colors['platinum_fill'], self.colors['platinum_stroke'],
                       "Platinum Layer", "Posterior Distribution\n(Latent Votes)", lw=2.0)

        # 三大任务 (底部并排)
        task_y = 0.5
        task_w, task_h = 3.0, 1.2
        self._draw_box(ax, (1.0, task_y), task_w, task_h, self.colors['task_fill'], self.colors['task_stroke'],
                       "Task 2: Forensics", "Multiverse Sim\n(Rank vs Percent)")
        self._draw_box(ax, (4.5, task_y), task_w, task_h, self.colors['task_fill'], self.colors['task_stroke'],
                       "Task 3: Attribution", "LMM + SHAP\n(Causal Inference)")
        self._draw_box(ax, (8.0, task_y), task_w, task_h, self.colors['task_fill'], self.colors['task_stroke'],
                       "Task 4: Design", "Pareto Optimizer\n(DAW Mechanism)")

        # 跨层箭头: Sampler -> Platinum
        # 关键修改：直接向下指，文本放在旁边
        self._draw_arrow(ax, (4.0, 4.4), (5.5, 3.4), label="MCMC Trace\n(Split R-hat < 1.1)")

        # 分发箭头: Platinum -> Tasks
        plat_bottom_center = (plat_x + 1.5, plat_y)
        self._draw_arrow(ax, plat_bottom_center, (2.5, task_y + 1.2), rad=0.1)  # To Task 2
        self._draw_arrow(ax, plat_bottom_center, (6.0, task_y + 1.2), style="->")  # To Task 3 (Straight)
        self._draw_arrow(ax, plat_bottom_center, (9.5, task_y + 1.2), rad=-0.1)  # To Task 4

        # ======================================================================
        # 底部标题
        # ======================================================================
        # 留白给 Figure 标题 (Word 中插入 caption，这里不画图内标题，或者画在最底端)
        # ax.text(6.0, -0.5, "Figure 1: The BIO-Pareto Framework Architecture", ha='center', fontsize=14, fontweight='bold')

        # 保存
        save_path = os.path.join(self.output_dir, "architecture_viz.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close()
        self.logger.info(f"架构图优化版已生成: {save_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    viz = ArchitectureVisualizer()
    viz.draw_architecture_diagram()