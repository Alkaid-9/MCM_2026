# ==============================================================================
# src/vis/architecture_viz.py
# Role: System Architecture Visualizer (The "Face" of the Project)
# Function: Generates the high-level BIO-Pareto Framework diagram (Figure 1).
# Fix: Resolved "Orthogonal edges do not handle edge labels" warning via 'xlabel'.
# Upgrade: Added Legend, High-Contrast Colors, and Structural Refinement.
# ==============================================================================
import os
import sys
from pathlib import Path

try:
    from graphviz import Digraph
except ImportError:
    print("[ERROR] graphviz 库未安装。请运行: pip install graphviz")
    sys.exit(1)

# --- 学术级配色方案 (Material Design & Nature Style) ---
COLORS = {
    'data_bg': '#E1F5FE', 'data_border': '#0277BD',  # 浅蓝/深蓝
    'calc_bg': '#FFEBEE', 'calc_border': '#C62828',  # 浅红/深红
    'rslt_bg': '#E8F5E9', 'rslt_border': '#2E7D32',  # 浅绿/深绿
    'app_bg': '#FFF3E0', 'app_border': '#EF6C00',  # 浅橙/深橙
    'edge_main': '#455A64', 'edge_feedback': '#EF6C00'  # 灰蓝/橙色
}


def draw_system_architecture(output_dir: str = "reports/figures"):
    """
    绘制 BIO-Pareto 框架架构图 (Figure 1)。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 初始化画布
    dot = Digraph('BIO_Pareto_Framework', comment='MCM 2026 System Architecture')

    # 全局排版设置 (工业级标准)
    dot.attr(rankdir='LR')  # 从左到右布局
    dot.attr(splines='ortho')  # 正交连线 (电路板风格)
    dot.attr(nodesep='0.6')  # 节点垂直间距
    dot.attr(ranksep='0.9')  # 层级水平间距
    dot.attr(dpi='300')  # 高分辨率输出
    dot.attr(fontname='Times-Roman')  # 学术衬线体
    dot.attr(compound='true')  # 允许子图间连线
    dot.attr(forcelabels='true')  # 强制显示 xlabel

    # 节点默认样式
    dot.attr('node', shape='box', style='filled,rounded',
             fontname='Helvetica', fontsize='12', penwidth='1.5')
    dot.attr('edge', fontname='Helvetica', fontsize='10', color=COLORS['edge_main'])

    # =========================================================
    # Layer 1: Data Forensics & ETL
    # =========================================================
    with dot.subgraph(name='cluster_0_data') as c:
        c.attr(label='Layer 1: Data Forensics & ETL', style='dashed',
               color=COLORS['data_border'], fontcolor=COLORS['data_border'])
        c.attr('node', fillcolor=COLORS['data_bg'], color=COLORS['data_border'], fontcolor='#000000')

        c.node('Bronze', 'Bronze Data\n(Raw CSV)', shape='cylinder')
        c.node('Silver', 'Silver Data\n(Robust Z-Score)', shape='cylinder')
        c.node('Gold', 'Gold Factors\n(Feature Eng.)', shape='cylinder', penwidth='2.0')

        # 内部流转 (使用 xlabel 解决 ortho 警告)
        c.edge('Bronze', 'Silver', xlabel=' Cleaning')
        c.edge('Silver', 'Gold', xlabel=' Aggregation')

    # =========================================================
    # Layer 2: The BIO Engine (HPC Core)
    # =========================================================
    with dot.subgraph(name='cluster_1_engine') as c:
        c.attr(label='Layer 2: The BIO Engine (C++ / 23 Cores)', style='bold',
               color=COLORS['calc_border'], fontcolor=COLORS['calc_border'])
        c.attr('node', fillcolor=COLORS['calc_bg'], color=COLORS['calc_border'], fontcolor='#000000')

        c.node('Bridge', 'Pybind11 Bridge\n(Zero-Copy)', shape='diamond')
        # [核心亮点]
        c.node('MCMC', 'Parallel MCMC Kernel\n(Dual-Averaging NUTS)',
               shape='component', style='filled,bold', fillcolor='#FFCDD2', penwidth='2.5')

        c.node('Likelihood', 'Energy Function\n(Soft-Rank)', shape='ellipse')
        c.node('Prior', 'Prior Field\n(Zipf Law)', shape='ellipse')

        # 内部流转
        c.edge('Bridge', 'MCMC', xlabel=' Spawn')
        c.edge('MCMC', 'Likelihood', dir='both', xlabel=' Eval', style='bold')
        c.edge('Prior', 'Likelihood', style='dotted')

    # =========================================================
    # Layer 3: Latent Knowledge (Output)
    # =========================================================
    with dot.subgraph(name='cluster_2_output') as c:
        c.attr(label='Layer 3: Latent Inference', style='dashed',
               color=COLORS['rslt_border'], fontcolor=COLORS['rslt_border'])
        c.attr('node', fillcolor=COLORS['rslt_bg'], color=COLORS['rslt_border'], fontcolor='#000000')

        # [核心产出]
        c.node('Platinum', 'Platinum Data\n(Latent Posteriors)', shape='cylinder', penwidth='2.5', fillcolor='#C8E6C9')
        c.node('Audit', 'Scientific Audit\n(Split-R-hat < 1.1)', shape='note')

        c.edge('Platinum', 'Audit', style='dotted', xlabel=' Verify')

    # =========================================================
    # Layer 4: Strategic Applications
    # =========================================================
    with dot.subgraph(name='cluster_3_apps') as c:
        c.attr(label='Layer 4: Strategic Applications', style='dashed',
               color=COLORS['app_border'], fontcolor=COLORS['app_border'])
        c.attr('node', fillcolor=COLORS['app_bg'], color=COLORS['app_border'], fontcolor='#000000', shape='tab')

        c.node('Task2', 'Task 2: Forensics\n(Multiverse Sim)')
        c.node('Task3', 'Task 3: Attribution\n(LMM + SHAP)')
        c.node('Task4', 'Task 4: Design\n(Pareto Opt)')

    # =========================================================
    # 全局连线 (Global Pipeline)
    # =========================================================
    # 使用 xlabel 替代 label 以适配 splines='ortho'

    # 1. ETL -> Engine
    dot.edge('Gold', 'Bridge', xlabel=' Tensor Map', color=COLORS['edge_main'])

    # 2. Engine -> Output
    dot.edge('MCMC', 'Platinum', xlabel=' 1M Samples', color=COLORS['edge_main'], penwidth='2.0')

    # 3. Output -> Applications
    dot.edge('Platinum', 'Task2')
    dot.edge('Platinum', 'Task3')
    dot.edge('Platinum', 'Task4')

    # 4. Feedback Loops (闭环逻辑)
    # 调整 port 位置 (e.g., Task4:s -> Task2:s) 优化布线
    dot.edge('Task4', 'Task2', style='dashed', constraint='false',
             color=COLORS['edge_feedback'], xlabel=' Robustness Check')

    # =========================================================
    # 图例 (Legend) - 这是一个不连接的子图，仅用于说明
    # =========================================================
    with dot.subgraph(name='cluster_legend') as c:
        c.attr(label='Legend', fontsize='10', style='solid', color='black')
        c.node('L_Data', 'Data Asset', shape='cylinder', width='0.8', fontsize='10')
        c.node('L_Proc', 'Process', shape='box', width='0.8', fontsize='10')
        c.node('L_Core', 'HPC Kernel', shape='component', width='0.8', fontsize='10')
        c.edge('L_Data', 'L_Proc', style='invis')  # 隐形线占位
        c.edge('L_Proc', 'L_Core', style='invis')

    # =========================================================
    # 渲染输出
    # =========================================================
    output_filename = 'framework_architecture'
    output_path = os.path.join(output_dir, output_filename)

    try:
        # 生成 PNG
        dot.render(output_path, format='png', cleanup=True)
        print(f"✅ [SUCCESS] 架构图生成成功: {output_path}.png")
        print(f"   (Fix: 使用 xlabel 解决了 Orthogonal 连线警告)")
    except Exception as e:
        print(f"❌ [ERROR] Graphviz 渲染失败: {e}")


if __name__ == '__main__':
    # 自动定位项目根目录
    project_root = Path(__file__).resolve().parent.parent.parent
    figures_dir = project_root / "reports/figures"

    draw_system_architecture(str(figures_dir))