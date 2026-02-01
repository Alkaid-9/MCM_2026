# ==============================================================================
# src/utils/exporter.py
# Role: Automated Research Deliverable Exporter (v6.1 - Interface Fix)
# Function:
#   1. Generating Code Appendix with syntax highlighting & structure tree.
#   2. Converting Pandas DataFrames into IEEE-style LaTeX tables.
# Fix: Renamed CodeExporter -> MCMProjectExporter to align with main.py.
# ==============================================================================

import os
import time
import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional


class TableExporter:
    """
    [学术工具箱] LaTeX 表格生成器。
    将 Pandas 数据框转化为符合顶刊标准的“三线表”。
    """

    @staticmethod
    def df_to_latex(df: pd.DataFrame,
                    caption: str,
                    label: str,
                    alignment: str = None) -> str:
        """
        核心转译函数。
        """
        if df.empty: return "% Empty DataFrame"

        # 1. 基础格式化
        latex_str = df.to_latex(
            index=False,
            column_format=alignment if alignment else 'l' + 'c' * (len(df.columns) - 1),
            float_format="%.4f",
            escape=True
        )

        # 2. 注入 Booktabs 风格
        latex_str = latex_str.replace('\\toprule', '\\toprule\n')
        latex_str = latex_str.replace('\\midrule', '\\midrule\n')
        latex_str = latex_str.replace('\\bottomrule', '\\bottomrule\n')

        # 3. 包装 Table 环境
        wrapper = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            latex_str.strip(),
            "\\end{table}"
        ]

        return "\n".join(wrapper)


class MCMProjectExporter:
    """
    [工程展示] 代码附录生成器。
    (原 CodeExporter，重命名以匹配 main.py 接口)
    """

    # 忽略列表 (噪音过滤)
    INCLUDE_EXTS = {'.py', '.cpp', '.hpp', '.h', '.yaml', '.cmake', '.txt'}
    EXCLUDE_DIRS = {
        '__pycache__', '.git', '.idea', '.vscode', 'venv', 'env',
        'build', 'bin', 'lib', 'obj', 'data', 'logs', 'reports', 'notebooks'
    }
    EXCLUDE_FILES = {
        '.env', '.DS_Store', 'requirements.txt', 'README.md',
        'appendix_code.tex', 'mcm_core_lib.so'
    }

    # LaTeX 导言区 (代码高亮配置)
    LATEX_PREAMBLE = r"""
% --- Automated Code Appendix Style ---
\usepackage{listings}
\usepackage{xcolor}
\usepackage{booktabs} 

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.97,0.97,0.97}

\lstdefinestyle{mcmstyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{blue}\bfseries,
    numberstyle=\tiny\color{codegray},
    stringstyle=\color{codepurple},
    basicstyle=\ttfamily\scriptsize,
    breakatwhitespace=false,         
    breaklines=true,                 
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2,
    frame=single,
    rulecolor=\color{codegray}
}
\lstset{style=mcmstyle}
% -------------------------------------
"""

    def __init__(self, project_root: str):
        # 兼容 Path 对象或字符串
        self.root = Path(project_root).resolve()
        self.scanned_files = []
        self.md_path = self.root / "Code_Appendix.md"
        self.tex_path = self.root / "appendix_code.tex"
        self.logger = logging.getLogger("CODE_EXPORTER")

    def _generate_tree(self, dir_path: Path, prefix: str = "") -> str:
        """生成 ASCII 项目结构树"""
        tree_str = ""
        try:
            items = sorted(list(dir_path.iterdir()),
                           key=lambda x: (not x.is_dir(), x.name.lower()))

            items = [i for i in items
                     if i.name not in self.EXCLUDE_DIRS
                     and i.name not in self.EXCLUDE_FILES
                     and not i.name.startswith('.')]

            for i, item in enumerate(items):
                connector = "└── " if i == len(items) - 1 else "├── "
                tree_str += f"{prefix}{connector}{item.name}{'/' if item.is_dir() else ''}\n"
                if item.is_dir():
                    extension = "    " if i == len(items) - 1 else "│   "
                    tree_str += self._generate_tree(item, prefix + extension)
        except PermissionError:
            pass
        return tree_str

    def _scan_files(self):
        """全盘扫描"""
        self.scanned_files = []
        for root, dirs, files in os.walk(self.root):
            dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS and not d.startswith('.')]
            for f in sorted(files):
                if f in self.EXCLUDE_FILES: continue
                path = Path(root) / f
                if path.suffix in self.INCLUDE_EXTS:
                    self.scanned_files.append(path.relative_to(self.root))

    def _escape_tex(self, text: str) -> str:
        """转义 LaTeX 特殊字符"""
        replacements = {'_': r'\_', '%': r'\%', '$': r'\$', '#': r'\#', '&': r'\&'}
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def run(self):
        """执行导出 (匹配 main.py 的调用)"""
        start = time.time()
        self.logger.info("启动代码附录生成引擎...")
        self._scan_files()

        with open(self.tex_path, 'w', encoding='utf-8') as f:
            # 1. 写入头文件
            f.write("% Auto-generated by MCM Project Exporter\n")
            f.write(self.LATEX_PREAMBLE)
            f.write("\n\\section{Code Appendix}\n")
            f.write("\\textit{This section creates the foundation of our Hybrid C++/Python Architecture.}\n\n")

            # 2. 写入项目结构树
            f.write("\\subsection{Project Directory Structure}\n")
            f.write("\\begin{verbatim}\n")
            f.write(f"{self.root.name}/\n")
            f.write(self._generate_tree(self.root))
            f.write("\\end{verbatim}\n\n")

            # 3. 写入核心代码文件
            f.write("\\subsection{Core Implementation Details}\n")
            for rel_path in self.scanned_files:
                lang = "Python"
                if rel_path.suffix in ['.cpp', '.hpp', '.h']:
                    lang = "C++"
                elif rel_path.suffix == '.yaml':
                    lang = "bash"
                elif rel_path.suffix == '.cmake':
                    lang = "bash"

                title = self._escape_tex(str(rel_path))

                try:
                    content = (self.root / rel_path).read_text(encoding='utf-8')
                    # 过滤空文件
                    if len(content.strip()) < 10: continue

                    f.write(f"\\subsubsection*{{File: \\texttt{{{title}}}}}\n")
                    f.write(f"\\begin{{lstlisting}}[language={lang}]\n")
                    f.write(content)
                    f.write("\n\\end{lstlisting}\n\n")
                except Exception as e:
                    self.logger.warning(f"无法读取文件 {rel_path}: {e}")

        self.logger.info(f"附录生成完毕: {self.tex_path} (耗时 {time.time() - start:.2f}s)")


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    root_dir = Path(__file__).resolve().parent.parent.parent
    exporter = MCMProjectExporter(str(root_dir))
    exporter.run()