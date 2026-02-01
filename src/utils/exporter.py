# ==============================================================================
# src/utils/exporter.py
# Role: Automated Research Project Exporter (v4.8 - Production Grade)
# Function: Generating professional LaTeX and Markdown code appendices.
# Standard: Publication-ready syntax highlighting & modular structure mapping.
# ==============================================================================

import os
import time
from pathlib import Path
from typing import List, Set, Dict


class MCMProjectExporter:
    """
    项目导出引擎：
    一键生成符合美赛/顶刊提交标准的代码附录。

    【核心职能】：
    1. 自动生成项目结构树 (ASCII Tree)。
    2. 提取 Python/C++ 文档摘要生成索引表。
    3. 自动处理 LaTeX 转义与 listings 高亮配置。
    4. 过滤非核心文件 (Data, Logs, Build artifacts)。
    """

    # --- 配置区域 ---
    INCLUDE_EXTS = {'.py', '.cpp', '.hpp', '.h', '.yaml', '.cmake', '.txt'}
    EXCLUDE_DIRS = {
        '__pycache__', '.git', '.vscode', 'venv', 'build', 'bin',
        'lib', 'obj', 'data', 'logs', 'outputs', 'notebooks'
    }
    EXCLUDE_FILES = {
        '.env', 'LICENSE', 'README.md', 'requirements.txt',
        'mcm_core_lib.so', 'Code_Appendix.md'
    }

    # LaTeX 样式定义 ( listings 宏包配置 )
    LATEX_PREAMBLE = r"""
% --- MCM Code Appendix Style Definition ---
\usepackage{listings}
\usepackage{xcolor}

\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{codepurple}{rgb}{0.58,0,0.82}
\definecolor{backcolour}{rgb}{0.96,0.96,0.96}

\lstdefinestyle{mcmstyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{magenta},
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
    frame=single
}
\lstset{style=mcmstyle}
% ------------------------------------------
"""

    def __init__(self, project_root: str):
        self.root = Path(project_root).resolve()
        self.scanned_files: List[Path] = []
        self.md_path = self.root / "Code_Appendix.md"
        self.tex_path = self.root / "appendix_code.tex"

    def _generate_tree(self, dir_path: Path, prefix: str = "") -> str:
        """生成 ASCII 项目结构树"""
        tree_str = ""
        try:
            items = sorted([item for item in dir_path.iterdir()
                            if item.name not in self.EXCLUDE_DIRS
                            and not item.name.startswith('.')])

            for i, item in enumerate(items):
                connector = "└── " if i == len(items) - 1 else "├── "
                tree_str += f"{prefix}{connector}{item.name}/\n" if item.is_dir() else f"{prefix}{connector}{item.name}\n"

                if item.is_dir():
                    extension = "    " if i == len(items) - 1 else "│   "
                    tree_str += self._generate_tree(item, prefix + extension)
        except PermissionError:
            pass
        return tree_str

    def _extract_summary(self, file_path: Path) -> str:
        """提取代码头部的第一行简介"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if not content: return "-"
                # 寻找 Python/C++ 风格的注释摘要
                lines = content.split('\n')
                for line in lines:
                    clean = line.strip().replace('"""', '').replace('#', '').replace('/*', '').replace('*/', '').strip()
                    if clean and len(clean) > 5:
                        return clean[:80]  # 截断
        except:
            pass
        return "-"

    def _scan_files(self):
        """全盘扫描核心源文件"""
        self.scanned_files = []
        for root, dirs, files in os.walk(self.root):
            # 原地修改 dirs 以便跳过排除目录
            dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS and not d.startswith('.')]

            for f in sorted(files):
                f_path = Path(root) / f
                if f_path.suffix in self.INCLUDE_EXTS and f not in self.EXCLUDE_FILES:
                    self.scanned_files.append(f_path.relative_to(self.root))

    def export_markdown(self):
        """导出为 Markdown (用于文档存档)"""
        with open(self.md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Code Appendix: {self.root.name}\n\n")
            f.write("## 1. Project Structure\n```text\n")
            f.write(self._generate_tree(self.root))
            f.write("```\n\n")

            f.write("## 2. Source Code Details\n")
            for rel_path in self.scanned_files:
                lang = "python" if rel_path.suffix == ".py" else "cpp"
                f.write(f"### `{rel_path}`\n")
                f.write(f"```{lang}\n")
                f.write((self.root / rel_path).read_text(encoding='utf-8', errors='replace'))
                f.write("\n```\n\n")

    def export_latex(self):
        """导出为 LaTeX (用于论文附件)"""

        def escape_tex(text):
            return text.replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')

        with open(self.tex_path, 'w', encoding='utf-8') as f:
            f.write("% --- Automated Code Appendix Generated for MCM 2026 ---\n")
            f.write(self.LATEX_PREAMBLE)
            f.write("\n\\section{Code Appendix}\n")
            f.write(
                "\\textit{This appendix contains the core implementation of the Bayesian Inference Engine and the Mechanism Forensics Pipeline. Data files and logs are omitted for brevity.}\n\n")

            # 1. 索引表
            f.write("\\subsection{Module Index}\n")
            f.write(
                "\\begin{tabular}{ll}\n\\toprule\n\\textbf{File Path} & \\textbf{Functional Description} \\\\\n\\midrule\n")
            for p in self.scanned_files:
                summary = self._extract_summary(self.root / p)
                f.write(f"\\texttt{{{escape_tex(str(p))}}} & {escape_tex(summary)} \\\\\n")
            f.write("\\bottomrule\n\\end{tabular}\n\n")

            # 2. 结构树
            f.write("\\subsection{Directory Architecture}\n")
            f.write("\\begin{verbatim}\n")
            f.write(self._generate_tree(self.root))
            f.write("\\end{verbatim}\n\n")

            # 3. 源代码
            f.write("\\subsection{Implementation Details}\n")
            for p in self.scanned_files:
                lang = "Python" if p.suffix == ".py" else "C++"
                if p.suffix == '.yaml': lang = "bash"  # listings 并不原生支持 yaml，用 bash 代替
                f.write(f"\\subsubsection*{{File: \\texttt{{{escape_tex(str(p))}}}}}\n")
                f.write(f"\\begin{{lstlisting}}[language={lang}]\n")
                f.write((self.root / p).read_text(encoding='utf-8', errors='replace'))
                f.write("\n\\end{lstlisting}\n\n")

    def run(self):
        start = time.time()
        print(f"🚀 启动项目全栈导出引擎...")
        self._scan_files()
        print(f"  - 扫描到 {len(self.scanned_files)} 个核心源文件")
        self.export_markdown()
        self.export_latex()
        print(f"✅ 导出完成！耗时: {time.time() - start:.2f}s")
        print(f"  - Markdown 附录: {self.md_path}")
        print(f"  - LaTeX 附录: {self.tex_path}")


# --- 执行入口 ---
if __name__ == "__main__":
    # 获取项目根目录 (假设脚本在 src/utils/)
    root_dir = Path(__file__).resolve().parent.parent.parent
    exporter = MCMProjectExporter(root_dir)
    exporter.run()