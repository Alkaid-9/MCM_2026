"""
MCM 2026 Problem C: Automated Project Exporter
Role: Generating publication-ready code appendices (LaTeX/Markdown) from the hybrid codebase.
Standard: Submission Format (Clean Tree + Syntax Highlighting).
"""

import os
import time
from pathlib import Path
from typing import List, Set


class MCMProjectExporter:
    """
    项目导出引擎：一键生成符合美赛提交标准的代码附录。

    【核心功能】：
    1. 智能目录树 (ASCII Tree): 自动生成项目结构图，展示 C++/Python 分层架构。
    2. LaTeX 自动化: 生成带有 listings 高亮的 .tex 文件，支持长代码自动换行。
    3. 噪音过滤: 自动屏蔽 venv, build, __pycache__ 等非核心文件。
    """

    # === 配置区域 ===
    INCLUDE_EXTENSIONS = {'.py', '.cpp', '.hpp', '.h', '.yaml', '.cmake', '.txt'}
    EXCLUDE_DIRS = {
        '__pycache__', '.git', '.idea', '.vscode', 'venv', 'env',
        'build', 'bin', 'lib', 'obj', 'wandb', 'logs', 'assets',
        'data', 'reports', 'notebooks'
    }
    EXCLUDE_FILES = {
        '.env', '.DS_Store', 'mcm_core_lib.so', 'mcm_core_lib.pyd',
        'requirements.txt', 'README.md', 'LICENSE', 'Code_Appendix.md'
    }

    def __init__(self, project_root: Path):
        self.root = project_root
        self.scanned_files = []

        # 输出目标
        self.md_path = self.root / "Code_Appendix.md"
        self.tex_path = self.root / "appendix_code.tex"

    def _generate_tree(self, dir_path: Path, prefix: str = "") -> str:
        """生成精美的 ASCII 项目结构树"""
        tree_str = ""
        try:
            # 排序：文件夹在前，文件在后，字母序
            entries = sorted(
                list(os.scandir(dir_path)),
                key=lambda e: (not e.is_dir(), e.name.lower())
            )
        except PermissionError:
            return ""

        # 过滤
        filtered = [e for e in entries if e.name not in self.EXCLUDE_DIRS and not e.name.startswith('.')]
        count = len(filtered)

        for index, entry in enumerate(filtered):
            connector = "├── " if index < count - 1 else "└── "
            tree_str += f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}\n"

            if entry.is_dir():
                extension = "│   " if index < count - 1 else "    "
                tree_str += self._generate_tree(Path(entry.path), prefix + extension)

        return tree_str

    def _scan_files(self):
        """全盘扫描并建立索引"""
        self.scanned_files = []
        for root, dirs, files in os.walk(self.root):
            # 原地剪枝
            dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS and not d.startswith('.')]

            for f in sorted(files):
                if f in self.EXCLUDE_FILES: continue
                path = Path(root) / f
                if path.suffix not in self.INCLUDE_EXTENSIONS: continue

                # 记录相对路径
                self.scanned_files.append(path.relative_to(self.root))

        # 再次排序保证输出稳定
        self.scanned_files.sort()

    def _read_content(self, rel_path: Path) -> str:
        """读取文件内容，增加熔断保护"""
        full_path = self.root / rel_path

        # 熔断：超过 1MB 的文件不展示 (如大数据 csv 被误读)
        if full_path.stat().st_size > 1024 * 1024:
            return f"# [Content Omitted] File size > 1MB. Please refer to external attachment."

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 针对 __init__.py 空文件做特殊标记
                if not content.strip():
                    return "# (Module Initialization)"
                return content
        except Exception as e:
            return f"# [Error reading file]: {str(e)}"

    def _escape_latex(self, text: str) -> str:
        """转义 LaTeX 特殊字符"""
        replacements = {
            '_': r'\_', '%': r'\%', '$': r'\$', '#': r'\#',
            '&': r'\&', '{': r'\{', '}': r'\}'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    def export_latex(self):
        """生成可直接 Copy 进论文 Main.tex 的代码附录"""
        print(f"📄 正在生成 LaTeX 附录: {self.tex_path.name}")

        preamble = r"""
% === Code Appendix Style Definition ===
\usepackage{listings}
\usepackage{xcolor}
\definecolor{codegreen}{rgb}{0,0.6,0}
\definecolor{codegray}{rgb}{0.5,0.5,0.5}
\definecolor{backcolour}{rgb}{0.96,0.96,0.96}
\lstdefinestyle{mcmstyle}{
    backgroundcolor=\color{backcolour},   
    commentstyle=\color{codegreen},
    keywordstyle=\color{blue},
    numberstyle=\tiny\color{codegray},
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
% ======================================
"""
        with open(self.tex_path, 'w', encoding='utf-8') as f:
            # 1. 写入 Preamble (可选，方便用户复制)
            f.write("% Copy this preamble to your main.tex if needed:\n")
            f.write(preamble)
            f.write("\n\n\\section{Code Appendix}\n")
            f.write("\\textit{This section implements the Hybrid C++/Python Architecture described in the paper.}\n\n")

            # 2. 写入项目结构树
            f.write("\\subsection{Project Structure}\n")
            f.write("\\begin{verbatim}\n")
            f.write(f"{self.root.name}/\n")
            f.write(self._generate_tree(self.root))
            f.write("\\end{verbatim}\n\n")

            # 3. 写入核心代码
            f.write("\\subsection{Core Implementation}\n")

            for rel_path in self.scanned_files:
                # 自动推断语言
                lang = "Python"
                if rel_path.suffix in ['.cpp', '.hpp', '.h']:
                    lang = "C++"
                elif rel_path.suffix == '.cmake':
                    lang = "bash"
                elif rel_path.suffix == '.yaml':
                    lang = "yaml"

                content = self._read_content(rel_path)
                safe_name = self._escape_latex(str(rel_path))

                f.write(f"\\subsubsection*{{File: {safe_name}}}\n")
                f.write(f"\\begin{{lstlisting}}[language={lang}]\n")
                f.write(content)
                f.write("\n\\end{lstlisting}\n\n")

    def run(self):
        start_t = time.time()
        print(f"🚀 启动项目导出引擎...")

        self._scan_files()
        print(f"📊 扫描到 {len(self.scanned_files)} 个核心文件")

        # 导出 LaTeX (核心产物)
        self.export_latex()

        # 导出 Markdown (辅助 AI 阅读或 GitHub 展示)
        with open(self.md_path, 'w', encoding='utf-8') as f:
            f.write(f"# Project Archive: {self.root.name}\n\n")
            f.write("## Directory Structure\n```text\n")
            f.write(self._generate_tree(self.root))
            f.write("```\n\n## Source Code\n")
            for p in self.scanned_files:
                lang = p.suffix[1:]
                f.write(f"### `{p}`\n```{lang}\n{self._read_content(p)}\n```\n\n")

        print(f"✅ 导出完成! 耗时 {time.time() - start_t:.2f}s")
        print(f"👉 LaTeX 附录: {self.tex_path}")


if __name__ == "__main__":
    # 智能定位项目根目录
    current = Path(__file__).resolve()
    project_root = current.parent.parent.parent  # src/utils/exporter.py -> root

    exporter = MCMProjectExporter(project_root)
    exporter.run()