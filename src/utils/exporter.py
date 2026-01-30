import os
import json
import time
from pathlib import Path
from typing import List, Set


class MCMProjectExporter:
    """
    MCM 2026 项目导出 工具
    用于将项目源码导出为 Markdown 和 LaTeX 格式，方便提交

    核心职责：
    1. 风控机制：自动熔断大文件 (>500KB)，屏蔽敏感文件 (.env)。
    2. 智能解析：解析 Jupyter Notebook 代码块，提取 Python 文档摘要。
    3. 出版级排版：生成带索引的 Markdown 和带换行箭头的 LaTeX 附录。
    4. 鲁棒运行：智能定位项目根目录，无需手动修改路径。
    """

    # === 1. 配置区域 (Configuration) ===

    # 需要导出的文件后缀
    INCLUDE_EXTENSIONS = {
        '.py', '.cpp', '.h', '.hpp', '.c',
        '.yaml', '.yml', '.toml',
        '.md', '.txt', '.cmake',
        '.ipynb'  # 支持 Jupyter Notebook
    }

    # 彻底忽略的目录 (不出现在树里，也不扫描)
    EXCLUDE_DIRS = {
        '__pycache__', '.git', '.idea', '.vscode', 'venv', '.venv',
        'cmake-build-debug', 'cmake-build-release', 'bin', 'obj', 'lib',
        'wandb', 'runs', 'logs', 'htmlcov', 'egg-info',
        'data', 'outputs', 'assets', 'notebooks'  # 调试用的 notebook 通常不放附录，除非是核心逻辑
    }

    # 忽略的具体文件名 (安全黑名单)
    EXCLUDE_FILES = {
        '.env', 'LICENSE', 'README.md', 'requirements.txt',
        'mcm_core_lib.so', 'mcm_core_lib.pyd', '.DS_Store',
        'Code_Appendix.md', 'appendix_code.tex',  # 防止递归把自己导出来
        '__init__.py.py'  # 如果是空的 __init__.py 就不导出了，节省纸张
    }

    # LaTeX 头部配置 (VS Code 风格高亮 + 换行箭头优化)
    LATEX_PREAMBLE = r"""
% --- MCM Code Style Definition ---
\usepackage{listings}
\usepackage{xcolor}
\usepackage{amssymb} % 用于显示箭头符号

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
    basicstyle=\ttfamily\scriptsize, % 极小字体节省空间
    breakatwhitespace=false,         
    breaklines=true,                 % 自动换行
    postbreak=\mbox{\textcolor{red}{$\hookrightarrow$}\space}, % [亮点] 换行处显示红色箭头
    captionpos=b,                    
    keepspaces=true,                 
    numbers=left,                    % 显示行号
    numbersep=5pt,                  
    showspaces=false,                
    showstringspaces=false,
    showtabs=false,                  
    tabsize=2,
    frame=single,                    % 代码框
    frameround=tttt,                 % 圆角
    rulecolor=\color{codegray}
}
\lstset{style=mcmstyle}
% ---------------------------------
"""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        # 输出路径
        self.md_file = project_root / "Code_Appendix.md"
        self.tex_file = project_root / "appendix_code.tex"
        self.scanned_files = []  # 存储扫描到的文件相对路径

    # === 2. 核心逻辑 (Core Logic) ===

    def _scan_files(self):
        """执行一次全盘扫描，建立文件索引"""
        self.scanned_files = []
        for root, dirs, files in os.walk(self.project_root):
            # 剪枝目录：原地修改 dirs 列表
            dirs[:] = [d for d in dirs if d not in self.EXCLUDE_DIRS and not d.startswith('.')]

            for filename in sorted(files):
                if filename in self.EXCLUDE_FILES: continue
                file_path = Path(root) / filename
                if file_path.suffix not in self.INCLUDE_EXTENSIONS: continue

                # 记录相对路径
                rel_path = file_path.relative_to(self.project_root)
                self.scanned_files.append(rel_path)

        # 按字母序排序，保证输出稳定性
        self.scanned_files.sort()

    def _generate_tree(self, dir_path: Path, prefix: str = "") -> str:
        """生成 ASCII 目录树"""
        tree_str = ""
        try:
            entries = sorted(list(os.scandir(dir_path)), key=lambda e: (not e.is_dir(), e.name.lower()))
        except:
            return ""

        filtered = [e for e in entries if e.name not in self.EXCLUDE_DIRS and not e.name.startswith('.')]
        count = len(filtered)

        for index, entry in enumerate(filtered):
            connector = "├── " if index < count - 1 else "└── "
            tree_str += f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}\n"
            if entry.is_dir():
                tree_str += self._generate_tree(Path(entry.path), prefix + ("│   " if index < count - 1 else "    "))
        return tree_str

    def _read_content(self, rel_path: Path) -> str:
        """智能读取内容：增加大小熔断机制"""
        full_path = self.project_root / rel_path

        # [熔断机制] 超过 500KB 的文件直接跳过，防止 PDF 爆炸
        if full_path.stat().st_size > 500 * 1024:
            return f"# [Skip] File is too large ({full_path.stat().st_size / 1024:.1f} KB). Please check the file directly."

        if full_path.suffix == '.ipynb':
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    nb = json.load(f)
                cells = []
                for cell in nb.get("cells", []):
                    if cell.get("cell_type") == "code":
                        # 过滤掉魔法命令 (!pip, %timeit)
                        source = "".join([l for l in cell.get("source", []) if not l.strip().startswith(('!', '%'))])
                        if source.strip():
                            cells.append(source)
                return "\n\n# %% [Jupyter Cell]\n".join(cells)
            except Exception as e:
                return f"# [Error Parsing Notebook] {e}"
        else:
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    return f.read()
            except UnicodeDecodeError:
                return "# [Error] Binary file or unknown encoding."

    def _extract_docstring(self, content: str) -> str:
        """提取代码文件的首部注释作为简介 (用于生成 Markdown 索引表)"""
        content = content.strip()
        if content.startswith('"""'):
            end = content.find('"""', 3)
            if end != -1:
                # 只取第一行，保持表格整洁
                return content[3:end].strip().split('\n')[0]
        return "-"

    def _escape_latex(self, text: str) -> str:
        """转义 LaTeX 特殊字符"""
        chars = {
            "_": r"\_", "%": r"\%", "$": r"\$", "#": r"\#",
            "&": r"\&", "{": r"\{", "}": r"\}", "^": r"\^{}",
            "~": r"\~{}", "\\": r"/"
        }
        for k, v in chars.items():
            text = text.replace(k, v)
        return text

    def _get_mermaid_graph(self) -> str:
        """返回系统架构图"""
        return """
```mermaid
graph TD
    classDef cpp fill:#f9f,stroke:#333,stroke-width:2px,color:black;
    classDef py fill:#bbf,stroke:#333,stroke-width:2px,color:black;
    classDef conf fill:#ff9,stroke:#333,stroke-width:2px,color:black;

    subgraph "High-Performance Kernel (C++)"
        CppSrc[fast_kernel.cpp]:::cpp --> Lib([mcm_core_lib.so]):::cpp
    end

    subgraph "Python Logic Layer"
        Wrapper[des_engine.py]:::py
        Solver[optimizer.py]:::py
        Evaluator[risk.py]:::py

        Lib -.-> Wrapper
        Solver --> Wrapper
        Wrapper --> Evaluator
    end

    Config[config.yaml]:::conf --> Solver
```
"""

    # === 3. 导出 Markdown (For AI/GitHub) ===
    def export_markdown(self):
        print(f"📘 正在生成 Markdown 资产: {self.md_file.name} ...")
        with open(self.md_file, "w", encoding="utf-8") as f:
            # 封面
            f.write(f"# Project Source Code: {self.project_root.name}\n\n")
            f.write("> **System Architecture:** Hybrid C++ and Python\n\n")

            # 1. 模块功能索引表 (新功能)
            f.write("## 1. Key Module Index\n")
            f.write("| File Path | Description |\n| :--- | :--- |\n")
            for rel_path in self.scanned_files:
                # 只列出 Python 和 C++ 核心文件
                if rel_path.suffix in ['.py', '.cpp', '.h']:
                    content = self._read_content(rel_path)
                    desc = self._extract_docstring(content)
                    if desc != "-":  # 只显示有注释的文件
                        f.write(f"| `{rel_path}` | {desc} |\n")
            f.write("\n\n")

            # 2. 架构图
            f.write("## 2. System Architecture\n")
            f.write(self._get_mermaid_graph())
            f.write("\n\n")

            # 3. 目录树
            f.write("## 3. Directory Structure\n```text\n")
            f.write(f"{self.project_root.name}/\n")
            f.write(self._generate_tree(self.project_root))
            f.write("```\n\n")

            # 4. 源码详情
            f.write("## 4. Source Files\n")
            for rel_path in self.scanned_files:
                lang = "python"
                if rel_path.suffix in ['.cpp', '.h', '.hpp']:
                    lang = "cpp"
                elif rel_path.suffix in ['.yaml', '.yml']:
                    lang = "yaml"
                elif rel_path.suffix == '.cmake':
                    lang = "cmake"

                content = self._read_content(rel_path)
                # 跳过空的 __init__.py
                if rel_path.name == '__init__.py.py' and not content.strip():
                    continue

                f.write(f"### `{rel_path}`\n")
                f.write(f"```{lang}\n{content}\n```\n\n")

    # === 4. 导出 LaTeX (For Paper) ===
    def export_latex(self):
        print(f"🎨 正在生成 LaTeX 附录: {self.tex_file.name} ...")
        with open(self.tex_file, "w", encoding="utf-8") as f:
            # 写入样式定义
            f.write(self.LATEX_PREAMBLE)
            f.write("\n% --- Start of Appendix ---\n")
            f.write("\\section{Code Appendix}\n")
            f.write(
                "\\textit{This section provides the core implementation details. Auxiliary files and logs are omitted for brevity.}\n\n")

            # 目录树
            f.write("\\subsection{Project Directory}\n")
            f.write("\\begin{verbatim}\n")
            f.write(f"{self.project_root.name}/\n")
            f.write(self._generate_tree(self.project_root))
            f.write("\\end{verbatim}\n\n")

            # 源码详情
            f.write("\\subsection{Core Implementation}\n")
            for rel_path in self.scanned_files:
                # 自动判定语言
                lang = "Python"
                if rel_path.suffix in ['.cpp', '.h', '.hpp']:
                    lang = "C++"
                elif rel_path.suffix in ['.yaml', '.yml']:
                    lang = "bash"

                content = self._read_content(rel_path)

                # 再次检查：如果是空文件或只包含跳过信息的，不写入 LaTeX
                if not content.strip() or content.startswith("# [Skip]"):
                    continue
                # 跳过 __init__.py.py 以节省版面
                if rel_path.name == '__init__.py.py':
                    continue

                safe_name = self._escape_latex(str(rel_path))

                # 写入代码块
                f.write(f"\\subsubsection*{{File: {safe_name}}}\n")
                f.write(f"\\begin{{lstlisting}}[language={lang}]\n")
                f.write(content)
                f.write("\n\\end{lstlisting}\n\n")

    # === 5. 执行入口 ===
    def run(self):
        start = time.time()
        print(f"🚀 启动全栈导出引擎...")

        # 1. 扫描
        self._scan_files()
        print(f"📊 扫描到 {len(self.scanned_files)} 个核心文件")

        # 2. 导出
        self.export_markdown()
        self.export_latex()

        print(f"✅ 全部完成！耗时 {time.time() - start:.2f}s")
        print(f"👉 Markdown: {self.md_file}")
        print(f"👉 LaTeX:    {self.tex_file}")


if __name__ == "__main__":
    # 智能定位根目录：向上寻找直到发现 'src' 或 'requirements.txt'
    # 无论你把脚本移到哪，只要在项目里就能找到根
    current_path = Path(__file__).resolve()
    project_root = current_path

    found = False
    for _ in range(5):  # 最多找5层
        if (project_root / "src").exists() or (project_root / "requirements.txt").exists():
            found = True
            break
        project_root = project_root.parent

    if not found:
        print("⚠️  Warning: 无法自动定位项目根目录，默认回退3层...")
        project_root = current_path.parent.parent.parent

    print(f"📂 锁定项目根目录: {project_root}")
    exporter = MCMProjectExporter(project_root)
    exporter.run()

    print("🏁 导出任务结束。请检查生成的 Markdown 和 LaTeX 文件。")
