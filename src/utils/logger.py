# ==============================================================================
# src/utils/logger.py
# Role: Industrial Dual-Logging System (The "Black Box" Recorder)
# Function: Real-time monitoring of MCMC convergence and deep exception tracking.
# Architecture:
#   1. Console Stream -> High-level Progress (Alpha Signals) -> INFO Level
#   2. File Stream    -> Deep Debugging & Stack Traces (Beta Noise) -> DEBUG Level
#   3. Audit Stream   -> Immutable Scientific Results (R-hat, Fidelity) -> INFO Level
# Standard: Industrial Robustness / Academic Traceability / UTF-8 Forced.
# ==============================================================================

import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# --- 全局常量 ---
LOG_DIR_NAME = "logs"
DEFAULT_ENCODING = "utf-8"

def _get_log_dir() -> Path:
    """
    [路径自愈] 自动定位项目根目录下的 logs 文件夹。
    逻辑：从 src/utils/logger.py 向上回溯三级 (src/utils -> src -> root)。
    物理意义：确保无论脚本在何处启动 (IDE, CLI, Docker)，日志都能落盘到统一位置。
    """
    try:
        # __file__ is src/utils/logger.py
        root_dir = Path(__file__).resolve().parent.parent.parent
        log_dir = root_dir / LOG_DIR_NAME

        # 线程安全的目录创建 (exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    except Exception as e:
        # 极端情况降级：输出到当前目录
        sys.stderr.write(f"[FATAL] 日志目录定位失败: {e}. 降级至 ./logs\n")
        return Path(f"./{LOG_DIR_NAME}")

def setup_logger(name: str = "MCM_MASTER", log_filename: str = None) -> logging.Logger:
    """
    配置通用双路日志系统 (Dual-Stream Logging)。

    :param name: 日志器唯一标识符 (e.g., 'MCM_KERNEL', 'ETL_PIPELINE')
    :param log_filename: 日志文件名。若为空，自动生成 'system_runtime_YYYYMMDD.log'
    :return: 配置好的 Logger 单例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # 捕获所有底层信息，分流处理

    # 1. 幂等性检查：防止 Handler 重复挂载 (常见于 Jupyter 重复运行或多模块 Import)
    if logger.handlers:
        return logger

    # 2. 自动文件名生成 (按日滚动)
    if log_filename is None:
        today = datetime.now().strftime("%Y%m%d")
        log_filename = f"system_runtime_{today}.log"

    log_path = _get_log_dir() / log_filename

    # 3. 定义高信息熵格式化器 (Formatter)
    # [文件端] 全息格式：时间 | 级别 | 模块 | 源码位置 | 消息
    file_fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-15s | [%(filename)s:%(lineno)d] | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # [控制台] 极简格式：时间 | 消息 (类似 TQDM 风格，保持界面清爽)
    console_fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # 4. 配置控制台流 (Screen Stream) -> INFO+
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 5. 配置文件流 (Disk Stream) -> DEBUG+
    try:
        # mode='a': 追加模式，保留历史记录
        file_handler = logging.FileHandler(log_path, encoding=DEFAULT_ENCODING, mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    except PermissionError:
        # 降级防御：如果文件被 Excel 打开占用，仅输出警告，不阻断主流程
        sys.stderr.write(f"[WARNING] 无法写入日志文件 (权限被拒): {log_path}\n")

    # 6. 防止日志向上传播导致双重打印
    logger.propagate = False

    return logger

def setup_audit_logger(name: str = "SCIENTIFIC_AUDIT", log_filename: str = "scientific_audit.log") -> logging.Logger:
    """
    [科学审计专用] 配置纯文件日志系统。

    物理意义：
    用于记录“不可变”的实验结果 (Immutable Results)，如 R-hat, ESS, p-values。
    该日志**不输出到控制台**，且格式纯净，便于后续 `abstract_helper.py` 脚本解析生成表格。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    # 审计日志格式：纯文本，无冗余元数据，方便 Regex 解析
    # 例如：[S27-W10] Fidelity=0.95, R_hat=1.02
    audit_fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    log_path = _get_log_dir() / log_filename

    try:
        file_handler = logging.FileHandler(log_path, encoding=DEFAULT_ENCODING, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(audit_fmt)
        logger.addHandler(file_handler)
    except Exception as e:
        sys.stderr.write(f"[FATAL] 审计日志初始化失败: {e}\n")

    # 核心：审计日志绝不应该污染控制台
    logger.propagate = False

    return logger

# --- 单元测试 (Unit Test) ---
if __name__ == "__main__":
    # 模拟系统启动
    main_logger = setup_logger("TEST_KERNEL", "test_run.log")
    audit_logger = setup_audit_logger("TEST_AUDIT", "test_audit.log")

    main_logger.info(">>> 系统启动自检...")
    main_logger.debug("正在检查 C++ 内存指针地址: 0x7ffee4c88aac (Debug 详情仅在文件中可见)")

    try:
        # 模拟计算过程
        audit_logger.info("Task 1 Convergence: R-hat=1.002 [PASS]")
        main_logger.info("Task 1 完成。审计记录已存档。")
        # 模拟异常
        x = 1 / 0
    except Exception as e:
        # [核心亮点] 自动抓取堆栈信息
        main_logger.critical("检测到致命计算错误:", exc_info=True)

    print(f"\n[测试完成] 请检查目录: {_get_log_dir()}")