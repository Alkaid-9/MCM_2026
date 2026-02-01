# ==============================================================================
# src/utils/logger.py
# Role: Industrial Dual-Logging System (The "Black Box" Recorder)
# Function: Real-time monitoring of MCMC convergence and deep exception tracking.
# Architecture:
#   1. Console Stream -> High-level Progress (Alpha Signals)
#   2. File Stream -> Deep Debugging & Stack Traces (Beta Noise)
# Standard: Industrial Robustness / Academic Traceability.
# ==============================================================================

import logging
import sys
import os
from pathlib import Path
from datetime import datetime

def _get_log_dir() -> Path:
    """
    [路径自愈] 自动定位项目根目录下的 logs 文件夹。
    逻辑：从 src/utils/logger.py 向上回溯两级 (src/utils -> src -> root)。
    """
    root_dir = Path(__file__).resolve().parent.parent.parent
    log_dir = root_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def setup_logger(name: str = "MCM_MASTER", log_filename: str = "system_runtime.log") -> logging.Logger:
    """
    配置通用双路日志系统。
    
    :param name: 日志器唯一标识符 (e.g., 'MCM_KERNEL')
    :param log_filename: 日志文件名 (e.g., 'mcm_runtime.log')
    :return: 配置好的 Logger 单例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG) # 捕获所有底层信息

    # 1. 幂等性检查：防止 Handler 重复挂载 (常见于 Jupyter 或多次调用)
    if logger.handlers:
        return logger

    # 2. 定义格式化器 (Formatter)
    # [文件端] 全息格式：时间 | 级别 | 模块 | 源码位置 | 消息
    file_fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-15s | [%(filename)s:%(lineno)d] | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # [控制台] 极简格式：时间 | 消息 (类似 TQDM 风格)
    console_fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # 3. 配置控制台流 (Screen Stream)
    # 策略：只看 INFO，保持界面清爽
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 4. 配置文件流 (Disk Stream)
    # 策略：记录 DEBUG，作为灾难回溯的黑匣子
    log_path = _get_log_dir() / log_filename
    try:
        # encoding='utf-8' 是必须的，否则 Windows 下中文日志会报错
        file_handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)
    except PermissionError:
        # 降级处理：如果文件被占用，仅输出警告，不阻断主流程
        sys.stderr.write(f"[WARNING] 无法写入日志文件: {log_path}\n")

    logger.debug(f"日志系统初始化完成。Log Path: {log_path}")
    return logger

def setup_audit_logger(name: str = "AUDIT_TRAIL", log_filename: str = "scientific_audit.log") -> logging.Logger:
    """
    [科学审计专用] 配置纯文件日志系统。
    
    物理意义：
    用于记录“不可变”的实验结果 (如 R-hat, ESS, p-values)。
    该日志不输出到控制台，且格式纯净，便于后续脚本解析生成表格。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    # 审计日志格式：纯文本，无冗余元数据
    audit_fmt = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    log_path = _get_log_dir() / log_filename
    file_handler = logging.FileHandler(log_path, encoding='utf-8', mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(audit_fmt)
    
    logger.addHandler(file_handler)
    # 确保不传播给父 Logger (防止污染控制台)
    logger.propagate = False
    
    return logger