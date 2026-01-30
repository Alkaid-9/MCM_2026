"""
MCM 2026 Problem C: Industrial Dual-Logging System
Role: Real-time monitoring of MCMC convergence and deep exception tracking.
Standard: Industrial Robustness / Academic Traceability"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime


def setup_logger(name: str = "MCM_MASTER", log_file: str = None) -> logging.Logger:
    """
    配置双路日志系统。

    物理意义：
    - Console Handler: 实时进度监控 (Alpha 信号跟踪)。
    - File Handler: 全量实验取证 (Beta 噪音与异常回溯)。

    :param name: 日志器唯一标识符
    :param log_file: 日志文件路径。若为空，则根据 rules.yaml 中的配置生成。
    """
    logger = logging.getLogger(name)

    # 1. 级别设定：捕获所有底层的 DEBUG 信息
    logger.setLevel(logging.DEBUG)

    # 2. 防止 Handler 重复挂载 (在多次初始化单例时常见)
    if logger.handlers:
        return logger

    # 3. 定义高信息熵的格式化器
    # 包含：精确时间 | 级别 | 模块名 | [文件名:行号] | 具体信息
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | [%(filename)s:%(lineno)d] | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- A. Console Handler (屏幕流) ---
    # 策略：只看 INFO 以上，避免被 MCMC 的千万次循环日志刷屏
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)

    # --- B. File Handler (文件流) ---
    # 策略：记录所有 DEBUG 细节，作为论文“收敛性分析”的原始证据
    if log_file:
        log_path = Path(log_file)
        # 自动建立日志目录，体现工程闭环
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # 使用 utf-8 编码，防止 Windows 环境下的中文字符崩溃
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "MCM_MASTER"):
    """
    快速获取已配置的 logger 单例。
    """
    return logging.getLogger(name)


# --- 单元测试 ---
if __name__ == "__main__":
    # 模拟从 config 加载路径
    test_log = "logs/test_run.log"

    logger = setup_logger("TEST_ENGINE", test_log)

    logger.info("🚀 正在点火反演引擎...")
    logger.debug("正在检查 C++ 内存指针地址: 0x7ffee4c88aac")

    try:
        # 模拟一个数值溢出错误
        res = 1 / 0
    except Exception as e:
        # [核心亮点] 使用 exc_info=True 自动抓取并记录崩溃时的堆栈详情
        logger.error("💥 检测到计算异常，正在回溯堆栈：", exc_info=True)

    print(f"\n[PASS] 日志系统部署完毕，请检查: {test_log}")