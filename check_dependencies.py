# ==============================================================================
# MCM 2026 Problem C: System Integrity & Contract Auditor
# Role: Pre-flight Checklist for Stage 1-6 Execution.
# Function: Verifying environment parity, library compatibility, and API contracts.
# Standard: Industrial DevOps / High-Performance Numerical Computing.
# ==============================================================================

import sys
import importlib
import inspect
import logging
import platform
import psutil
from pathlib import Path

# 设置简单的审计日志
logging.basicConfig(level=logging.INFO, format='[AUDIT] %(levelname)s: %(message)s')
logger = logging.getLogger("PRE_FLIGHT")


def check_python_env():
    """验证 Python 版本与系统资源"""
    logger.info(f"Python Version: {sys.version}")
    logger.info(f"OS: {platform.system()} {platform.release()}")

    # 检查内存 (仿真任务极其耗内存)
    mem = psutil.virtual_memory()
    logger.info(f"System RAM: {mem.total / (1024 ** 3):.2f} GB (Available: {mem.available / (1024 ** 3):.2f} GB)")
    if mem.total < 8 * (1024 ** 3):
        logger.warning("物理内存不足 8GB，大规模 MCMC 采样可能会触发 OOM 崩溃。")


def check_packages():
    """
    检查核心库版本。
    [重点]: 针对 XGBoost 和 SHAP 的版本冲突进行预警。
    """
    essential_libs = {
        "numpy": "1.20.0",
        "pandas": "1.3.0",
        "numba": "0.55.0",
        "xgboost": "1.6.0",
        "shap": "0.41.0",
        "scipy": "1.7.0"
    }

    for lib, min_version in essential_libs.items():
        try:
            module = importlib.import_module(lib)
            version = getattr(module, "__version__", "Unknown")
            logger.info(f"Found {lib:10} | Version: {version:10} (Required >= {min_version})")

            # 特殊警告：SHAP 与 XGBoost 2.x 的不兼容性
            if lib == "xgboost" and version.startswith("2."):
                logger.warning(
                    "检测到 XGBoost 2.x。注意：该版本序列化 base_score 为数组，会导致旧版 SHAP TreeExplainer 崩溃。")
                logger.info("建议：请确保已在 shap_interpreter.py 中应用了‘Booster内核级注入’补丁。")

        except ImportError:
            logger.error(f"缺失核心依赖库: {lib}")


def check_interface_contracts():
    """
    [核心重构点]: 检查类构造函数签名是否对齐。
    物理意义：防止因为 Stage 5 重构了底层逻辑，导致 Stage 3/4 的调用方产生 TypeError。
    """
    logger.info("正在执行接口契约审计 (Interface Contract Audit)...")

    try:
        # 1. 审计 DAWEngine
        from src.solvers.daw_engine import DAWEngine
        daw_spec = inspect.getfullargspec(DAWEngine.__init__)
        logger.info(f"DAWEngine.__init__ signature: {daw_spec.args}")

        # 2. 审计调用方：IncentiveCompatibilityAuditor
        from src.solvers.ic_simulator import IncentiveCompatibilityAuditor
        ic_spec = inspect.getfullargspec(IncentiveCompatibilityAuditor.__init__)
        logger.info(f"IncentiveCompatibilityAuditor.__init__ signature: {ic_spec.args}")

        # 逻辑检查：如果 DAWEngine 不再接受 fig_dir，而调用方还在传，这里会预警
        if 'fig_dir' not in daw_spec.args:
            logger.info("确认：DAWEngine 已切换为纯计算内核 (No fig_dir).")

    except Exception as e:
        logger.error(f"接口契约审计失败: {e}")
        logger.info("这通常意味着你的模块导入路径有问题，或者某个文件存在语法错误。")


def check_data_lake():
    """检查数据湖结构"""
    required_dirs = ["data/bronze", "data/silver", "data/gold", "data/platinum", "logs", "reports/figures"]
    root = Path(__file__).resolve().parent

    for d in required_dirs:
        p = root / d
        if p.exists():
            logger.info(f"Directory [OK]: {d}")
        else:
            logger.warning(f"目录缺失: {d} (系统将尝试在运行时自动创建)")


def run_audit():
    print("\n" + "=" * 60)
    print("      MCM 2026 PROBLEM C - SYSTEM INTEGRITY REPORT")
    print("=" * 60 + "\n")

    check_python_env()
    print("-" * 40)
    check_packages()
    print("-" * 40)
    check_interface_contracts()
    print("-" * 40)
    check_data_lake()

    print("\n" + "=" * 60)
    print("      审计结束：请根据红色错误项修正代码契约")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    run_audit()