# ==============================================================================
# src/etl/config_loader.py
# Role: Singleton Strategic Configuration Provider (v5.2 - Edition)
# Function: Centralized management of global rules, priors, and metadata.
# Key Logic: 3-Layer Judge Identification Fallback & Defensive Path Resolution.
# Standard: Industrial Reliability / Singleton Pattern / Type Safety.
# ==============================================================================

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Union

class ConfigLoader:
    """
    单例配置加载器：
    负责将 rules.yaml (全局规则), priors.yaml (分布先验), judges_mapping.json (裁判元数据)
    熔炼为内存中的只读配置对象。

    [设计哲学]:
    1. Single Source of Truth: 所有参数修改仅在 conf/ 目录发生。
    2. Fail-Fast: 关键配置缺失直接熔断，防止脏数据污染下游计算。
    3. Path Agnostic: 自动锚定项目根目录，无视启动位置。
    """

    _instance = None
    _initialized = False

    # 内存缓存
    _config: Dict[str, Any] = {}
    _judges: Dict[str, Any] = {}
    _priors: Dict[str, Any] = {}
    _project_root: Path = None

    def __new__(cls):
        """实现线程安全的单例模式"""
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        """
        初始化加载：自动定位根目录并执行一次性磁盘 I/O。
        使用 _initialized 标志位防止单例重复加载。
        """
        if self._initialized:
            return

        # 配置日志（临时使用 basicConfig，避免循环依赖 logger.py）
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger("CONFIG_LOADER")

        # 1. 路径自愈逻辑：从当前文件向上回溯到达项目根目录
        # 文件位置: src/etl/config_loader.py -> ... -> root
        self._project_root = Path(__file__).resolve().parent.parent.parent
        self._conf_dir = self._project_root / "conf"

        self._load_all_configs()
        self._initialized = True

    def _load_all_configs(self):
        """加载所有核心配置文件"""
        try:
            # A. 加载核心业务规则 (rules.yaml)
            rules_path = self._conf_dir / "rules.yaml"
            if not rules_path.exists():
                raise FileNotFoundError(f"核心配置文件缺失: {rules_path}")
            with open(rules_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}

            # B. 加载分布先验参数 (priors.yaml)
            priors_path = self._conf_dir / "priors.yaml"
            if priors_path.exists():
                with open(priors_path, 'r', encoding='utf-8') as f:
                    self._priors = yaml.safe_load(f) or {}
            else:
                self.logger.warning(f"先验配置缺失: {priors_path}，将使用默认 Zipf 分布。")

            # C. 加载评委映射表 (judges_mapping.json)
            judges_path = self._conf_dir / "judges_mapping.json"
            if judges_path.exists():
                with open(judges_path, 'r', encoding='utf-8') as f:
                    self._judges = json.load(f) or {}

            self.logger.info(f"配置加载完成。项目根目录锁定为: {self._project_root}")

        except Exception as e:
            self.logger.critical(f"配置系统致命崩溃: {str(e)}", exc_info=True)
            raise RuntimeError("System Halt: Configuration Failure")

    # ==========================================================================
    # 核心 API 接口 (供各 Stage 子系统调用)
    # ==========================================================================

    def load_config(self) -> Dict[str, Any]:
        """返回 rules.yaml 的完整字典 (Raw Access)"""
        return self._config

    def get_path(self, key: str) -> str:
        """
        [全流程] 路径解析器。
        将 rules.yaml 中的相对路径 (data/bronze/...) 解析为当前系统的绝对路径。
        """
        rel_path = self._config.get('paths', {}).get(key)
        if not rel_path:
            raise KeyError(f"rules.yaml 的 'paths' 块中未定义键: '{key}'")

        # 兼容不同操作系统的路径分隔符
        abs_path = self._project_root / Path(rel_path)
        return str(abs_path)

    # --- Stage 1: ETL 配置 ---

    def get_etl_config(self) -> Dict[str, Any]:
        """[修复点] 专门返回 etl 块配置，解决 transformers.py 的调用崩溃"""
        return self._config.get('etl', {})

    def get_features_config(self) -> Dict[str, Any]:
        """返回特征工程配置 (Mapping, Age segmentation)"""
        return self._config.get('features', {})

    def get_judge_id(self, season: int, week: int, slot_idx: int) -> str:
        """
        [复杂业务逻辑] 评委身份识别引擎。
        为了处理 DWTS 34 年历史中的人员变动，实现了三级回退寻址：
        优先级：
        1. 周度异常 (Weekly Anomaly): 如 S19W04 客座评委。
        2. 赛季覆盖 (Season Override): 如 S29 评委团变动。
        3. 范围默认 (Default Range): 如 S01-S18 的稳定阵容。

        :param slot_idx: 0-based index (0, 1, 2, 3)
        :return: Judge ID (e.g., 'LEN', 'CAI') or 'UNKNOWN'
        """
        # A. 检查周度异常 (Format: s19w04)
        week_key = f"s{season}w{week:02d}"
        anomalies = self._judges.get('weekly_anomalies', {}).get(week_key, {})
        if anomalies:
            # slot_key: "slot1", "slot2"...
            slot_key = f"slot{slot_idx + 1}"
            jid = anomalies.get('judge_slots', {}).get(slot_key)
            if jid: return jid

        # B. 检查赛季覆盖
        season_key = f"season_{season}"
        overrides = self._judges.get('seasonal_configurations', {}).get('overrides', {})
        if season_key in overrides:
            try:
                # overrides 存储的是列表
                judge_list = overrides[season_key]
                if slot_idx < len(judge_list):
                    return judge_list[slot_idx]
            except IndexError:
                pass

        # C. 范围回退 (Format: S01_S18)
        defaults = self._judges.get('seasonal_configurations', {}).get('defaults', {})
        for range_key, judge_list in defaults.items():
            try:
                # 解析范围字符串 "S01_S18"
                parts = range_key.replace('S', '').split('_')
                if len(parts) == 2:
                    start_s, end_s = int(parts[0]), int(parts[1])
                    if start_s <= season <= end_s:
                        if slot_idx < len(judge_list):
                            return judge_list[slot_idx]
            except (ValueError, IndexError):
                continue

        return "UNKNOWN_JUDGE"

    # --- Stage 2: Inference 配置 ---

    def get_inference_params(self) -> Dict[str, Any]:
        """获取 MCMC 采样器超参数 (N_chains, Samples)"""
        return self._config.get('inference', {})

    def get_priors_config(self) -> Dict[str, Any]:
        """获取 priors.yaml 的概率模型参数 (Zipf Alpha, Dirichlet Strength)"""
        return self._priors

    def get_mechanism(self, season: int) -> str:
        """
        [核心逻辑] 判定某赛季属于哪种赛制。
        用于自动切换 C++ 内核的似然函数逻辑。
        """
        m_cfg = self._config.get('mechanisms', {})

        if season in m_cfg.get('rank_based_seasons', []):
            return "RANK"

        if season in m_cfg.get('percent_based_seasons', []):
            return "PERCENT"

        return "UNKNOWN"

    def is_judge_save_active(self, season: int) -> bool:
        """判断该赛季是否引入了评委救人机制 (S28+)"""
        js_cfg = self._config.get('mechanisms', {}).get('judge_save', {})
        active_from = js_cfg.get('active_from', 999)
        return season >= active_from

    # --- Stage 4: Design 配置 ---

    def get_design_params(self) -> Dict[str, Any]:
        """获取 Task 4 机制设计的搜索空间"""
        return self._config.get('task4_mechanism_design', {})

# --- 单元测试 ---
if __name__ == "__main__":
    # 模拟环境测试
    try:
        loader = ConfigLoader()

        print("\n=== Config Integrity Check ===")
        print(f"Project Root: {loader._project_root}")

        # 测试路径解析
        print(f"Bronze Path: {loader.get_path('bronze_raw')}")

        # 测试评委寻址逻辑
        print("\n=== Judge ID Resolution Test ===")
        # S19 W04 是著名的客座评委周
        j_s19w04_3 = loader.get_judge_id(19, 4, 2) # Slot 3
        print(f"S19 W04 Slot 3 (Should be Guest): {j_s19w04_3}")

        # S01 是标准三人组
        j_s01_0 = loader.get_judge_id(1, 1, 0) # Slot 1
        print(f"S01 W01 Slot 1 (Should be Len/Carrie): {j_s01_0}")

        # 测试机制判定
        mech_s27 = loader.get_mechanism(27)
        mech_s28 = loader.get_mechanism(28)
        print(f"\nMechanism S27 (Expect PERCENT): {mech_s27}")
        print(f"Mechanism S28 (Expect RANK): {mech_s28}")

        print("\n[PASS] ConfigLoader is ready for deployment.")

    except Exception as e:
        print(f"\n[FAIL] Test crashed: {e}")