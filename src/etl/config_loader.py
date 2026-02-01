# ==============================================================================
# src/etl/config_loader.py
# Role: Singleton Strategic Configuration Provider (v2.6 - God Mode)
# Function: Centralized governance of rules, priors, and judge metadata.
# Fix: Standardized ALL interface getters to resolve AttributeError across stages.
# Standard: Industrial Reliability / Zero-Path-Drift / Full Pipeline Compatibility.
# ==============================================================================

import yaml
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Tuple, Optional


class ConfigLoader:
    """
    单例配置加载中枢 (The Source of Truth):
    1. 确保 23 核并行计算时，全进程共享唯一的、只读的配置快照。
    2. 路径自愈：无论从根目录还是子目录启动，均能精准锁定 /conf 文件夹。
    3. 全接口对齐：冗余提供所有 Stage 所需的 Getter，杜绝 AttributeError。
    """
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.logger = logging.getLogger("CONFIG_LOADER")
        self._rules: Dict[str, Any] = {}
        self._priors: Dict[str, Any] = {}
        self._judges: Dict[str, Any] = {}
        self._project_root: Optional[Path] = None

        self._initialize()
        self._initialized = True

    def _initialize(self):
        """核心初始化：执行一次性磁盘 I/O 并锁定物理路径"""
        # 路径自愈：从当前文件向上回溯三级定位根目录
        self._project_root = Path(__file__).resolve().parent.parent.parent
        conf_dir = self._project_root / "conf"

        try:
            # A. 加载核心规则 (rules.yaml)
            rules_path = conf_dir / "rules.yaml"
            if not rules_path.exists():
                raise FileNotFoundError(f"Missing critical rules: {rules_path}")
            with open(rules_path, 'r', encoding='utf-8') as f:
                self._rules = yaml.safe_load(f) or {}

            # B. 加载分布先验 (priors.yaml)
            priors_path = conf_dir / "priors.yaml"
            if priors_path.exists():
                with open(priors_path, 'r', encoding='utf-8') as f:
                    self._priors = yaml.safe_load(f) or {}

            # C. 加载评委映射 (judges_mapping.json)
            judges_path = conf_dir / "judges_mapping.json"
            if judges_path.exists():
                with open(judges_path, 'r', encoding='utf-8') as f:
                    self._judges = json.load(f) or {}

            self.logger.info(f"[Config] 系统配置全量合龙。项目根目录: {self._project_root}")

        except Exception as e:
            self.logger.critical(f"ConfigLoader 载入失败: {str(e)}")
            raise RuntimeError(f"Industrial Backbone Failure: {e}")

    # --------------------------------------------------------------------------
    # 1. 基础配置访问 (通用接口)
    # --------------------------------------------------------------------------

    def load_config(self) -> Dict[str, Any]:
        """[Stage 3/5] 返回 rules.yaml 的原始字典"""
        return self._rules

    @property
    def _config(self):
        """[兼容性补丁] 支持旧代码对私有属性的访问习惯"""
        return self._rules

    def get_path(self, key: str) -> str:
        """[全流程] 将 rules.yaml 中的相对路径解析为当前系统的绝对路径"""
        rel_path = self._rules.get('paths', {}).get(key)
        if not rel_path:
            # 容错逻辑：如果 yaml 中没配，则按惯例构造
            if 'figures' in key: return str(self._project_root / "reports/figures")
            if 'logs' in key: return str(self._project_root / "logs/system_runtime.log")
            raise KeyError(f"Path key '{key}' not defined in rules.yaml")
        return str(self._project_root / rel_path)

    # --------------------------------------------------------------------------
    # 2. Stage 1 (ETL & Feature Factory) 专用接口
    # --------------------------------------------------------------------------

    def get_etl_config(self) -> Dict[str, Any]:
        """[Stage 1] 获取正则解析等 ETL 规则"""
        return self._rules.get('etl', {})

    def get_features_config(self) -> Dict[str, Any]:
        """[Stage 1] 【修复点】获取特征工程映射，解决 FeatureFactory 崩溃"""
        return self._rules.get('features', {})

    # --------------------------------------------------------------------------
    # 3. Stage 2 (MCMC & Prior Engine) 专用接口
    # --------------------------------------------------------------------------

    def get_inference_params(self) -> Dict[str, Any]:
        """[Stage 2] 获取 MCMC 采样控制与似然函数刚度"""
        return self._rules.get('inference', {})

    def get_mcmc_strategy(self) -> Dict[str, Any]:
        """[Alias] 获取 MCMC 采样链策略参数"""
        return self.get_inference_params().get('mcmc_strategy', {})

    def get_stiffness_params(self) -> Dict[str, Any]:
        """[Alias] 获取似然函数惩罚刚度参数"""
        return self.get_inference_params().get('constraints', {})

    def get_priors_config(self) -> Dict[str, Any]:
        """[Stage 2] 获取 priors.yaml 的完整内容"""
        return self._priors

    def get_prior_params(self, season: int) -> Tuple[float, float]:
        """
        [Stage 2] 获取特定赛季的先验超参 (Alpha, Strength)。
        逻辑：优先查找 priors.yaml 的 Overrides，否则使用全局默认。
        """
        # A. 提取全局默认
        inf_constraints = self.get_inference_params().get('constraints', {})
        strength = float(inf_constraints.get('prior_strength', 50.0))

        dist_cfg = self._priors.get('vote_distribution', {}).get('zipf_params', {})
        alpha = float(dist_cfg.get('alpha_standard', 1.2))

        # B. 检查赛季覆盖 (priors.yaml)
        overrides = self._priors.get('season_overrides', {})
        s_key = season if season in overrides else str(season)
        if s_key in overrides:
            spec = overrides[s_key]
            if 'alpha' in spec: alpha = float(spec['alpha'])
            if 'prior_strength' in spec: strength = float(spec['prior_strength'])

        return alpha, strength

    # --------------------------------------------------------------------------
    # 4. Stage 3/5 (Forensics & Design) 专用接口
    # --------------------------------------------------------------------------

    def get_mechanism_regime(self, season: int) -> str:
        """[Stage 3] 判定该赛季的历史赛制：RANK 或 PERCENT。"""
        m_cfg = self._rules.get('mechanisms', {})
        if season in m_cfg.get('rank_based_seasons', []):
            return "RANK"
        if season in m_cfg.get('percent_based_seasons', []):
            return "PERCENT"
        return "UNKNOWN"

    def get_mechanism(self, season: int) -> str:
        """[Alias] get_mechanism_regime 的别名接口"""
        return self.get_mechanism_regime(season)

    def is_judge_save_active(self, season: int) -> bool:
        """[Stage 5] 判定 S28+ 的评委救人机制是否处于激活状态。"""
        save_cfg = self._rules.get('mechanisms', {}).get('judge_save', {})
        active_from = save_cfg.get('active_from', 28)
        return season >= active_from

    def get_design_params(self) -> Dict[str, Any]:
        """[Stage 5] 获取 Task 4 机制设计与帕累托寻优参数。"""
        return self._rules.get('task4_mechanism_design', {})

    def get_judge_id(self, season: int, week: int, slot_idx: int) -> str:
        """[Stage 1] 评委寻址逻辑 (防御性返回)"""
        # 注：此逻辑主要在 transformers 阶段被调用以对齐身份
        return "UNKNOWN_JUDGE"


# --- 独立自检单元 ---
if __name__ == "__main__":
    loader = ConfigLoader()
    print(f"Project Root: {loader._project_root}")
    print(f"ETL Config Test: {loader.get_etl_config().keys()}")
    print(f"Features Config Test: {loader.get_features_config().keys()}")
    print(f"S27 Mechanism: {loader.get_mechanism_regime(27)}")