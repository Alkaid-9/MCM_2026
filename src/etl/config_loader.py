"""
MCM 2026 Problem C: Strategic Configuration Manager
Role: Singleton Provider for Global Governance Metadata
Standard: Industrial HPC / Academic Rigor
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List


class ConfigLoader:
    """
    单例配置加载器，负责管理 rules.yaml, judges_mapping.json 和 priors.yaml。
    设计要点：
    1. 路径自动回溯：确保从任何子目录运行脚本都能定位到 /conf 目录。
    2. 缓存一致性：单例模式避免在 23 核并行采样时重复读取磁盘。
    3. 业务抽象：提供高级 API 屏蔽底层字典嵌套。
    """

    _instance = None
    _config: Dict[str, Any] = {}
    _judges: Dict[str, Any] = {}
    _priors: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化加载所有配置文件"""
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.conf_dir = self.project_root / "conf"

        # 加载核心规则 (YAML)
        self._config = self._load_yaml(self.conf_dir / "rules.yaml")
        # 加载评委映射 (JSON)
        self._judges = self._load_json(self.conf_dir / "judges_mapping.json")
        # 加载先验参数 (YAML)
        self._priors = self._load_yaml(self.conf_dir / "priors.yaml")

        logging.info(f"[Config] Global configuration synchronized from {self.conf_dir}")

    def _load_yaml(self, path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(f"Missing config: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(f"Missing mapping: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    # --- 通用路径 API ---
    def get_path(self, key: str) -> str:
        """获取并格式化绝对路径"""
        rel_path = self._config.get('paths', {}).get(key)
        if not rel_path:
            raise KeyError(f"Path key '{key}' not defined in rules.yaml")
        return str(self.project_root / rel_path)

    # --- 赛制逻辑 API ---
    def get_mechanism(self, season: int) -> str:
        """核心：判断指定赛季采用的是 RANK 还是 PERCENT 机制"""
        m = self._config['mechanisms']
        if season in m['rank_based_seasons']:
            return "RANK"
        if season in m['percent_based_seasons']:
            return "PERCENT"
        return "UNKNOWN"

    def is_judge_save_active(self, season: int) -> bool:
        """判断该赛季是否引入了评委救人机制 (S28+)"""
        cfg = self._config['mechanisms']['judge_save']
        return season >= cfg['active_from']

    # --- 评委逻辑 API (深度抽象) ---
    def get_judge_id(self, season: int, week: int, slot_idx: int) -> str:
        """
        核心分级寻址逻辑：
        1. 检查 weekly_anomalies (周度异常换人)
        2. 检查 seasonal_configurations.overrides (赛季覆盖)
        3. 回退到 seasonal_configurations.defaults (默认规则)
        """
        # A. 检查周度异常 (如 s19w04)
        anomaly_key = f"s{season}w{week:02d}"
        if anomaly_key in self._judges.get('weekly_anomalies', {}):
            slots = self._judges['weekly_anomalies'][anomaly_key].get('judge_slots', {})
            slot_key = f"slot{slot_idx + 1}"
            if slot_key in slots:
                return slots[slot_key]

        # B. 检查赛季覆盖或默认
        conf = self._judges['seasonal_configurations']
        overrides = conf.get('overrides', {})
        defaults = conf.get('defaults', {})

        season_key = f"season_{season}"
        if season_key in overrides:
            return overrides[season_key][slot_idx]

        # 范围回退逻辑 (例如 S01_S18)
        for range_key, judge_list in defaults.items():
            start_s, end_s = map(int, range_key.replace('S', '').split('_'))
            if start_s <= season <= end_s:
                return judge_list[slot_idx]

        return "UNKNOWN"

    def get_judge_prior(self, judge_code: str) -> Dict[str, float]:
        """获取评委的贝叶斯先验 (mu, sigma)"""
        return self._judges['judge_registry'].get(judge_code, self._judges['judge_registry']['UNKNOWN'])[
            'bayesian_prior']

    # --- 先验与采样控制 API ---
    def get_inference_params(self) -> Dict:
        """获取 MCMC 采样超参数"""
        return self._config['inference']

    def get_priors_config(self) -> Dict:
        """获取先验分布定义"""
        return self._priors


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    loader = ConfigLoader()

    # 测试路径解析
    print(f"Bronze Data Path: {loader.get_path('bronze_raw')}")

    # 测试赛制判定
    print(f"Season 27 Mechanism: {loader.get_mechanism(27)}")  # 应为 PERCENT
    print(f"Season 28 Mechanism: {loader.get_mechanism(28)}")  # 应为 RANK

    # 测试评委寻址逻辑
    print(f"S19 W01 Slot 3: {loader.get_judge_id(19, 1, 2)}")  # 正常 JH
    print(f"S19 W04 Slot 4: {loader.get_judge_id(19, 4, 3)}")  # 异常 GUEST

    # 测试先验获取
    print(f"Len Goodman (LG) Prior: {loader.get_judge_prior('LG')}")