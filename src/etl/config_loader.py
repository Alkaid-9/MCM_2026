"""
MCM 2026 Problem C: Strategic Configuration Manager (Industrial Refactor)
Role: Singleton Provider for Global Metadata & Governance Rules.
Function: Centralized access to YAML/JSON configs with path resilience.
Standard: Academic Rigor / Production-Grade Robustness.
"""

import yaml
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, List

class ConfigLoader:
    """
    配置加载器（单例模式）：
    确保全系统（包括 23 核并行采样器）共享同一套参数，并提供防御性的访问接口。
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
        """初始化加载所有配置文件，建立项目根目录的绝对坐标。"""
        # 路径自愈逻辑：无论从 main.py 运行还是从子目录运行，都能定位到 /conf
        self.project_root = Path(__file__).resolve().parent.parent.parent
        self.conf_dir = self.project_root / "conf"

        try:
            # 1. 加载核心业务规则 (rules.yaml)
            self._config = self._load_yaml(self.conf_dir / "rules.yaml")

            # 2. 加载评委先验映射 (judges_mapping.json)
            self._judges = self._load_json(self.conf_dir / "judges_mapping.json")

            # 3. 加载贝叶斯先验参数 (priors.yaml)
            self._priors = self._load_yaml(self.conf_dir / "priors.yaml")

            logging.info(f"[Config] 配置文件同步成功。根目录: {self.project_root}")

            # [关键] 强制校验 etl 块是否存在
            if 'etl' not in self._config:
                logging.warning("⚠️ 'rules.yaml' 中缺失 'etl' 配置块，将使用硬编码默认值。")
                self._config['etl'] = {
                    'regex': r"week(\d+)_judge(\d+)_score",
                    'score_range': [1, 10]
                }

        except Exception as e:
            logging.critical(f"配置加载失败: {str(e)}")
            sys.exit(1)

    def _load_yaml(self, path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(f"Missing mandatory YAML config: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}

    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(f"Missing mandatory JSON mapping: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f) or {}

    # =========================================================================
    # 通用访问接口 (Defensive API)
    # =========================================================================

    def get_path(self, key: str) -> str:
        """获取路径并转换为系统绝对路径。"""
        rel_path = self._config.get('paths', {}).get(key)
        if not rel_path:
            raise KeyError(f"Path key '{key}' not defined in rules.yaml")
        return str(self.project_root / rel_path)

    def get_mechanism(self, season: int) -> str:
        """根据赛季判定当前执行的赛制。"""
        m = self._config.get('mechanisms', {})
        if season in m.get('rank_based_seasons', []):
            return "RANK"
        return "PERCENT"

    def get_inference_params(self) -> Dict[str, Any]:
        """获取 MCMC 采样器的超参数。"""
        return self._config.get('inference', {})

    def get_priors_config(self) -> Dict[str, Any]:
        """获取贝叶斯先验分布定义。"""
        return self._priors

    def get_etl_config(self) -> Dict[str, Any]:
        """专门获取 ETL 解析规则，防止之前出现的 KeyError。"""
        return self._config.get('etl', {})

    # =========================================================================
    # 评委寻址高级逻辑 (用于 bindings.cpp 和 transformers.py)
    # =========================================================================
    def get_judge_id(self, season: int, week: int, slot_idx: int) -> str:
        """
        实现多级回退的评委身份识别：
        1. 检查周度异常 (Weekly Anomalies)
        2. 检查赛季覆盖 (Seasonal Overrides)
        3. 回退到默认规则 (Defaults)
        """
        # A. 检查周度异常 (如 S19W04 临时换人)
        anomaly_key = f"s{season}w{week:02d}"
        anomalies = self._judges.get('weekly_anomalies', {}).get(anomaly_key, {})
        if anomalies:
            judge_id = anomalies.get('judge_slots', {}).get(f"slot{slot_idx+1}")
            if judge_id: return judge_id

        # B. 检查赛季覆盖
        season_key = f"season_{season}"
        overrides = self._judges.get('seasonal_configurations', {}).get('overrides', {})
        if season_key in overrides:
            return overrides[season_key][slot_idx]

        # C. 范围回退 (如 S01_S18 的通用席位)
        defaults = self._judges.get('seasonal_configurations', {}).get('defaults', {})
        for range_key, judge_list in defaults.items():
            try:
                start_s, end_s = map(int, range_key.replace('S', '').split('_'))
                if start_s <= season <= end_s:
                    return judge_list[slot_idx]
            except (ValueError, IndexError):
                continue

        return "UNKNOWN"

if __name__ == "__main__":
    # 单元测试：验证单例模式与路径解析
    logging.basicConfig(level=logging.INFO)
    loader1 = ConfigLoader()
    loader2 = ConfigLoader()
    print(f"Singleton test: {id(loader1) == id(loader2)}")
    print(f"ETL Regex: {loader1.get_etl_config().get('regex')}")