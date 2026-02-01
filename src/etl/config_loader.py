# ==============================================================================
# src/etl/config_loader.py
# Role: Singleton Strategic Configuration Provider (v2.2 - Final Alignment)
# Fix: Added get_etl_config() and aligned all downstream pipeline APIs.
# Standard: Industrial Reliability / Defensive Programming / Zero-Path-Drift.
# ==============================================================================

import yaml
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """
    单例配置加载器：
    管理 rules.yaml (全局规则), priors.yaml (分布先验), judges_mapping.json (裁判元数据)。

    【核心修复】：
    - 补全 get_etl_config() 接口，修复 transformers.py 的调用崩溃。
    - 强化 get_judge_id() 的多级回退逻辑，适配历史长周期数据。
    """
    _instance = None
    _config: Dict[str, Any] = {}
    _judges: Dict[str, Any] = {}
    _priors: Dict[str, Any] = {}
    _project_root: Path = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigLoader, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """初始化加载：自动定位根目录并执行一次性磁盘 I/O"""
        # 路径自愈逻辑：从当前文件向上回溯三层到达项目根目录
        # 文件位置: /project/src/etl/config_loader.py
        self._project_root = Path(__file__).resolve().parent.parent.parent
        conf_dir = self._project_root / "conf"

        try:
            # 1. 加载核心业务规则 (rules.yaml)
            rules_path = conf_dir / "rules.yaml"
            with open(rules_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}

            # 2. 加载评委先验映射 (judges_mapping.json)
            judges_path = conf_dir / "judges_mapping.json"
            if judges_path.exists():
                with open(judges_path, 'r', encoding='utf-8') as f:
                    self._judges = json.load(f) or {}

            # 3. 加载分布先验参数 (priors.yaml)
            priors_path = conf_dir / "priors.yaml"
            if priors_path.exists():
                with open(priors_path, 'r', encoding='utf-8') as f:
                    self._priors = yaml.safe_load(f) or {}

            logging.debug(f"[ConfigLoader] 已锁定根目录: {self._project_root}")

        except Exception as e:
            # 配置文件加载失败是系统级故障，必须熔断
            raise RuntimeError(f"ConfigLoader 关键配置文件加载失败: {str(e)}")

    # --------------------------------------------------------------------------
    # 核心 API 接口 (供各 Stage 子系统调用)
    # --------------------------------------------------------------------------

    def load_config(self) -> Dict[str, Any]:
        """[Stage 3/5] 返回 rules.yaml 完整字典"""
        return self._config

    def get_etl_config(self) -> Dict[str, Any]:
        """[Stage 1] 【修复点】专门返回 etl 块配置，解决 AttributeError"""
        return self._config.get('etl', {})

    def get_inference_params(self) -> Dict[str, Any]:
        """[Stage 2] 获取 MCMC 采样器超参数"""
        return self._config.get('inference', {})

    def get_priors_config(self) -> Dict[str, Any]:
        """[Stage 2] 获取 priors.yaml 的概率模型参数"""
        return self._priors

    def get_path(self, key: str) -> str:
        """[全流程] 将 rules.yaml 中的相对路径解析为当前系统的绝对路径"""
        rel_path = self._config.get('paths', {}).get(key)
        if not rel_path:
            raise KeyError(f"rules.yaml 的 paths 块中未定义键: '{key}'")
        return str(self._project_root / rel_path)

    def get_mechanism(self, season: int) -> str:
        """[Stage 3/5] 赛制判定逻辑"""
        m_cfg = self._config.get('mechanisms', {})
        if season in m_cfg.get('rank_based_seasons', []):
            return "RANK"
        return "PERCENT"

    def get_judge_id(self, season: int, week: int, slot_idx: int) -> str:
        """
        [Stage 1] 评委身份识别引擎。
        优先级：1.周度异常(Weekly Anom) > 2.赛季覆盖(Season Override) > 3.范围默认(Defaults)
        """
        # A. 检查周度异常 (s19w04 格式)
        week_key = f"s{season}w{week:02d}"
        anomalies = self._judges.get('weekly_anomalies', {}).get(week_key, {})
        if anomalies:
            jid = anomalies.get('judge_slots', {}).get(f"slot{slot_idx + 1}")
            if jid: return jid

        # B. 检查赛季覆盖
        season_key = f"season_{season}"
        overrides = self._judges.get('seasonal_configurations', {}).get('overrides', {})
        if season_key in overrides:
            try:
                return overrides[season_key][slot_idx]
            except IndexError:
                pass

        # C. 范围回退 (S01_S18 格式)
        defaults = self._judges.get('seasonal_configurations', {}).get('defaults', {})
        for range_key, judge_list in defaults.items():
            try:
                parts = range_key.replace('S', '').split('_')
                if len(parts) == 2:
                    start_s, end_s = int(parts[0]), int(parts[1])
                    if start_s <= season <= end_s:
                        return judge_list[slot_idx]
            except (ValueError, IndexError):
                continue

        return "UNKNOWN_JUDGE"