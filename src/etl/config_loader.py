# ==============================================================================
# src/etl/config_loader.py
# Role: Configuration Manager (Industrial Standard)
# Function: Centralized access to conf/rules.yaml with path auto-resolution
# ==============================================================================

import yaml
import os
from pathlib import Path
import logging


class ConfigLoader:
    """
    配置加载类：负责定位、读取并解析 rules.yaml。
    采用静态方法设计，方便全局调用。
    """

    @staticmethod
    def get_project_root() -> Path:
        """
        自动定位项目根目录。
        逻辑：从当前文件向上回溯两层 (etl -> src -> root)。
        """
        return Path(__file__).resolve().parent.parent.parent

    @classmethod
    def load_config(cls):
        """
        加载并返回完整的 YAML 配置字典。
        """
        root = cls.get_project_root()
        config_path = root / "conf" / "rules.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"核心配置文件缺失: {config_path}. 请检查项目结构！")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config
        except Exception as e:
            logging.error(f"解析 rules.yaml 失败: {str(e)}")
            raise

    @classmethod
    def get_path(cls, path_key: str) -> str:
        """
        快捷获取绝对路径。
        例如: ConfigLoader.get_path('silver_data')
        """
        config = cls.load_config()
        relative_path = config['paths'].get(path_key)
        if not relative_path:
            raise KeyError(f"路径配置中未找到键: {path_key}")

        # 将相对路径转换为绝对路径，确保不同环境下运行一致
        return str(cls.get_project_root() / relative_path)

    @classmethod
    def get_etl_rules(cls) -> dict:
        """获取数据清洗专用规则"""
        return cls.load_config().get('etl', {})

    @classmethod
    def get_mechanism_rules(cls) -> dict:
        """获取赛制逻辑规则 (Rank vs Percent)"""
        return cls.load_config().get('mechanisms', {})


# ------------------------------------------------------------------------------
# 单元测试 (仅当直接运行此文件时触发)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Project Root: {ConfigLoader.get_project_root()}")
    try:
        cfg = ConfigLoader.load_config()
        print("Successfully loaded configuration.")
        print(f"Bronze Path: {ConfigLoader.get_path('bronze_data')}")
        print(f"Rank Based Seasons: {ConfigLoader.get_mechanism_rules()['rank_based_seasons']}")
    except Exception as e:
        print(f"Error: {e}")