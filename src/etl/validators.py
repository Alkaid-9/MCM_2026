# ==============================================================================
# src/etl/validators.py
# Role: Data Quality Assurance (QA) & Audit Surgeon
# Function: Logical consistency checks, survival sanity, and range validation
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from src.etl.config_loader import ConfigLoader


class DataValidator:
    """
    数据验证引擎：负责在 Silver 层生成前进行终极逻辑审计。
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.cfg = ConfigLoader.load_config()
        self.validation_passed = True
        self.logs = []

    def _log_check(self, name: str, passed: bool, message: str):
        status = "PASSED" if passed else "FAILED"
        log_entry = f"[{status}] {name}: {message}"
        self.logs.append(log_entry)
        if not passed:
            self.validation_passed = False
            logging.error(log_entry)
        else:
            logging.info(log_entry)

    def check_nulls(self):
        """检查关键路径列是否存在非法空值"""
        critical_cols = ['celebrity_name', 'season', 'week_num', 'raw_score']
        for col in critical_cols:
            null_count = self.df[col].isna().sum()
            self._log_check(f"Null Check - {col}", null_count == 0, f"发现 {null_count} 个空值")

    def check_score_ranges(self):
        """检查分数是否符合逻辑界限 (考虑到 Bonus 分)"""
        v_cfg = self.cfg['validation']
        limit = self.cfg['etl']['imputation']['bonus_threshold']

        # 允许一定范围内的 Bonus (如 10.5)
        out_of_bounds = self.df[(self.df['raw_score'] < 0) | (self.df['raw_score'] > limit)]
        self._log_check(
            "Score Range Check",
            len(out_of_bounds) == 0,
            f"发现 {len(out_of_bounds)} 行分数超出范围 [0, {limit}]"
        )

    def check_survival_consistency(self):
        """
        【学术核心校验】生存逻辑一致性检查。
        逻辑：如果某选手在第 T 周被淘汰，那就不应该有第 T+1 周的分数记录。
        """
        # 仅针对已标记为 Eliminated 的选手
        elim_df = self.df[self.df['final_status'] == 'Eliminated'].copy()
        if elim_df.empty:
            return

        # 检查是否有关联冲突：当前周 > 淘汰周
        # 注意：eliminated_week 是在 Week X 结束时离开的
        conflict = elim_df[elim_df['week_num'] > elim_df['eliminated_week']]

        self._log_check(
            "Survival Logic Check",
            len(conflict) == 0,
            f"发现 {len(conflict)} 条逻辑冲突：选手淘汰后依然有得分记录"
        )

    def check_season_completeness(self):
        """检查赛季覆盖度"""
        expected = self.cfg['validation']['expected_seasons']
        actual = self.df['season'].nunique()
        self._log_check(
            "Season Completeness",
            actual == expected,
            f"预期 {expected} 个赛季，实际观测到 {actual} 个"
        )

    def run_all(self) -> bool:
        """运行所有体检项目并输出报告"""
        logging.info(f"当前 Silver 表列名: {self.df.columns.tolist()}") # 调试用
        logging.info("--- 启动 Silver Data 质量审计 ---")
        self.check_nulls()
        self.check_score_ranges()
        self.check_survival_consistency()
        self.check_season_completeness()

        # 写入物理日志
        log_path = ConfigLoader.get_path('logs')
        with open(log_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(self.logs))

        if self.validation_passed:
            logging.info("--- 审计通过：数据可用于反演建模 ---")
        else:
            logging.warning("--- 审计未通过：请检查原始数据及清洗逻辑 ---")

        return self.validation_passed


# ------------------------------------------------------------------------------
# 单元测试 (Mock Data)
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 创建一个故意出错的测试集
    test_df = pd.DataFrame({
        'celebrity_name': ['Star_A', 'Star_A'],
        'season': [1, 1],
        'week_num': [1, 2],
        'raw_score': [8.0, 9.0],
        'final_status': ['Eliminated', 'Eliminated'],
        'eliminated_week': [1.0, 1.0]  # Week 2 的分就是逻辑冲突
    })

    validator = DataValidator(test_df)
    validator.run_all()