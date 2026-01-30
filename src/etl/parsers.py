# ==============================================================================
# src/etl/parsers.py
# Role: Non-structured to Structured Data Surgeon
# Function: Extracting numeric signals from messy text and validating column meta
# ==============================================================================

import pandas as pd
import numpy as np
import re
import logging
from src.etl.config_loader import ConfigLoader


class TextParser:
    """
    文本解析引擎：负责处理结果列、人名标准化以及动态列名解析。
    """

    @staticmethod
    def parse_results_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        解析 'results' 列。
        核心逻辑：
        1. 提取淘汰周 (Eliminated Week X) -> int
        2. 归一化最终状态 (Winner, RunnerUp, Eliminated, Withdrew)
        """
        logging.info("开始解析 Results 状态列...")

        # 这里的正则要小心：数据中存在 'Eliminated Week 110' 这种错误，推测是 Week 1 且 Placement 10
        # 只提取第一组数字
        df['eliminated_week'] = df['results'].str.extract(r'Week\s*(\d+)').astype(float)

        # 状态映射逻辑
        # 使用 np.select 实现向量化分支判断，比 apply(lambda) 快 100 倍
        conds = [
            df['results'].str.contains('1st Place|Winner', case=False, na=False),
            df['results'].str.contains('2nd Place|Runner', case=False, na=False),
            df['results'].str.contains('3rd Place|Finalist', case=False, na=False),
            df['results'].str.contains('Eliminated', case=False, na=False),
            df['results'].str.contains('Withdrew', case=False, na=False)
        ]
        choices = ['Winner', 'RunnerUp', 'Finalist', 'Eliminated', 'Withdrew']
        df['final_status'] = np.select(conds, choices, default='Active')

        return df

    @staticmethod
    def standardize_strings(df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗所有人名、行业、地名列的空格和大小写。
        这是防止模型因 'Athlete ' 和 'Athlete' 产生特征偏离的关键。
        """
        str_cols = df.select_dtypes(include=['object']).columns
        for col in str_cols:
            df[col] = df[col].astype(str).str.strip().str.title()
        return df

    @classmethod
    def get_score_column_map(cls, df: pd.DataFrame) -> dict:
        """
        【动态列解析】
        根据 rules.yaml 中的正则，找出所有评分列，并返回 {col_name: (week, judge)}
        """
        rules = ConfigLoader.get_etl_rules()
        pattern = re.compile(rules['score_column_regex'])

        score_map = {}
        for col in df.columns:
            match = pattern.match(col)
            if match:
                week_num, judge_id = match.groups()
                score_map[col] = (int(week_num), int(judge_id))

        logging.info(f"解析到 {len(score_map)} 个有效的评分观测列。")
        return score_map


# ------------------------------------------------------------------------------
# 单元测试代码块
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 模拟一段脏数据测试正则
    test_data = pd.DataFrame({
        'results': ['1st Place', 'Eliminated Week 3', 'Withdrew', 'Eliminated Week 11'],
        'celebrity_name': [' John Doe ', 'Jane smith', 'Bob ', 'ALICE']
    })

    parser = TextParser()
    df_test = parser.parse_results_column(test_data)
    df_test = parser.standardize_strings(df_test)

    print("Parsed Data Preview:")
    print(df_test[['celebrity_name', 'final_status', 'eliminated_week']])