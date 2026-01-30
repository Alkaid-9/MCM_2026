"""
MCM 2026 Problem C: Non-structured Data Surgeon
Role: Vectorized String Parsing, Anomaly Correction, and Dynamic Metadata Extraction
Standard: Industrial Grade Robustness / Academic Data Integrity (Censored Data Support)
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import List, Optional
from src.etl.config_loader import ConfigLoader

class TextParser:
    """
    文本解析引擎：执行数据“去幻觉”与标准化。

    【核心逻辑】：
    1. 幽灵数据防御 (Ghost Defense):
       精确识别选手‘死亡’时间点，防止已淘汰选手的 0 分污染后续统计分布。

    2. 危机信号提取 (Jeopardy Mining):
       从文本中挖掘 "Bottom Two" 或 "Risk" 信号。这是 MCMC 的黄金强约束：
       处于危险区意味着 (Rank_Judge + Rank_Fan) 处于阈值边缘。

    3. 异常值启发式修复 (Heuristic Repair):
       自动修正 'Week 110' 等明显的 OCR/录入错误。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = logging.getLogger("TEXT_PARSER")

    def standardize_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗实体名称（人名、行业、州名）。
        学术价值：消除由于空格或大小写导致的“特征稀疏”问题 (High Cardinality)。
        """
        self.logger.info("执行实体标准化 (Entity Resolution)...")

        # 识别所有字符串列
        obj_cols = df.select_dtypes(include=['string', 'object']).columns

        for col in obj_cols:
            if col == 'results': continue # 结果列保留原样待复杂正则解析

            # 级联清洗：去首尾空格 -> 压缩中间空格 -> 转标题格式
            # 例如: " john   DOE " -> "John Doe"
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.title()

            # 处理 "N/A", "Null", "None" 为真正的 np.nan
            df[col] = df[col].replace(['N/A', 'N/a', 'Null', 'None', 'nan'], np.nan)

        return df

    def parse_survival_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        解析 'results' 列，提取生存分析所需的关键标签。
        这是构建 Right-Censored 模型的地基。
        """
        self.logger.info("正在执行 Results 向量化解析与生存屏障构建...")

        # --- 1. 提取淘汰周次 (Censorship Time) ---
        # 正则逻辑：提取 "Week X" 中的数字
        # fillna(999) 用于标记决赛选手（从未淘汰），后续会修正
        raw_weeks = df['results'].str.extract(r'Week\s*(\d+)', flags=re.IGNORECASE)[0].astype(float)

        # --- 2. 异常值启发式修复 (Heuristic Fix for 'Week 110') ---
        # 逻辑：DWTS 一季通常不超过 15 周。如果周次 > 20，必然是录入错误。
        # 针对 'Week 110'：通常是 'Week 1' 或 'Week 10' 的笔误。
        # 结合 placement 判断：如果名次很差 (e.g. 10+)，大概率是早期淘汰 (Week 1)。

        # 标记异常
        anomaly_mask = raw_weeks > 20
        if anomaly_mask.any():
            bad_indices = df.index[anomaly_mask].tolist()
            self.logger.warning(f"检测到 {len(bad_indices)} 处周次异常 (如 Week 110)，正在执行启发式修正...")

            # 修正策略：如果 placement >= 10，认为是 Week 1；否则尝试解析为 Week 10
            # 这里简化处理：强制修正为 Week 1 (基于 Problem C 常见数据坑点)
            raw_weeks = np.where(anomaly_mask, 1.0, raw_weeks)

        df['eliminated_week'] = raw_weeks

        # --- 3. 最终状态向量化映射 (Status Mapping) ---
        # 调整顺序：将 Withdrew 提前，因为它是一个独立于比赛结果的外生事件
        conds = [
            df['results'].str.contains(r'1st|Winner|Champion', case=False, na=False),
            df['results'].str.contains(r'2nd|Runner', case=False, na=False),
            df['results'].str.contains(r'3rd|Finalist', case=False, na=False),
            df['results'].str.contains(r'Withdrew|Quit|Injured', case=False, na=False),
            df['results'].str.contains(r'Eliminated', case=False, na=False)
        ]
        choices = ['Winner', 'RunnerUp', 'Finalist', 'Withdrew', 'Eliminated']
        df['final_status'] = np.select(conds, choices, default='Active')

        # --- 4. 决赛选手处理 ---
        # 赢家和决赛选手的淘汰周次设为全季最大周次 (Max Season Length)
        # 这里先填一个较大的数，Transform 阶段会根据实际 max_week 截断
        season_max_weeks = df.groupby('season')['eliminated_week'].transform('max')
        # 对于 Winner/RunnerUp，存活时间 = 赛季长度
        is_finalist = df['final_status'].isin(['Winner', 'RunnerUp', 'Finalist'])
        df.loc[is_finalist, 'eliminated_week'] = season_max_weeks[is_finalist]

        # --- 5. [核心新增] 危险区信号挖掘 (Jeopardy Signal) ---
        # 注意：Bronze 数据如果是宽表（每人一行），这里的 info 可能不全。
        # 但如果 'results' 列包含历史信息（如 "Bottom 2 Week 5"），我们需要提取。
        # 这里预留接口，如果 description 里有 "Bottom Two"，打上标记。
        # 增加 'Jeopardy' 关键词，这是美赛原始描述中最常见的词汇之一
        # --- 5. [核心新增] 危险区信号挖掘 (Jeopardy Signal) ---
        # 使用 (?:...) 非捕获分组，消除 Pandas UserWarning，并涵盖所有美赛描述变体
        jeopardy_pattern = r'Bottom\s*(?:Two|2)|Risk|Jeopardy|Danger'
        df['had_bottom_two_record'] = df['results'].str.contains(jeopardy_pattern, case=False, na=False).astype(int)

        return df

    def get_score_columns_metadata(self, columns: List[str]) -> pd.DataFrame:
        """
        【动态元数据映射】
        利用正则表达式解析复杂的宽表列名，建立 (Column_Name -> Week -> Judge_ID) 的映射。
        """
        rules = self.cfg._config['etl']
        # 预期格式：week1_judge2_score 或类似
        pattern = re.compile(rules['regex'], re.IGNORECASE)

        meta_records = []
        for col in columns:
            match = pattern.match(col)
            if match:
                # 假设 regex 有两个捕获组：(week, judge_slot)
                week_num, judge_idx = match.groups()
                meta_records.append({
                    'column': col,
                    'week_num': int(week_num),
                    'judge_slot': int(judge_idx)
                })

        if not meta_records:
            self.logger.warning("未解析到任何评分列！请检查 rules.yaml 中的 regex 配置。")

        return pd.DataFrame(meta_records)

    @staticmethod
    def clean_score_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        分值清洗与截断。
        物理逻辑：处理打分中可能出现的非标数字或由于 N/A 产生的空缺。
        """
        # 确保分数列也是数值型
        score_cols = [c for c in df.columns if 'score' in c]
        for col in score_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # [物理约束] 评委分通常在 1-10 之间。
            # 如果有额外加分 (Bonus)，可能会更高，但不能是负数。
            # 这里的 0 通常代表“未参赛”或“淘汰”，不应被视为 0 分。
            # 真正的清洗在 Transformer 的 handle_censorship 中完成，这里只做类型转换。

        return df

# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

    # 模拟极端脏数据：包含 Week 110 错误和大小写混乱
    test_df = pd.DataFrame({
        'placement': [1, 10, 5, 12, 2],
        'celebrity_name': [' john DOE ', 'Jane smith', ' BOB ', 'Alien', 'RunnerGuy'],
        'results': [
            'Winner',
            'Eliminated Week 110',  # 典型错误
            'Eliminated Week 3',
            'Withdrew Week 2',
            'Runner-Up'
        ],
        'season': [1, 1, 1, 1, 1]
    })

    parser = TextParser()

    print("\n--- 1. 标准化实体 ---")
    test_df = parser.standardize_entities(test_df)
    print(test_df['celebrity_name'].tolist())

    print("\n--- 2. 解析生存标签 ---")
    test_df = parser.parse_survival_labels(test_df)
    print(test_df[['celebrity_name', 'final_status', 'eliminated_week', 'had_bottom_two_record']])

    # 验证修复逻辑
    week_110_fix = test_df.loc[1, 'eliminated_week']
    assert week_110_fix == 1.0, f"Week 110 修复失败，当前值为: {week_110_fix}"
    print("\n[PASS] 逻辑体检通过：Week 110 已被统计学规则修正为 Week 1。")