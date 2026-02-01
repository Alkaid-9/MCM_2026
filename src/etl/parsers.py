# ==============================================================================
# src/etl/parsers.py
# Role: Non-structured Data Surgeon (v5.5 - O-Prize Standard)
# Function: Vectorized String Parsing, Anomaly Correction, and Censorship Labeling.
# Physics: Transforming textual evidence into mathematical constraints for MCMC.
# Standard: Industrial Robustness / Academic Data Integrity / Survival Guard.
# ==============================================================================

import pandas as pd
import numpy as np
import re
import logging
from typing import List, Optional, Dict, Any
from src.etl.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class TextParser:
    """
    文本解析引擎：执行数据“去幻觉”与标准化。
    [学术背景]:
    本模块不只是清洗字符串，它在构建贝叶斯推理的“证据空间”。
    特别是对被淘汰周次的识别，直接决定了似然函数 P(Outcome | Votes) 的时间切片。
    """

    def __init__(self):
        self.cfg_loader = ConfigLoader()
        self.logger = setup_logger("TEXT_PARSER")

    def standardize_entities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗实体名称（人名、行业、州名）。
        学术价值：消除由于冗余空格或大小写引起的“虚假维度增殖” (Dimensionality Bloat)。
        """
        self.logger.info("执行实体标准化 (Entity Resolution)...")

        # 识别文本列 (排除 results 这种需要特殊处理的列)
        obj_cols = df.select_dtypes(include=['string', 'object']).columns
        cols_to_fix = [c for c in obj_cols if c != 'results']

        for col in cols_to_fix:
            # 1. 强制转为字符串并清洗
            # 逻辑：去首尾空格 -> 压缩中间多重空格 -> 统一转为 Title Case
            df[col] = df[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True).str.title()

            # 2. 处理空值占位符
            # 物理直觉：'None' 或 'N/A' 在数学模型中应当作为真正的缺失值处理
            df[col] = df[col].replace(['N/A', 'N/a', 'Null', 'None', 'Nan', 'nan'], np.nan)

        return df

    def parse_survival_results(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        解析 'results' 列，提取生存分析标签。
        物理意义：识别右删失 (Right-Censored) 样本。
        获胜者（Winner）和退赛者（Withdrew）在统计上属于“生存时间未完全观测”。
        """
        self.logger.info("解析结果列并构建生存屏障 (Survival Barrier)...")

        # --- 1. 提取淘汰周次 (Censorship Time) ---
        # 利用正则捕捉 "Week X"
        # 兼容 "Eliminated Week 5" 和 "Week 5" 等多种写法
        raw_weeks = df['results'].str.extract(r'Week\s*(\d+)', flags=re.IGNORECASE)[0].astype(float)

        # --- 2. 启发式异常修复 (Heuristic Repair for 'Week 110') ---
        # 逻辑：DWTS 一季通常 6-12 周。如果出现 'Week 110'，结合 placement 判断。
        # 若 placement 极低（如 10+），大概率是 Week 1；若高，可能是 Week 10。

        # 标记异常点 (大于 20 周通常是不可能的)
        anomaly_mask = raw_weeks > 20
        if anomaly_mask.any():
            bad_indices = df.index[anomaly_mask].tolist()
            self.logger.warning(f"检测到 {len(bad_indices)} 处周次异常 (e.g., Week 110)，执行上下文修复...")

            # 如果存在 placement 列，利用排名辅助判断
            if 'placement' in df.columns:
                # 规则：排名 > 8 (倒数)，修正为 Week 1；否则修正为 Week 10
                # fillna(1.0) 是为了防止 placement 缺失时的保守策略
                placements = df.loc[anomaly_mask, 'placement'].fillna(99)
                corrected_weeks = np.where(placements >= 8, 1.0, 10.0)

                # 回填修正值
                # 注意：这里需要对齐索引
                df.loc[anomaly_mask, 'eliminated_week'] = corrected_weeks
                self.logger.info(f" -> 已根据最终排名修正异常周次。")
            else:
                # 如果没有 placement，保守修正为 Week 1 (假设录入错误多发生在早期)
                df.loc[anomaly_mask, 'eliminated_week'] = 1.0
        else:
            # 无异常直接赋值
            df['eliminated_week'] = raw_weeks

        # --- 3. 最终状态向量化映射 (Status Mapping) ---
        # 严格区分：Event (1, 确定的失败) vs Censored (0, 存活/退出)
        # 使用 np.select 替代慢速 apply
        conds = [
            df['results'].str.contains(r'1st|Winner|Champion', case=False, na=False),
            df['results'].str.contains(r'2nd|Runner', case=False, na=False),
            df['results'].str.contains(r'3rd|Finalist', case=False, na=False),
            df['results'].str.contains(r'Withdrew|Quit|Injured', case=False, na=False),
            df['results'].str.contains(r'Eliminated', case=False, na=False)
        ]
        choices = ['Winner', 'RunnerUp', 'Finalist', 'Withdrew', 'Eliminated']
        df['final_status'] = np.select(conds, choices, default='Active')

        # --- 4. 决赛选手时间对齐 ---
        # 赢家和亚军的淘汰周次设为全季最大观测周 (Max Season Length)
        # 这在 Survival Analysis 中代表这些观测点直到最后都是“存活”的
        season_max_weeks = df.groupby('season')['eliminated_week'].transform('max')
        is_censored = df['final_status'].isin(['Winner', 'RunnerUp', 'Finalist'])

        # 仅填充空值，避免覆盖已有的逻辑
        df.loc[is_censored, 'eliminated_week'] = df.loc[is_censored, 'eliminated_week'].fillna(season_max_weeks)

        # --- 5. 危险区信号探测 (Jeopardy Mining) ---
        # 物理直觉：这是贝叶斯反演的“黄金约束”。
        # 如果 row 包含 "Bottom 2"，模型在第 X 周的似然计算会强制该选手的 v_i + j_i 处于末尾。
        # 正则匹配：Bottom Two, Bottom 2, Risk, Danger
        jeopardy_pattern = r'Bottom\s*(?:Two|2|Three|3)|Risk|Danger|Jeopardy'
        df['had_bottom_two_record'] = df['results'].str.contains(jeopardy_pattern, case=False, na=False).astype(int)

        return df

    def get_score_columns_metadata(self, columns: List[str]) -> pd.DataFrame:
        """
        【动态元数据映射】
        利用正则表达式解析复杂的宽表列名，建立物理索引映射。
        物理意义：确定每一列评分对应的是哪一周、哪位评委。
        """
        etl_cfg = self.cfg_loader.get_etl_config()
        regex_pattern = etl_cfg.get('regex', r"week(\d+)_judge(\d+)_score")
        pattern = re.compile(regex_pattern, re.IGNORECASE)

        meta_records = []
        for col in columns:
            match = pattern.match(col)
            if match:
                # 假设 regex 有两个捕获组：(week, judge_slot)
                week_num, judge_idx = match.groups()
                meta_records.append({
                    'column_name': col,
                    'week_num': int(week_num),
                    'judge_slot': int(judge_idx)
                })

        if not meta_records:
            self.logger.critical("致命错误：无法从列名中解析评分元数据。请检查 rules.yaml 中的 regex。")
            return pd.DataFrame()

        return pd.DataFrame(meta_records)

    @staticmethod
    def clean_numerical_garbage(series: pd.Series) -> pd.Series:
        """
        清洗数值列中的“幽灵字符”。
        例如处理类似 '[7.8E0]' 这种被错误识别为文本的科学计数法或带括号的数字。
        """

        def _force_float(x):
            if pd.isna(x): return np.nan
            try:
                # 尝试直接转换
                return float(x)
            except ValueError:
                try:
                    # 尝试剥离非数字字符 (保留 ., -, e/E)
                    clean_val = re.sub(r'[^\d\.eE\-]', '', str(x))
                    return float(clean_val)
                except:
                    return np.nan

        return series.apply(_force_float)


# --- 单元测试 (Unit Test) ---
if __name__ == "__main__":
    # 配置基础日志用于独立测试
    logging.basicConfig(level=logging.INFO)
    parser = TextParser()

    # 构造包含争议案例和脏数据的模拟集
    mock_data = pd.DataFrame({
        'celebrity_name': [' JERRY RICE ', 'Bobby Bones', 'Trista Sutter'],
        'results': ['Winner (Season 2)', 'Winner (S27)', 'Eliminated Week 110'],  # 模拟 OCR 错误
        'placement': [2, 1, 12],
        'season': [2, 27, 1]
    })

    print("\n>>> 实体标准化结果:")
    mock_data = parser.standardize_entities(mock_data)
    print(mock_data['celebrity_name'].tolist())

    print("\n>>> 生存标签解析结果 (含右删失逻辑):")
    mock_data = parser.parse_survival_results(mock_data)
    print(mock_data[['celebrity_name', 'final_status', 'eliminated_week', 'had_bottom_two_record']])

    # 验证 Week 110 是否被修正
    # Trista Sutter 排名 12 (倒数)，应被修正为 Week 1
    fixed_week = mock_data.loc[2, 'eliminated_week']
    print(f"\n[Validation] 'Week 110' repaired to: {fixed_week}")

    if fixed_week == 1.0:
        print(" [PASS] 启发式修正测试通过。")
    else:
        print(" [FAIL] 异常值修复失败。")