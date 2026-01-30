# ==============================================================================
# src/etl/transformers.py
# Role: Signal Refinery & Feature Engineering (Industrial/Academic Hybrid)
# Function: Long-format reshaping, Normalization, Structural Break Testing, Aggregation
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from scipy import stats
from src.etl.config_loader import ConfigLoader


class DataTransformer:
    """
    数据变换引擎：将清洗后的数据转化为具有统计学意义的‘黄金因子层’。
    """

    @classmethod
    def wide_to_long(cls, df: pd.DataFrame) -> pd.DataFrame:
        """将宽表转换为面板数据(Panel Data)。"""
        logging.info("执行 Wide-to-Long 变换 (Melt)...")
        cfg = ConfigLoader.get_etl_rules()
        id_vars = cfg.get('id_columns')
        score_pattern = cfg.get('score_column_regex')

        # 识别评分列
        score_cols = [c for c in df.columns if 'judge' in c and 'score' in c]

        melted = df.melt(
            id_vars=id_vars,
            value_vars=score_cols,
            var_name='score_meta',
            value_name='raw_score'
        )

        # 使用正则解析周数和评委ID
        extracted = melted['score_meta'].str.extract(score_pattern)
        melted['week_num'] = extracted[0].astype(int)
        melted['judge_id'] = extracted[1].astype(int)

        return melted.drop(columns=['score_meta'])

    @classmethod
    def handle_censorship(cls, df: pd.DataFrame) -> pd.DataFrame:
        """物理剔除：过滤掉选手被淘汰后的‘幽灵’行(0分或NaN)。"""
        # 仅保留评委实际给分的大于0的行
        clean_df = df.dropna(subset=['raw_score']).copy()
        clean_df = clean_df[clean_df['raw_score'] > 0]
        return clean_df

    @classmethod
    def apply_robust_normalization(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        【数学亮点】Robust Z-Score 去通胀。
        公式: (x - median) / IQR
        """
        logging.info("按赛季执行 Robust Z-Score 标准化...")

        def _season_scaler(x):
            if len(x) < 2: return x - x
            q25, q50, q75 = x.quantile([0.25, 0.5, 0.75])
            iqr = q75 - q25
            # 防止 IQR 为 0 导致除零错误
            if iqr < 1e-6:
                return x - q50
            return (x - q50) / iqr

        df['score_z'] = df.groupby('season')['raw_score'].transform(_season_scaler)
        return df

    @classmethod
    def map_categorical_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """【特征工程】执行行业映射与年龄分段。"""
        logging.info("执行特征映射与行业聚合...")
        cfg = ConfigLoader.load_config()

        mapping = cfg['features']['industry_mapping']
        df['industry_group'] = df['celebrity_industry'].map(mapping).fillna('Other')

        bins = cfg['features']['age_bins']
        labels = cfg['features']['age_labels']
        df['age_group'] = pd.cut(df['celebrity_age_during_season'], bins=bins, labels=labels)

        return df

    @classmethod
    def calculate_alpha_factors(cls, df: pd.DataFrame) -> pd.DataFrame:
        """【Quant 思维】计算舞伴历史表现因子。"""
        logging.info("计算舞伴历史溢价因子 (Partner Alpha)...")
        # 修正：按舞伴计算历史平均得分，作为其能力基准
        partner_stats = df.groupby('ballroom_partner')['raw_score'].mean().rename('partner_alpha')
        df = df.merge(partner_stats, on='ballroom_partner', how='left')
        return df

    @classmethod
    def detect_structural_break(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        【O奖必备】结构性断裂检测。
        用 T-检验 验证 Season 28 前后的评分分布差异。
        """
        logging.info("执行结构性断裂检测 (Season 28 Check)...")
        pre_28 = df[df['season'].between(3, 27)]['raw_score']
        post_28 = df[df['season'] >= 28]['raw_score']

        if len(pre_28) > 30 and len(post_28) > 30:
            t_stat, p_val = stats.ttest_ind(pre_28, post_28, equal_var=False)
            logging.info(f"T-test 结果: t_stat={t_stat:.4f}, p_value={p_val:.4e}")
            if p_val < 0.05:
                logging.info(">>> 结论：Season 28 评分尺度存在显著结构性偏移。")
            else:
                logging.info(">>> 结论：Season 28 评分尺度无显著统计差异。")
        return df

    @classmethod
    def generate_aggregates(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        【反演引擎必需】计算选手当周的技术聚合分。
        """
        logging.info("正在生成周度聚合分 (Technical Totals)...")

        # 核心：计算每位选手每周获得的评委总分、均分和评委人数
        agg = df.groupby(['season', 'week_num', 'celebrity_name'])['raw_score'].agg(
            ['mean', 'sum', 'count']).reset_index()
        agg.columns = ['season', 'week_num', 'celebrity_name', 'week_avg_score', 'week_total_score', 'judge_count']

        # 合并回主表
        df = df.merge(agg, on=['season', 'week_num', 'celebrity_name'], how='left')
        return df


# ------------------------------------------------------------------------------
# 集成调用接口
# ------------------------------------------------------------------------------
def run_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """按逻辑顺序执行所有 Stage 1 变换"""
    df = DataTransformer.wide_to_long(df)
    df = DataTransformer.handle_censorship(df)
    df = DataTransformer.apply_robust_normalization(df)
    df = DataTransformer.map_categorical_features(df)
    df = DataTransformer.calculate_alpha_factors(df)
    df = DataTransformer.detect_structural_break(df)
    # 注意：generate_aggregates 通常在 pipeline 中作为独立步骤调用，
    # 或者在这里统一调用亦可。为了保持 pipeline 的可读性，我们放在 pipeline 里。
    return df