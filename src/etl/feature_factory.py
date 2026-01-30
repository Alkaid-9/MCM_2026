# ==============================================================================
# src/etl/feature_factory.py
# Role: Factor Library Architect (The Alpha Generator)
# Function: Creating high-dimensional features with robust index alignment
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from src.etl.config_loader import ConfigLoader


class FeatureFactory:
    """
    因子工厂：负责构建静态特征（行业、年龄）与动态特征（动量、环境因子）。
    采用 .transform() 机制确保计算结果与原 DataFrame 索引完美对齐。
    """

    @staticmethod
    def build_celebrity_factors(df: pd.DataFrame) -> pd.DataFrame:
        """
        构建明星静态特征的 Dummy 变量。
        """
        logging.info("正在构建 Celebrity 静态因子 (Dummies)...")

        # 针对 Task 3：分析行业背景（运动员、歌手等）的潜在票数溢价
        # 使用 pd.get_dummies 转换类别变量
        # 强制转换为 int (0/1) 以确保后续算法层（如 XGBoost）的数值兼容性
        cols_to_dummy = ['industry_group', 'age_group']
        df = pd.get_dummies(df, columns=cols_to_dummy, prefix=['ind', 'age'], dtype=int)

        return df

    @staticmethod
    def build_performance_dynamics(df: pd.DataFrame) -> pd.DataFrame:
        """
        构建表现动力学因子：进步动量与累计声望。
        物理意义：捕捉观众对‘黑马成长型’选手的心理偏好。
        """
        logging.info("正在构建动态表现因子 (Momentum Logic)...")

        # 预排序是关键，确保 diff() 和 expanding() 的物理时间顺序正确
        # 但注意：transform 会自动将结果映射回原始索引，无惧排序
        df_sorted = df.sort_values(['celebrity_name', 'season', 'week_num'])

        # 1. 进步动量 (Week-over-Week improvement)
        # 计算当前周均分相对于上一周的增量
        df['score_delta'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(lambda x: x.diff())

        # 2. 累计表现 (Expanding Mean)
        # 反映选手的历史口碑积累，排除单场失误的噪音
        df['cum_score_avg'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(
            lambda x: x.expanding().mean())

        return df

    @staticmethod
    def build_contextual_factors(df: pd.DataFrame) -> pd.DataFrame:
        """
        构建赛场环境变量：竞争强度与技术排名。
        """
        logging.info("正在构建竞争环境因子...")

        # 1. 竞争烈度：计算当季当周还有多少选手存活
        # 物理意义：竞争人数越少，边际得票难度越大
        df['n_competitors'] = df.groupby(['season', 'week_num'])['celebrity_name'].transform('nunique')

        # 2. 相对技术位置 (Relative Percentile Rank)
        # 物理意义：选手在当周技术分榜单中的百分比排名
        df['relative_technical_rank'] = df.groupby(['season', 'week_num'])['week_avg_score'].transform(
            lambda x: x.rank(pct=True))

        return df

    @classmethod
    def generate_gold_library(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        一键执行所有因子构建逻辑，产出 Gold 层因子库。
        """
        # 注意逻辑顺序：先算动态因子（依赖原始分），再转 Dummy（会改变列结构）
        df = cls.build_performance_dynamics(df)
        df = cls.build_contextual_factors(df)
        df = cls.build_celebrity_factors(df)

        # 填补计算产生的初始 NaN (例如第一周没有 delta)
        # 填充为 0 符合物理意义：第一周没有“进步幅度”
        df = df.fillna(0)

        logging.info(f"黄金因子库构建完成。特征维度: {df.shape}")
        return df


# ------------------------------------------------------------------------------
# 高阶因子：舞伴历史光环 (Cross-Season Alpha)
# ------------------------------------------------------------------------------
def calculate_historical_partner_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算舞伴历史胜率因子。
    物理意义：量化‘名舞伴’对明星选手的生存保障作用（Partner Alpha）。
    """
    logging.info("计算舞伴历史溢价因子 (Partner Alpha)...")

    # 1. 舞伴基本面：历史上带队的所有打分均值
    df['partner_alpha'] = df.groupby('ballroom_partner')['raw_score'].transform('mean')

    # 2. 舞伴稳定性：历史上带队的平均排名 (Placement)
    # 使用 transform 确保跨行计算后直接对齐到每一观测点
    df['partner_hist_avg_placement'] = df.groupby('ballroom_partner')['placement'].transform('mean')

    return df