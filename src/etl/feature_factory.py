"""
MCM 2026 Problem C: Strategic Factor Library (The Alpha Generator)
Role: Constructing high-dimensional features for causal attribution and Bayesian priors.
"""

import pandas as pd
import numpy as np
import logging
from src.etl.config_loader import ConfigLoader


class FeatureFactory:
    """
    因子工厂：负责构建静态特征（行业、年龄）与动态表现因子。
    设计准则：
    1. 索引对齐：强制使用 .transform() 确保因子直接合并回原始面板数据。
    2. 信号分离：分离‘技术实力’(Beta)与‘身份红利’(Alpha)。
    3. 鲁棒性：处理冷启动问题（如第一周无动量数据）。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = logging.getLogger("FEATURE_FACTORY")

    def build_celebrity_static_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建明星静态背景因子。
        物理意义：捕捉不同身份背景自带的‘流量池’差异。
        """
        self.logger.info("正在提取明星背景特征 (Industry & Age)...")

        # 1. 行业映射 (从 rules.yaml 获取语义映射)
        mapping = self.cfg._config['features']['industry_mapping']
        df['industry_group'] = df['celebrity_industry'].map(mapping).fillna('Baseline')

        # 2. 年龄分段 (代际偏好分析)
        age_cfg = self.cfg._config['features']['age_segmentation']
        df['age_group'] = pd.cut(
            df['celebrity_age_during_season'],
            bins=age_cfg['bins'],
            labels=age_cfg['labels']
        )

        # 3. 转换为 Dummy 变量 (为 Task 3 的线性模型和 XGBoost 做准备)
        # 注意：这里我们保留原列，同时生成辅助 Dummy
        df = pd.get_dummies(df, columns=['industry_group', 'age_group'], prefix=['ind', 'age'], drop_first=False)

        return df

    def build_performance_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建表现动力学因子。
        物理意义：观众不仅看分数高低，更看重‘成长性’(Underdog Story)。
        """
        self.logger.info("正在计算表现动量因子 (Momentum Logic)...")

        # 确保时序正确
        df = df.sort_values(['celebrity_name', 'season', 'week_num'])

        # 1. 技术进步动量 (Score Momentum)
        # 计算当周均分与上一周的差值
        df['score_delta'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(lambda x: x.diff())

        # 2. 累计声望 (Cumulative Reputation)
        # 反映观众对该选手形成的长线技术认知，消除单周失误噪音
        df['cum_avg_score'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(
            lambda x: x.expanding().mean()
        )

        # 3. 稳定性因子 (Volatility)
        # 评委分波动大的选手更具话题性
        df['score_volatility'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(
            lambda x: x.expanding().std()
        ).fillna(0)

        return df

    def build_contextual_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建赛场环境变量。
        物理意义：同样的 8 分，在高手如云的周次和菜鸡互啄的周次，吸粉能力完全不同。
        """
        self.logger.info("正在计算竞争环境因子...")

        # 1. 相对排名位置 (Relative Technical Rank)
        # 当周技术分在所有存活选手中的百分比排名
        df['relative_rank'] = df.groupby(['season', 'week_num'])['week_avg_score'].transform(
            lambda x: x.rank(pct=True)
        )

        # 2. 存活压力因子 (Survival Pressure)
        # 剩余选手越少，边际竞争越激烈
        df['n_competitors'] = df.groupby(['season', 'week_num'])['celebrity_name'].transform('nunique')

        return df

    def build_partner_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建舞伴历史红利因子 (Pro-Partner Alpha)。
        学术价值：顶刊极其看重这种“名师出高徒”的内生性问题。
        """
        self.logger.info("正在量化舞伴历史红利 (Partner Alpha)...")

        # 计算专业舞伴在所有赛季带队的历史中位排名 (Placement)
        # 物理意义：衡量这个舞伴是否有‘化腐朽为神奇’的能力
        partner_stats = df.groupby('ballroom_partner')['placement'].transform('mean')

        # 归一化：排名越小 Alpha 越高
        df['partner_alpha'] = 1.0 / (partner_stats + 1e-9)

        return df

    def generate_gold_library(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        一键合成黄金因子库。
        """
        df = self.build_celebrity_static_factors(df)
        df = self.build_performance_dynamics(df)
        df = self.build_contextual_factors(df)
        df = self.build_partner_alpha(df)

        # 最终填充 NaN (如第一周无 delta)
        df = df.fillna(0)

        self.logger.info(f"因子库构建成功。产出特征维度: {df.shape}")
        return df


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # 模拟输入 (假设已经跑完了 ETL 的前几步)
    mock_df = pd.DataFrame({
        'celebrity_name': ['A', 'A', 'B', 'B'],
        'season': [1, 1, 1, 1],
        'week_num': [1, 2, 1, 2],
        'week_avg_score': [7.0, 8.5, 9.0, 8.0],
        'celebrity_industry': ['Singer', 'Singer', 'NFL Player', 'NFL Player'],
        'celebrity_age_during_season': [25, 25, 45, 45],
        'ballroom_partner': ['Derek', 'Derek', 'Mark', 'Mark'],
        'placement': [1, 1, 2, 2]
    })

    factory = FeatureFactory()
    gold_df = factory.generate_gold_library(mock_df)
    print(gold_df[['celebrity_name', 'score_delta', 'partner_alpha', 'ind_Music_Industry']])