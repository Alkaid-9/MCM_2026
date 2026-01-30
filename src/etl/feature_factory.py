"""
MCM 2026 Problem C: Strategic Factor Library (The Alpha Generator)
Role: Constructing high-dimensional features for causal attribution and Bayesian priors.
Standard: Academic Rigor (Causality Firewall) & Industrial Scalability.
"""

import pandas as pd
import numpy as np
import logging
from src.etl.config_loader import ConfigLoader

class FeatureFactory:
    """
    因子工厂：负责构建静态特征与动态表现因子。

    【核心逻辑】：
    1. 因果防火墙 (Causality Firewall)：
       严禁使用未来数据。所有涉及‘能力评估’的因子（如 Partner Alpha）必须采用
       Lagged Expanding Window（滞后扩张窗口）计算。

    2. 环境信噪比 (Signal Clarity)：
       量化单场比赛的评委打分区分度。区分度低时，粉丝投票的权重被动放大。

    3. 冷启动防御 (Cold Start Defense)：
       针对新秀舞伴或第一周数据，使用贝叶斯先验填充，而非简单的 0。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = logging.getLogger("FEATURE_FACTORY")

    def build_celebrity_static_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建明星静态背景因子。
        物理意义：捕捉不同身份背景自带的‘流量池’差异 (Prior Bias)。
        """
        self.logger.info("正在提取明星背景特征 (Industry & Age)...")

        # 1. 行业映射 (从 rules.yaml 获取语义映射)
        mapping = self.cfg._config['features']['industry_mapping']
        df['industry_group'] = df['celebrity_industry'].map(mapping).fillna('Baseline')

        # 2. 年龄分段 (代际偏好分析)
        # 注意：此处需处理空值，若年龄缺失填入中位数
        age_cfg = self.cfg._config['features']['age_segmentation']
        median_age = df['celebrity_age_during_season'].median()
        df['celebrity_age_during_season'] = df['celebrity_age_during_season'].fillna(median_age)

        df['age_group'] = pd.cut(
            df['celebrity_age_during_season'],
            bins=age_cfg['bins'],
            labels=age_cfg['labels']
        )

        # 3. 转换为 Dummy 变量 (为 Task 3 的线性模型和 XGBoost 做准备)
        # 注意：保留原列用于可视化，生成 dummy 用于回归
        df = pd.get_dummies(df, columns=['industry_group', 'age_group'],
                            prefix=['ind', 'age'], drop_first=False)

        return df

    def build_performance_dynamics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建表现动力学因子。
        物理意义：观众不仅看分数高低，更看重‘成长性’ (Underdog Story)。
        """
        self.logger.info("正在计算表现动量因子 (Momentum Logic)...")

        # 必须确保时序严格正确
        df = df.sort_values(['season', 'week_num', 'celebrity_name'])

        # 1. 技术进步动量 (Score Momentum)
        # 逻辑：本周均分 - 上周均分
        # 注意：第一周会产生 NaN，后续fillna处理，不可填 0 误导模型认为其“无进步”
        df['score_delta'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(lambda x: x.diff())

        # 2. 累计声望 (Cumulative Reputation)
        # 物理意义：反映观众对该选手形成的长线技术认知
        # 使用 expanding().mean() 模拟观众记忆的累积过程
        df['cum_avg_score'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(
            lambda x: x.expanding().mean()
        )

        # 3. 表现波动率 (Volatility / Consistency)
        # 物理意义：发挥不稳的选手往往争议大，熵值高
        df['score_volatility'] = df.groupby(['celebrity_name', 'season'])['week_avg_score'].transform(
            lambda x: x.expanding().std()
        ).fillna(0)  # 第一周波动率为 0

        return df

    def build_contextual_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建赛场环境变量。
        物理意义：同样的 8 分，在高手如云的周次和菜鸡互啄的周次，吸粉能力完全不同。
        """
        self.logger.info("正在计算竞争环境因子与信噪比...")

        # 1. 相对排名位置 (Relative Technical Rank)
        # 当周技术分在所有存活选手中的百分比排名 (0.0 = 最差, 1.0 = 最好)
        df['relative_rank'] = df.groupby(['season', 'week_num'])['week_avg_score'].transform(
            lambda x: x.rank(pct=True, ascending=True)
        )

        # 2. 存活压力因子 (Survival Pressure)
        # 剩余选手越少，边际竞争越激烈
        df['n_competitors'] = df.groupby(['season', 'week_num'])['celebrity_name'].transform('nunique')

        # 3. [核心新增] 单集信噪比 (Episode Signal Clarity)
        # 物理意义：计算当周所有选手得分的标准差。
        # Std Dev 小 -> 评委给分雷同 -> 技术信号弱 -> 粉丝投票权重被动放大。
        df['episode_signal_clarity'] = df.groupby(['season', 'week_num'])['week_avg_score'].transform('std').fillna(0)

        # 归一化信噪比 (方便后续加权)
        max_clarity = df['episode_signal_clarity'].max() + 1e-9
        df['signal_clarity_norm'] = df['episode_signal_clarity'] / max_clarity

        return df

    def build_partner_alpha(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        构建舞伴历史红利因子 (Pro-Partner Alpha) —— 【严禁穿越版】。

        学术修正：
        原逻辑：Partner_Alpha = Global_Mean(Placement) -> 包含未来赛季，严重违规。
        新逻辑：Partner_Alpha_T = Expanding_Mean(Placement_1_to_T-1) -> 仅使用历史数据。
        """
        self.logger.info("正在量化舞伴历史红利 (Partner Alpha) [Lagged Expanding Window]...")

        # 1. 提取舞伴的历史战绩表 (去重，每季每舞伴一行)
        # 注意：这里我们只关心最终排名 (placement)
        partner_history = df[['season', 'ballroom_partner', 'placement']].dropna().drop_duplicates()
        partner_history = partner_history.sort_values(['ballroom_partner', 'season'])

        # 2. 计算滞后扩张均值
        # shift(1) 是核心：第 N 季的 Alpha 只能由前 N-1 季决定
        partner_history['hist_avg_placement'] = partner_history.groupby('ballroom_partner')['placement'].apply(
            lambda x: x.expanding().mean().shift(1)
        )

        # 3. 处理冷启动 (Rookies)
        # 对于第一季参赛的舞伴，或者是第1季本身，hist_avg_placement 为 NaN。
        # 使用所有舞伴的全局中位排名作为先验 (Prior)。通常中位排名约为参赛人数的一半。
        global_median_rank = df['placement'].median()
        partner_history['hist_avg_placement'] = partner_history['hist_avg_placement'].fillna(global_median_rank)

        # 4. 计算 Alpha 值
        # 物理定义：排名数字越小越好。Alpha = Global_Median / Hist_Avg
        # Alpha > 1.0 : 金牌舞伴 (如 Derek Hough)
        # Alpha < 1.0 : 弱势舞伴
        partner_history['partner_alpha'] = global_median_rank / (partner_history['hist_avg_placement'] + 1e-9)

        # 5. 将因子 Merge 回主表
        # 注意：使用左连接，因为主表可能包含一些未完成的赛季数据
        df = df.merge(
            partner_history[['season', 'ballroom_partner', 'partner_alpha']],
            on=['season', 'ballroom_partner'],
            how='left'
        )

        # 6. 最后的保险：如果 Merge 后仍有 NaN (极少数情况)，填 1.0 (中性)
        df['partner_alpha'] = df['partner_alpha'].fillna(1.0)

        return df

    def generate_gold_library(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        一键合成黄金因子库。
        """
        # 1. 静态特征
        df = self.build_celebrity_static_factors(df)

        # 2. 动态特征
        df = self.build_performance_dynamics(df)

        # 3. 环境特征 (含信噪比)
        df = self.build_contextual_factors(df)

        # 4. 舞伴能力 (因果修正版)
        df = self.build_partner_alpha(df)

        # 5. 最终清洗
        # score_delta 第一周为 NaN，填 0 表示无动量
        df['score_delta'] = df['score_delta'].fillna(0)

        self.logger.info(f"因子库构建成功。产出特征维度: {df.shape}")
        return df

# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 构造模拟数据：测试穿越问题
    # 假设 Derek 在 S1 拿了第 1，在 S2 拿了第 10。
    # 在计算 S2 的 Alpha 时，应该只看到 S1 的成绩（Alpha 高）。
    # 在计算 S3 的 Alpha 时，应该看到 S1+S2 的均值（Alpha 下降）。
    mock_df = pd.DataFrame({
        'celebrity_name': ['StarA', 'StarB', 'StarC'],
        'season': [1, 2, 3],
        'week_num': [10, 10, 10],
        'ballroom_partner': ['Derek', 'Derek', 'Derek'],
        'placement': [1, 10, 5], # 历史战绩
        'celebrity_industry': ['Singer', 'Actor', 'Model'],
        'celebrity_age_during_season': [20, 30, 40],
        'week_avg_score': [9.0, 8.0, 8.5]
    })

    factory = FeatureFactory()
    gold_df = factory.generate_gold_library(mock_df)

    print("\n--- Partner Alpha Causality Check ---")
    print(gold_df[['season', 'ballroom_partner', 'placement', 'partner_alpha']])

    # 验证逻辑：
    # S1 Alpha: 应为 1.0 (冷启动)
    # S2 Alpha: 基于 S1 (Rank 1) -> Alpha 应很高 (Median/1)
    # S3 Alpha: 基于 S1, S2 (Avg 5.5) -> Alpha 应回落 (Median/5.5)