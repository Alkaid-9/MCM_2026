"""
MCM 2026 Problem C: Signal Refinery & Feature Transformer
Role: Transforming raw wide-format data into normalized gold-tier factors with signal auditing.
Standard: Academic Rigor (SNR Analytics) & Industrial Scalability.
"""

import pandas as pd
import numpy as np
import logging
from scipy import stats
from src.etl.config_loader import ConfigLoader


class DataTransformer:
    """
    数据变换引擎：执行维度转换、去通胀标准化与信号强度审计。
    设计逻辑：
    1. 信号-噪音分析 (SNR)：量化评委打分的区分度。
    2. 单集鲁棒标准化：消除评委评分尺度随时间的漂移。
    3. 制度断裂检测：统计学验证规则变更的显著性。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = logging.getLogger("TRANSFORMER")

    def wide_to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        物理变换：将 Judge1..Judge4 宽表打平。
        集成动态评委映射，确保 Task 3 归因的准确性。
        """
        self.logger.info("执行 Wide-to-Long 变换 (Melt)...")

        # 1. 自动识别评分列
        etl_cfg = self.cfg._config['etl']
        score_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
        id_vars = [c for c in df.columns if c not in score_cols]

        melted = df.melt(
            id_vars=id_vars,
            value_vars=score_cols,
            var_name='score_meta',
            value_name='raw_score'
        )

        # 2. 解析正则元数据
        pattern = etl_cfg['regex']
        extracted = melted['score_meta'].str.extract(pattern)
        melted['week_num'] = extracted[0].astype(int)
        melted['judge_slot'] = extracted[1].astype(int)

        # 3. 动态映射真实评委 ID (例如 CAI, LG)
        # 物理意义：将‘席位信号’转化为‘人格信号’
        melted['judge_id'] = melted.apply(
            lambda x: self.cfg.get_judge_id(x['season'], x['week_num'], x['judge_slot']-1),
            axis=1
        )

        return melted.drop(columns=['score_meta'])

    def handle_censorship(self, df: pd.DataFrame) -> pd.DataFrame:
        """剔除幽灵观测：仅保留有效的打分记录。"""
        clean_df = df.dropna(subset=['raw_score']).copy()
        clean_df = clean_df[clean_df['raw_score'] > 0]
        return clean_df

    def apply_robust_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【数学亮点】Robust Z-Score 去通胀。
        公式：Z = (x - Median) / IQR
        学术价值：比标准 Z-Score 更能抵抗评委的极端偏好。
        """
        self.logger.info("执行单集 (Per-Episode) 鲁棒标准化...")

        def _robust_scaler(x):
            if len(x) < 2: return np.zeros_like(x)
            q25, q50, q75 = x.quantile([0.25, 0.5, 0.75])
            iqr = q75 - q25
            # 处理全场满分或打分完全一致的病态情况
            if iqr < 1e-6:
                return x - q50
            return (x - q50) / iqr

        # 核心：按 (Season, Week) 分组，消除跨周打分漂移
        df['score_z'] = df.groupby(['season', 'week_num'])['raw_score'].transform(_robust_scaler)
        return df

    def calculate_signal_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【学术核心】量化评委信号强度 (SNR)。
        CV (变异系数) = Sigma / Mu
        逻辑：CV 越小，评委分越无区分度，观众票的‘扰动支配力’越强。
        """
        self.logger.info("正在审计评委信号强度 (Signal-to-Noise)...")

        group_cols = ['season', 'week_num']

        # 计算单周选手间的得分统计量
        stats_df = df.groupby(group_cols + ['celebrity_name'])['raw_score'].mean().reset_index()
        stats_df = stats_df.groupby(group_cols)['raw_score'].agg(
            mu_signal='mean',
            sigma_signal='std'
        ).reset_index()

        # 计算变异系数 (CV)
        stats_df['signal_cv'] = stats_df['sigma_signal'] / (stats_df['mu_signal'] + 1e-9)

        # 归一化信号强度 [0, 1]
        stats_df['signal_strength'] = stats_df['signal_cv'] / (stats_df['signal_cv'].max() + 1e-9)

        return stats_df

    def detect_structural_break(self, df: pd.DataFrame):
        """
        【O奖护城河】结构性断裂检测。
        用 KS-检验 和 T-检验 验证规则突变点。
        """
        trans_s = self.cfg._config['mechanisms']['transition_season']

        pre_data = df[df['season'] < trans_s]['raw_score']
        post_data = df[df['season'] >= trans_s]['raw_score']

        if not pre_data.empty and not post_data.empty:
            t_stat, p_val = stats.ttest_ind(pre_data, post_data, equal_var=False)
            ks_stat, ks_p = stats.ks_2samp(pre_data, post_data)

            self.logger.info(f"[Break-Test] S{trans_s} 前后均值 T-测试 p-value: {p_val:.4e}")
            self.logger.info(f"[Break-Test] S{trans_s} 前后分布 KS-测试 p-value: {ks_p:.4e}")

    def generate_inference_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        聚合周度技术基本面：为 Task 1 的反演提供观测张量。
        """
        self.logger.info("聚合周度技术基本面与信号指标...")

        # 1. 选手层面聚合
        agg = df.groupby(['season', 'week_num', 'celebrity_name']).agg(
            week_avg_score=('raw_score', 'mean'),
            week_z_sum=('score_z', 'sum'),
            judge_count=('raw_score', 'count')
        ).reset_index()

        # 2. 注入信号强度指标 (周层面)
        signal_metrics = self.calculate_signal_metrics(df)
        agg = agg.merge(signal_metrics, on=['season', 'week_num'], how='left')

        # 3. 计算技术排名 (用于 Task 1 约束)
        # method='min' 确保并列第一的情况被正确处理
        agg['tech_rank'] = agg.groupby(['season', 'week_num'])['week_avg_score'].rank(ascending=False, method='min')

        return agg


# --- 集成流水线调用接口 ---
def run_transform_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    一键运行全量转换工序。
    """
    transformer = DataTransformer()

    # A. 物理变换
    df_long = transformer.wide_to_long(df)
    df_clean = transformer.handle_censorship(df_long)

    # B. 统计对齐
    df_norm = transformer.apply_robust_normalization(df_clean)

    # C. 信号审计
    transformer.detect_structural_break(df_norm)

    # D. 聚合生成反演底表
    df_agg = transformer.generate_inference_aggregates(df_norm)

    # 合并回 Silver 层主表
    df_final = df_norm.merge(df_agg, on=['season', 'week_num', 'celebrity_name'], how='left')

    return df_final