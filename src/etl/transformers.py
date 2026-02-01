# ==============================================================================
# src/etl/transformers.py
# Role: Signal Refinery & Feature Transformer (v5.6 - Robust Edition)
# Function: Transforming raw wide-format data into aggregated fundamental tensors.
# Fix: Resolved KeyError during merge via explicit schema enforcement & ID alignment.
# Standard: Industrial Reliability / Zero-Copy Potential / Vectorized Aggregation.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from scipy import stats
from src.etl.config_loader import ConfigLoader
from src.utils.logger import setup_logger


class DataTransformer:
    """
    数据变换引擎：
    负责执行从“评委观测维度”向“选手技术本位”的降维映射。

    [学术逻辑]:
    1. 消除评委个体偏见 (Judge Specific Bias)。
    2. 消除跨赛季的分数通胀 (Grade Inflation)。
    3. 构建生存屏障，确保推断基于“存活”样本。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = setup_logger("TRANSFORMER")
        # 加载 ETL 专用配置块
        self.etl_cfg = self.cfg.get_etl_config()
        # 0.7413 的倒数约等于 1.3489，用于将 IQR 映射为标准差
        self.scaling_factor = float(self.etl_cfg.get('robust_scaling_factor', 1.3489))

    def _precompute_judge_map(self, active_seasons: list, max_weeks: int = 20) -> pd.DataFrame:
        """
        【工业级防御】预计算评委映射表。
        显式预定义 Schema，防止因空映射导致的 Merge KeyError ('season')。
        """
        columns = ['season', 'week_num', 'judge_slot', 'judge_id']
        map_records = []

        for season in active_seasons:
            for week in range(1, max_weeks + 1):
                for slot_idx in range(4):
                    # 调用 ConfigLoader 的逻辑寻址接口
                    j_id = self.cfg.get_judge_id(int(season), int(week), slot_idx)

                    # [关键修复]: 匹配 ConfigLoader 可能返回的所有 "UNKNOWN" 变体
                    if j_id and "UNKNOWN" not in j_id:
                        map_records.append({
                            'season': season,
                            'week_num': week,
                            'judge_slot': slot_idx + 1,
                            'judge_id': j_id
                        })

        # 即使一条匹配记录都没有，也要确保返回带有列名的空 DataFrame
        if not map_records:
            return pd.DataFrame(columns=columns)

        df_map = pd.DataFrame(map_records)

        # 强制类型对齐：确保 merge 键的类型与主表一致 (Int64 是 pandas 的可空整型)
        for col in ['season', 'week_num', 'judge_slot']:
            df_map[col] = df_map[col].astype('Int64')

        return df_map

    def wide_to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """物理变换：将宽表打散为标准观测流。"""
        self.logger.info("执行 Wide-to-Long 维度变换与向量化评委匹配...")

        pattern = self.etl_cfg.get('regex', r"week(\d+)_judge(\d+)_score")

        # 1. 自动识别评分列 (例如: week1_judge2_score)
        score_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
        id_vars = [c for c in df.columns if c not in score_cols]

        # 2. 物理融解 (Melting)
        melted = df.melt(
            id_vars=id_vars,
            value_vars=score_cols,
            var_name='score_meta',
            value_name='raw_score'
        )

        # 3. 解析 Regex 元数据
        extracted = melted['score_meta'].str.extract(pattern)
        melted['week_num'] = extracted[0].astype(float).astype('Int64')
        melted['judge_slot'] = extracted[1].astype(float).astype('Int64')

        # 4. 向量化挂载评委身份 (解决跨赛季对比的裁判一致性问题)
        unique_seasons = melted['season'].dropna().unique().tolist()
        judge_map_df = self._precompute_judge_map(unique_seasons)

        # 物理合并: 这一步现在是安全的
        melted = melted.merge(
            judge_map_df,
            on=['season', 'week_num', 'judge_slot'],
            how='left'
        )

        # 补全缺失 ID
        melted['judge_id'] = melted['judge_id'].fillna('UNKNOWN_JUDGE')

        return melted.drop(columns=['score_meta'])

    def apply_survival_barrier(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【学术严谨性】生存屏障构建 (Censorship Guard)。
        物理意义：过滤掉淘汰后的“幽灵得分”和未参赛周的 N/A，防止 0 值污染分布。
        """
        initial_len = len(df)

        # 决赛选手或 Winner 标记为存活至无穷远 (999周)
        elim_week = df['eliminated_week'].fillna(999)

        # 准则：当前周必须在存活期内，且分数必须有效 (>0)
        is_active = (df['week_num'] <= elim_week)
        has_score = (df['raw_score'].notna()) & (df['raw_score'] > 0)

        clean_df = df[is_active & has_score].copy()

        dropped = initial_len - len(clean_df)
        if dropped > 0:
            self.logger.info(f" [Ghost Defense] 拦截了 {dropped} 条淘汰后的幽灵记录或无效观测。")

        return clean_df

    def apply_robust_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【数学亮点】Episode-level Robust Z-Score 去通胀。
        公式：$Z = (Score - Median) / (IQR / 1.3489)$
        """
        self.logger.info("执行单场次 (Per-Episode) 鲁棒标准化...")

        def _robust_scaler(x):
            # 样本太少无法计算统计量 (如决赛周只有2人)
            if len(x) < 2: return np.zeros_like(x, dtype=float)

            vals = x.values
            # 使用 nanpercentile 对抗极端打分波动
            q25, q50, q75 = np.nanpercentile(vals, [25, 50, 75])
            iqr = q75 - q25

            # [奇异性防御]: 如果 IQR 极小 (和稀泥)，则该场次 Z-Score 设为 0
            if iqr < 1e-6:
                return vals - q50

            return (vals - q50) / (iqr / self.scaling_factor)

        # 关键点：groupby 确保标准化的基准是“当场比赛的竞争对手”
        # 这在论文中是“消除评委评分通胀”的核心依据
        df['score_z'] = df.groupby(['season', 'week_num'])['raw_score'].transform(_robust_scaler)
        return df.fillna({'score_z': 0})

    def detect_structural_break(self, df: pd.DataFrame):
        """
        制度断裂审计：验证 S28 规则变更对打分生态的影响。
        """
        trans_s = self.cfg.load_config().get('mechanisms', {}).get('transition_season', 28)

        test_data = df.dropna(subset=['raw_score', 'season'])
        pre_28 = test_data[test_data['season'] < trans_s]['raw_score']
        post_28 = test_data[test_data['season'] >= trans_s]['raw_score']

        self.logger.info(f"--- 启动 S{trans_s} 制度断裂统计取证 ---")
        if len(pre_28) > 50 and len(post_28) > 50:
            # 1. 均值偏移检验 (Welch's T-Test)
            t_stat, p_t = stats.ttest_ind(pre_28, post_28, equal_var=False)
            # 2. 分布差异检验 (KS-Test)
            ks_stat, p_ks = stats.ks_2samp(pre_28, post_28)

            self.logger.info(f" [Forensics] Mean Shift (p-val): {p_t:.2e}")
            self.logger.info(f" [Forensics] Dist Shift (p-val): {p_ks:.2e}")

            if p_ks < 0.05:
                self.logger.info(" ✅ 结论：规则变更引发了极显著的打分模态断裂。")

    def generate_inference_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        【关键步骤】多维信号聚合。
        将评委维度的“长表”压缩为选手-周级的“聚合因子表”，彻底解决 KeyError。
        """
        self.logger.info("执行周度信号全量聚合 (Aggregation)...")

        # 1. 计算核心聚合统计量
        agg = df.groupby(['season', 'week_num', 'celebrity_name']).agg(
            week_avg_score=('raw_score', 'mean'),  # 基础技术分
            week_z_sum=('score_z', 'sum'),  # 鲁棒强度累计
            judge_count=('judge_id', 'count'),  # 信号源深度
            signal_clarity=('raw_score', 'std')  # 评委共识度
        ).reset_index()

        agg['signal_clarity'] = agg['signal_clarity'].fillna(0.0)

        # 2. 计算周度技术排名 (Tech Rank)
        agg['tech_rank'] = agg.groupby(['season', 'week_num'])['week_avg_score'].rank(
            ascending=False, method='min'
        )

        # 3. 补全静态元数据 (从原始观测中恢复特征)
        static_cols = [
            'celebrity_industry', 'celebrity_age_during_season', 'final_status',
            'eliminated_week', 'placement', 'ballroom_partner', 'had_bottom_two_record'
        ]
        valid_static_cols = [c for c in static_cols if c in df.columns]

        # 取每个选手在赛季中的第一次观测作为静态特征基准
        df_static = df.drop_duplicates(subset=['season', 'celebrity_name'])[
            valid_static_cols + ['season', 'celebrity_name']
            ]

        return agg.merge(df_static, on=['season', 'celebrity_name'], how='left')


# --- 流水线集成函数 ---
def run_transform_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 1 核心变换流水线：
    宽表 -> 长表 -> 生存过滤 -> 标准化 -> 聚合 -> 黄金因子库。
    """
    transformer = DataTransformer()

    # A. 物理形态转换 (含评委 ID 挂载)
    df_long = transformer.wide_to_long(df)

    # B. 清理淘汰后的幽灵记录
    df_clean = transformer.apply_survival_barrier(df_long)

    # C. 执行 Episode-level 标准化 (消除通胀)
    df_norm = transformer.apply_robust_normalization(df_clean)

    # D. 统计取证 (制度断裂)
    transformer.detect_structural_break(df_norm)

    # E. 最终聚合 (解决下游 KeyError: 'week_avg_score' 的命门)
    df_gold_candidate = transformer.generate_inference_aggregates(df_norm)

    return df_gold_candidate


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Transformer module logic updated. Ready for high-precision inference.")