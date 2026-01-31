# ==============================================================================
# src/etl/transformers.py
# Role: Signal Refinery & Feature Transformer
# Function: Transforming raw wide-format data into aggregated gold-tier factors.
# Standard: Academic Rigor (SNR Analytics) & Industrial Scalability.
# ==============================================================================

import pandas as pd
import numpy as np
import logging
from scipy import stats
from src.etl.config_loader import ConfigLoader

class DataTransformer:
    """
    数据变换引擎：
    负责将带有评委偏见的原始信号转化为“技术基本面”聚合张量。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = logging.getLogger("TRANSFORMER")

    def _precompute_judge_map(self, active_seasons: list, max_weeks: int = 15) -> pd.DataFrame:
        """【工业级优化】预计算评委映射表，利用 Vectorized Join 替代单行寻址。"""
        map_records = []
        for season in active_seasons:
            for week in range(1, max_weeks + 1):
                for slot_idx in range(4):
                    j_id = self.cfg.get_judge_id(season, week, slot_idx)
                    if j_id != "UNKNOWN":
                        map_records.append({
                            'season': season,
                            'week_num': week,
                            'judge_slot': slot_idx + 1,
                            'judge_id': j_id
                        })
        return pd.DataFrame(map_records)

    def wide_to_long(self, df: pd.DataFrame) -> pd.DataFrame:
        """物理变换：执行熔断机制下的 Wide-to-Long 展平。"""
        self.logger.info("执行 Wide-to-Long 变换与向量化评委匹配...")

        # 1. 获取配置 (防御性编程)
        etl_cfg = self.cfg.get_etl_config()
        pattern = etl_cfg.get('regex', r"week(\d+)_judge(\d+)_score")

        # 2. 识别评分列
        score_cols = [c for c in df.columns if 'judge' in c and 'score' in c]
        id_vars = [c for c in df.columns if c not in score_cols]

        # 3. 展平
        melted = df.melt(
            id_vars=id_vars,
            value_vars=score_cols,
            var_name='score_meta',
            value_name='raw_score'
        )

        # 4. 解析 Regex 元数据
        extracted = melted['score_meta'].str.extract(pattern)
        melted['week_num'] = extracted[0].astype(float).astype('Int64')
        melted['judge_slot'] = extracted[1].astype(float).astype('Int64')

        # 5. 向量化挂载评委 ID
        unique_seasons = melted['season'].dropna().unique().tolist()
        judge_map_df = self._precompute_judge_map(unique_seasons)

        melted = melted.merge(
            judge_map_df,
            on=['season', 'week_num', 'judge_slot'],
            how='left'
        )
        melted['judge_id'] = melted['judge_id'].fillna('UNKNOWN')

        return melted.drop(columns=['score_meta'])

    def apply_survival_barrier(self, df: pd.DataFrame) -> pd.DataFrame:
        """【学术严谨性】生存屏障构建，过滤淘汰后的幽灵数据。"""
        initial_len = len(df)
        elim_week = df['eliminated_week'].fillna(999)
        
        is_active = (df['week_num'] <= elim_week)
        # 排除 0 分和 NaN (通常代表缺席)
        has_score = (df['raw_score'].notna()) & (df['raw_score'] > 0)

        clean_df = df[is_active & has_score].copy()
        
        dropped = initial_len - len(clean_df)
        if dropped > 0:
            self.logger.info(f"生存屏障拦截了 {dropped} 条无效/幽灵数据记录。")
        return clean_df

    def apply_robust_normalization(self, df: pd.DataFrame) -> pd.DataFrame:
        """【数学亮点】Robust Z-Score 去通胀，消除跨赛季评分尺度差异。"""
        self.logger.info("执行单集 (Per-Episode) 鲁棒标准化...")

        def _robust_scaler(x):
            if len(x) < 2: return np.zeros_like(x)
            vals = x.values
            q25, q50, q75 = np.nanpercentile(vals, [25, 50, 75])
            iqr = q75 - q25
            if iqr < 1e-6: return vals - q50
            # 引入高斯缩放因子：将 IQR 转换为标准差当量
            return (vals - q50) / (iqr / 1.3489)

        # 关键点：groupby 确保了标准化的基准是“当场比赛的所有竞争者”
        df['score_z'] = df.groupby(['season', 'week_num'])['raw_score'].transform(_robust_scaler)
        df['score_z'] = df['score_z'].fillna(0)
        return df

    def detect_structural_break(self, df: pd.DataFrame):
        """制度断裂审计：统计实证 S28 规则变更的显著性 (Forensics)。"""
        trans_s = self.cfg._config.get('mechanisms', {}).get('transition_season', 28)
        
        test_data = df.dropna(subset=['raw_score', 'season'])
        pre_28 = test_data[test_data['season'] < trans_s]['raw_score']
        post_28 = test_data[test_data['season'] >= trans_s]['raw_score']

        self.logger.info(f"--- 启动 S{trans_s} 制度断裂统计取证 ---")
        if len(pre_28) < 50 or len(post_28) < 50:
            self.logger.warning("样本不足，跳过统计检验。")
            return

        t_stat, p_t = stats.ttest_ind(pre_28, post_28, equal_var=False)
        ks_stat, p_ks = stats.ks_2samp(pre_28, post_28)
        
        self.logger.info(f"Mean Shift (T-test):  p={p_t:.2e}")
        self.logger.info(f"Dist Shift (KS-test): p={p_ks:.2e}")
        
        if p_ks < 0.05:
            self.logger.info("✅ 结论：规则变更引发了显著的打分模态断裂。")

    def generate_inference_aggregates(self, df: pd.DataFrame) -> pd.DataFrame:
            """聚合周度技术表现。这是解决 KeyError 的核心：确保 week_avg_score 存在。"""
            self.logger.info("执行周度信号聚合 (Generating Aggregates)...")

            # 1. 计算核心聚合统计量
            agg = df.groupby(['season', 'week_num', 'celebrity_name']).agg(
                week_avg_score=('raw_score', 'mean'),  # <--- 关键列！
                week_z_sum=('score_z', 'sum'),
                judge_count=('judge_id', 'count')
            ).reset_index()

            # 2. 计算技术排名
            agg['tech_rank'] = agg.groupby(['season', 'week_num'])['week_avg_score'].rank(
                ascending=False, method='min'
            )

            # 3. 挂载危险区标记 (如果存在)
            if 'had_bottom_two_record' in df.columns:
                jeopardy = df.groupby(['season', 'week_num', 'celebrity_name'])[
                    'had_bottom_two_record'].max().reset_index()
                agg = agg.merge(jeopardy, on=['season', 'week_num', 'celebrity_name'], how='left')

            # 4. 挂载静态属性 (Age, Industry, Ballroom Partner)
            static_cols = ['celebrity_industry', 'celebrity_age_during_season', 'final_status',
                           'eliminated_week', 'placement', 'ballroom_partner']
            existing_static = [c for c in static_cols if c in df.columns]
            df_static = df.groupby(['season', 'week_num', 'celebrity_name'])[existing_static].first().reset_index()

            agg = agg.merge(df_static, on=['season', 'week_num', 'celebrity_name'], how='left')
            return agg

def run_transform_pipeline(df: pd.DataFrame) -> pd.DataFrame:
        transformer = DataTransformer()
        df_long = transformer.wide_to_long(df)
        df_clean = transformer.apply_survival_barrier(df_long)
        df_norm = transformer.apply_robust_normalization(df_clean)
        transformer.detect_structural_break(df_norm)

        # [关键修复] 直接返回聚合后的 Candidate，这就是我们的 Gold Layer 基准
        return transformer.generate_inference_aggregates(df_norm)

# --- 集成流水线调用接口 ---
def run_transform_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage 1 核心流水线：原始数据 -> 聚合因子表 (The Gold Candidate)。
    """
    transformer = DataTransformer()

    # 1. 宽转长
    df_long = transformer.wide_to_long(df)
    
    # 2. 清洗幽灵记录
    df_clean = transformer.apply_survival_barrier(df_long)
    
    # 3. 鲁棒去通胀标准化
    df_norm = transformer.apply_robust_normalization(df_clean)
    
    # 4. 制度断裂检测 (Forensics)
    transformer.detect_structural_break(df_norm)
    
    # 5. [关键] 执行聚合：从评委维度压缩到选手周度维度
    # 物理意义：将 4 个评委信号融合成 1 个技术基本面信号
    df_gold_candidate = transformer.generate_inference_aggregates(df_norm)
    
    return df_gold_candidate

if __name__ == "__main__":
    # 单元测试逻辑略...
    pass