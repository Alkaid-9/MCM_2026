"""
MCM 2026: High-Performance MCMC Python Bridge
Role: Marshalling Gold-tier feature data into C++ pointers and capturing posterior distributions.
Standard: Industrial HPC Wrapper / Bayesian Convergence Monitoring.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, List, Any

# 动态定位并加载 C++ 编译生成的二进制库
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
sys.path.append(str(BIN_DIR))

try:
    import mcm_core_lib
except ImportError as e:
    logging.critical(f"致命错误：未能在 {BIN_DIR} 找到编译好的 mcm_core_lib.so/pyd。请先执行编译指令！")
    raise e

from src.etl.config_loader import ConfigLoader
from src.core.priors import VotePriors


class MCMCInferenceWrapper:
    """
    贝叶斯推理指挥官：
    1. 协调 ETL 产出的黄金因子库与底层 C++ 采样器。
    2. 执行周度切片反演。
    3. 自动化汇总不确定性指标 (R-hat, Entropy, Sigma)。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = logging.getLogger("MCMC_BRIDGE")

        # 实例化 C++ 类
        seed = self.cfg._config['project'].get('random_seed', 2026)
        self.cpp_sampler = mcm_core_lib.MCMCSampler(seed)
        self.prior_generator = VotePriors()

    def run_week_inference(self, week_df: pd.DataFrame) -> Optional[Any]:
        """
        针对特定（Season, Week）运行并行采样。
        物理意义：在给定的排名约束下，反演该周每位选手的观众得票率分布。
        """
        # 提取上下文信息
        season = int(week_df['season'].iloc[0])
        week = int(week_df['week_num'].iloc[0])
        mechanism = self.cfg.get_mechanism(season)

        # 1. 准备评委信号 (Judge Signals)
        # 确保为连续内存的 Float64 数组，以便 Eigen 无缝接管
        judge_signals = week_df['week_avg_score'].values.astype(np.float64)

        # 2. 定位淘汰者 (The Censored Index)
        # 逻辑：寻找当前周状态为 'Eliminated' 且淘汰周次匹配的人
        elim_mask = (week_df['final_status'] == 'Eliminated') & (week_df['eliminated_week'] == week)
        elim_indices = np.where(elim_mask)[0]

        if len(elim_indices) == 0:
            # 特殊处理：如决赛周或无淘汰周，由于缺乏强约束，反演退化为自由采样或先验保持
            self.logger.debug(f"S{season}W{week}: 无淘汰观测点，跳过贝叶斯约束反演。")
            return None

        elim_idx = int(elim_indices[0])

        # 3. 构建先验 (Prior Injection)
        # 利用 Stage 1 因子（舞伴 Alpha, 表现动量）生成 Zipf 分布初值
        prior_mu = self.prior_generator.informative_prior(week_df)

        # 4. 调度 MCMC 配置
        mcmc_params = self.cfg.get_inference_params()['mcmc_strategy']

        # --- 跨语言调用 (核心点火) ---
        # 此时 Python 释放 GIL，C++ 接管并启动 23 核并行
        try:
            result = self.cpp_sampler.run_parallel_inference(
                judge_signals,
                elim_idx,
                prior_mu,
                mechanism,
                n_chains=mcmc_params['n_chains'],
                n_samples=mcmc_params['samples_per_chain'],
                jump_size=0.05
            )
            return result
        except Exception as e:
            self.logger.error(f"C++ 内核在计算 S{season}W{week} 时崩溃: {e}")
            return None

    def run_batch_inference(self, df_gold: pd.DataFrame) -> pd.DataFrame:
        """
        遍历所有赛季和周次，将 Gold 层数据‘炼制’为 Platinum 层。
        """
        self.logger.info(f"开始批量贝叶斯反演，目标观测点: {df_gold['season'].nunique()} 赛季。")

        platinum_records = []

        # 按赛程顺序分组，确保逻辑一致性
        groups = df_gold.groupby(['season', 'week_num'])

        for (s, w), week_data in groups:
            # 这里的 reset_index 很关键，确保 elim_idx 对应 C++ 数组的偏移量
            week_data = week_data.reset_index(drop=True)

            res = self.run_week_inference(week_data)

            if res:
                # 提取选手名单
                contestants = week_data['celebrity_name'].values
                for i in range(len(contestants)):
                    platinum_records.append({
                        'season': s,
                        'week_num': w,
                        'celebrity_name': contestants[i],
                        # 核心反演产出
                        'est_fan_vote_mu': res.posterior_mean[i],
                        'est_fan_vote_sigma': res.posterior_std[i],
                        # 收敛与确定性量化 (Task 1 核心回答)
                        'inference_entropy': res.shannon_entropy,
                        'r_hat': res.r_hat,
                        'is_converged': res.converged,
                        'mcmc_acceptance': res.acceptance_rate
                    })

        self.logger.info("全量批量反演任务完成。")
        return pd.DataFrame(platinum_records)


# --- 单元测试 ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    # 模拟数据环境 (需确保 data/gold/factor_library.csv 已存在)
    try:
        df_test = pd.read_csv(PROJECT_ROOT / "data" / "gold" / "factor_library.csv")
        wrapper = MCMCInferenceWrapper()
        # 仅测试前 2 个周次
        sample_df = df_test[df_test['season'] == 1]
        results = wrapper.run_batch_inference(sample_df)
        print(results.head())
    except Exception as e:
        print(f"测试跳过或失败 (通常是因为文件不存在): {e}")