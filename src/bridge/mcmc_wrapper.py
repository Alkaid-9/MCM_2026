"""
MCM 2026 Problem C: High-Performance MCMC Python Bridge
Role: Marshalling Gold-tier feature data into C++ pointers and capturing posterior distributions.
Standard: Industrial HPC Wrapper / Bayesian Convergence Monitoring / Memory Safety.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any

# --- 动态加载 C++ 内核 ---
# 自动定位 bin 目录，兼容 IDE 和 命令行 运行环境
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
sys.path.append(str(BIN_DIR))

try:
    import mcm_core_lib
except ImportError as e:
    # 友好的报错提示，防止因为编译失败导致 Python 侧一头雾水
    logging.critical(f"🛑 致命错误：未能在 {BIN_DIR} 找到编译好的 mcm_core_lib。")
    logging.critical("请先执行: cd cpp_kernel && mkdir build && cd build && cmake .. && make")
    raise e

from src.etl.config_loader import ConfigLoader
from src.core.priors import VotePriors

class MCMCInferenceWrapper:
    """
    贝叶斯推理指挥官 (The Bridge)：
    1. 数据编排：将 DataFrame 拆解为 C++ 可读的连续内存块 (Contiguous Memory Blocks)。
    2. 约束注入：将 'Eliminated' 和 'Bottom Two' 标签转化为数学索引。
    3. 并行调度：释放 GIL，指挥 C++ 核进行 23 路并行采样。
    4. 结果回收：将 C++ 的 Struct 结果映射回 Pandas Series。
    """

    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = logging.getLogger("MCMC_BRIDGE")

        # 初始化 C++ 采样器实例 (持有随机数引擎状态)
        seed = self.cfg._config['project'].get('random_seed', 2026)
        self.cpp_sampler = mcm_core_lib.MCMCSampler(seed)

        # 先验生成器
        self.prior_generator = VotePriors()

    def run_week_inference(self, week_df: pd.DataFrame) -> Optional[Any]:
        """
        针对特定（Season, Week）运行并行采样。
        物理意义：在给定的排名约束下，反演该周每位选手的观众得票率分布。
        """
        # --- 0. 上下文提取 ---
        if week_df.empty: return None
        season = int(week_df['season'].iloc[0])
        week = int(week_df['week_num'].iloc[0])
        mechanism = self.cfg.get_mechanism(season)

        # --- 1. 内存防御：准备评委信号 (Judge Signals) ---
        # 必须确保 dtype=float64 且内存连续 (C_CONTIGUOUS)
        # 否则 pybind11/Eigen 可能会读取到错误的内存地址
        judge_signals = np.ascontiguousarray(
            week_df['week_avg_score'].values,
            dtype=np.float64
        )

        # --- 2. 强约束：定位淘汰者 (The Censored Index) ---
        # 逻辑：寻找当前周状态为 'Eliminated' 且淘汰周次匹配的人
        elim_mask = (week_df['final_status'] == 'Eliminated') & (week_df['eliminated_week'] == week)
        elim_indices = np.where(elim_mask)[0]

        if len(elim_indices) == 0:
            # 决赛周或无淘汰周：退化为先验分布+评委分的一致性检查
            # 这里的 -1 是与 C++ 约定的“无淘汰”标志
            elim_idx = -1
        else:
            elim_idx = int(elim_indices[0])

        # --- 3. 辅助约束：定位危险区 (Jeopardy / Bottom Two) ---
        # 这是一个极强的硬约束：Bottom Two 的总分必须是全场倒数第二/第三
        # 从 ETL 解析的 'had_bottom_two_record' 字段获取
        if 'had_bottom_two_record' in week_df.columns:
            jeopardy_mask = np.ascontiguousarray(
                week_df['had_bottom_two_record'].values,
                dtype=np.int32 # C++ 侧接收 int 向量
            )
        else:
            jeopardy_mask = np.zeros(len(week_df), dtype=np.int32)

        # --- 4. 构建先验 (Prior Injection) ---
        # 利用 Stage 1 因子（舞伴 Alpha, 表现动量）生成 Zipf 分布初值
        # 同样需要内存对齐
        prior_mu = np.ascontiguousarray(
            self.prior_generator.informative_prior(week_df),
            dtype=np.float64
        )

        # --- 5. 调度 MCMC 配置 ---
        mcmc_params = self.cfg.get_inference_params()['mcmc_strategy']

        # --- 6. 跨语言调用 (核心点火) ---
        # 此时 Python 释放 GIL，C++ 接管并启动 OpenMP 并行
        try:
            # 注意：这里调用的是 bindings.cpp 中暴露的接口
            # 我们需要更新 C++ 接口以接收 jeopardy_mask (Task 1 精度提升的关键)
            result = self.cpp_sampler.run_parallel_inference(
                judge_signals,
                elim_idx,
                jeopardy_mask,  # [NEW] 传入危险区约束
                prior_mu,
                mechanism,
                n_chains=mcmc_params['n_chains'],
                n_samples=mcmc_params['samples_per_chain'],
                jump_size=0.05
            )
            return result

        except Exception as e:
            self.logger.error(f"💥 C++ 内核在计算 S{season}W{week} 时崩溃: {e}")
            # 可能是维度不匹配或数值溢出，返回 None 避免中断整个 Batch
            return None

    def run_batch_inference(self, df_gold: pd.DataFrame) -> pd.DataFrame:
        """
        遍历所有赛季和周次，将 Gold 层数据‘炼制’为 Platinum 层。
        """
        self.logger.info(f"🚀 开始批量贝叶斯反演，目标观测点: {df_gold['season'].nunique()} 赛季。")

        platinum_records = []

        # 按赛程顺序分组，确保逻辑一致性
        # 使用 sort_values 确保时间轴正确
        df_sorted = df_gold.sort_values(['season', 'week_num'])
        groups = df_sorted.groupby(['season', 'week_num'])

        total_groups = len(groups)
        processed = 0

        for (s, w), week_data in groups:
            # 这里的 reset_index 很关键，确保 elim_idx 对应 C++ 数组的偏移量 (0-based index)
            week_data = week_data.reset_index(drop=True)

            # 跳过只有1人的异常周（无法排名）
            if len(week_data) < 2: continue

            # 调用核心反演
            res = self.run_week_inference(week_data)

            if res:
                # 解包 C++ 结果 (InferenceResult Struct)
                contestants = week_data['celebrity_name'].values

                # 再次校验维度（防止 C++ 返回的维度与 Python 不一致）
                if len(res.posterior_mean) != len(contestants):
                    self.logger.error(f"S{s}W{w} 维度失配: Py={len(contestants)} vs C++={len(res.posterior_mean)}")
                    continue

                for i in range(len(contestants)):
                    platinum_records.append({
                        'season': s,
                        'week_num': w,
                        'celebrity_name': contestants[i],

                        # --- 核心反演产出 (Latent Variables) ---
                        'est_fan_vote_mu': res.posterior_mean[i],     # 估算票数均值
                        'est_fan_vote_sigma': res.posterior_std[i],   # 估算票数标准差

                        # --- 统计审计指标 (Forensics) ---
                        'inference_entropy': res.shannon_entropy,     # 整体系统的不确定性
                        'r_hat': res.r_hat,                           # 收敛性诊断 (需 < 1.1)
                        'is_converged': res.converged,                # 布尔收敛标志
                        'mcmc_acceptance': res.acceptance_rate        # 采样接受率 (0.2-0.5 为佳)
                    })

            processed += 1
            if processed % 50 == 0:
                self.logger.info(f"⏳ 进度: {processed}/{total_groups} 场比赛反演完成...")

        self.logger.info("✅ 全量批量反演任务完成。")
        return pd.DataFrame(platinum_records)

# --- 单元测试 ---
if __name__ == "__main__":
    # 配置日志格式
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 模拟数据环境
    try:
        # 构造一个模拟的 Gold Frame
        mock_gold = pd.DataFrame({
            'season': [99] * 4,
            'week_num': [1] * 4,
            'celebrity_name': ['StarA', 'StarB', 'StarC', 'Loser'],
            'week_avg_score': [9.0, 8.5, 8.0, 7.0], # 评委分
            'final_status': ['Safe', 'Safe', 'Safe', 'Eliminated'],
            'eliminated_week': [10, 10, 10, 1],
            'had_bottom_two_record': [0, 0, 1, 1], # C 和 Loser 处于危险区
            # 因子
            'partner_alpha': [1.0, 1.0, 1.0, 1.0],
            'score_delta': [0, 0, 0, 0]
        })

        wrapper = MCMCInferenceWrapper()
        print(">>> 正在测试 C++ 内核连接...")

        results = wrapper.run_batch_inference(mock_gold)

        if not results.empty:
            print("\n--- 反演结果预览 ---")
            print(results[['celebrity_name', 'est_fan_vote_mu', 'r_hat']].head())
            print("\n[PASS] 混合架构联调成功。")
        else:
            print("\n[FAIL] 未生成结果，请检查 C++ 库或输入数据。")

    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        print("提示: 确保已编译 C++ 内核 (bin/mcm_core_lib.so 存在)。")