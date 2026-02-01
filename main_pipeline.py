# ==============================================================================
# MCM 2026 Problem C: "The Invisible Hand"
# PIPELINE ORCHESTRATOR (DevOps Edition)
# Role: Modular execution engine supporting breakpoints and granular control.
# Usage: python main_pipeline.py --stage 3 --season 27
# Standard: Industrial DevOps / CI/CD Ready.
# ==============================================================================

import os
import sys
import argparse
import logging
import pandas as pd
import time
from pathlib import Path

# --- 环境霸权配置 (必须最先执行) ---
os.environ["OMP_NUM_THREADS"] = "23"
os.environ["MKL_NUM_THREADS"] = "23"
os.environ["NUMBA_NUM_THREADS"] = "23"

# 导入核心组件
from src.etl.config_loader import ConfigLoader
from src.utils.logger import setup_logger
from src.etl.pipeline import run_etl_stage
from src.bridge.mcmc_wrapper import MCMCInferenceWrapper
from src.analysis.mechanism_pipeline import MechanismAnalysisPipeline
from src.analysis.causality_pipeline import run_causality_stage
from src.solvers.design_pipeline import MechanismDesignPipeline
from src.utils.abstract_helper import AbstractHelper
from src.utils.exporter import MCMProjectExporter

class PipelineOrchestrator:
    """
    流水线指挥官：
    封装各阶段的业务逻辑，支持从磁盘加载中间数据进行‘断点续传’。
    """
    def __init__(self):
        self.cfg = ConfigLoader()
        self.logger = setup_logger("PIPELINE_ORCHESTRATOR")
        self.paths = self.cfg._config['paths']

    def _load_checkpoint(self, layer: str) -> pd.DataFrame:
        """[防御性加载] 从数据湖加载中间层数据"""
        path = self.cfg.get_path(layer) # e.g., 'gold_factors', 'platinum_results'
        if not os.path.exists(path):
            self.logger.critical(f"❌ 断点数据缺失: {path}")
            self.logger.critical(f"请先运行前置 Stage 生成该层数据！")
            sys.exit(1)

        self.logger.info(f"📂 加载数据检查点 ({layer}): {path}")
        df = pd.read_csv(path)

        # 类型安全补丁 (Pandas 读取 CSV 时偶尔会丢失 Int64 类型)
        if 'season' in df.columns: df['season'] = df['season'].astype(int)
        if 'week_num' in df.columns: df['week_num'] = df['week_num'].astype(int)

        return df

    def run_stage_1_etl(self):
        """[数据清洗] Bronze -> Gold"""
        self.logger.info(">>> [Stage 1] Executing ETL Pipeline...")
        run_etl_stage()

    def run_stage_2_inference(self):
        """[贝叶斯反演] Gold -> Platinum (C++ Kernel)"""
        self.logger.info(">>> [Stage 2] Igniting Bayesian Inference Engine...")
        df_gold = self._load_checkpoint('gold_factors')

        engine = MCMCInferenceWrapper()
        df_platinum = engine.run_batch_inference(df_gold)

        # 持久化
        out_path = self.cfg.get_path('platinum_results')
        df_platinum.to_csv(out_path, index=False)
        self.logger.info(f"✅ Platinum Layer Saved: {out_path}")

    def run_stage_3_forensics(self):
        """[机制审计] Platinum -> Audit Report"""
        self.logger.info(">>> [Stage 3] Launching Mechanism Forensics...")
        df_platinum = self._load_checkpoint('platinum_results')

        pipeline = MechanismAnalysisPipeline(df_platinum)
        report = pipeline.run_full_audit()

        gain = report['sensitivity_metrics'].get('robustness_advantage', 0)
        self.logger.info(f"✅ Forensics Complete. Robustness Gain: {gain:.2%}")

    def run_stage_4_attribution(self):
        """[因果归因] Platinum -> Attribution Plots"""
        self.logger.info(">>> [Stage 4] Analyzing Causal Drivers...")
        df_platinum = self._load_checkpoint('platinum_results')
        fig_dir = self.cfg.get_path('figures_dir')

        run_causality_stage(df_platinum, fig_dir=fig_dir)
        self.logger.info("✅ Causal Attribution Visualized.")

    def run_stage_5_design(self, target_season: int = 27):
        """[机制设计] Platinum -> Pareto Frontier"""
        self.logger.info(f">>> [Stage 5] Optimizing Mechanism Design (Target S{target_season})...")
        df_platinum = self._load_checkpoint('platinum_results')

        pipeline = MechanismDesignPipeline(df_platinum)
        pipeline.run_design_suite(target_season=target_season)
        self.logger.info("✅ Mechanism Design Optimized.")

    def run_stage_6_harvest(self):
        """[成果收割] Reports -> Abstract/Appendix"""
        self.logger.info(">>> [Stage 6] Harvesting Final Deliverables...")

        # 1. 摘要生成
        helper = AbstractHelper()
        metrics = helper.harvest_all_metrics()
        helper.generate_punchlines(metrics)

        # 2. 代码附录
        root_dir = Path(__file__).resolve().parent
        exporter = MCMProjectExporter(root_dir)
        exporter.run()
        self.logger.info("✅ Final Deliverables Ready.")

    def run_full_sequence(self):
        """一键运行全流程"""
        start = time.time()
        self.run_stage_1_etl()
        self.run_stage_2_inference()
        self.run_stage_3_forensics()
        self.run_stage_4_attribution()
        self.run_stage_5_design()
        self.run_stage_6_harvest()

        elapsed = (time.time() - start) / 60
        self.logger.info(f"🎉 All Stages Completed in {elapsed:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCM 2026 Pipeline Orchestrator")

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        help="Specify which stage to run (1=ETL, 2=Inference, ...)"
    )

    parser.add_argument(
        "--season",
        type=int,
        default=27,
        help="Target season for Mechanism Design optimization (Stage 5)"
    )

    args = parser.parse_args()

    orchestrator = PipelineOrchestrator()

    try:
        if args.stage == "1":
            orchestrator.run_stage_1_etl()
        elif args.stage == "2":
            orchestrator.run_stage_2_inference()
        elif args.stage == "3":
            orchestrator.run_stage_3_forensics()
        elif args.stage == "4":
            orchestrator.run_stage_4_attribution()
        elif args.stage == "5":
            orchestrator.run_stage_5_design(target_season=args.season)
        elif args.stage == "6":
            orchestrator.run_stage_6_harvest()
        elif args.stage == "all":
            orchestrator.run_full_sequence()

    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logging.critical(f"🔥 Pipeline Crashed: {e}", exc_info=True)
        sys.exit(1)