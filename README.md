# MCM_2026
# The Invisible Hand of the Audience: A Bayesian Inverse Optimization Framework for Mechanism Audit and Design

## 1. Project Overview
This repository contains the full industrial-grade pipeline implemented for **MCM 2026 Problem C**. We investigate the meritocracy-populism paradox within the *Dancing with the Stars* (DWTS) ecosystem. 

By treating the competition as a micro-market with asymmetric information, we developed a **Bayesian Inverse Optimization (BIO)** engine to reconstruct latent fan voting distributions from observed elimination outcomes. Furthermore, we proposed and validated a **Dynamic Adaptive Weighting (DAW)** mechanism to resolve systemic evaluative dissonance.

## 2. Core Architecture: The BIO-Pareto Pipeline
The system is orchestrated via a 6-stage high-performance pipeline:

*   **Stage 1: Data Forensics (ETL):** Robust Z-score calibration of judge signals across 34 seasons, filtering "ghost data" from eliminated contestants.
*   **Stage 2: Latent Reconstruction:** A hybrid C++17/Python MCMC engine utilizing OpenMP for 23-core parallel sampling on the probability simplex.
*   **Stage 3: Multiverse Forensics:** Counterfactual simulations of 34 seasons under alternative regimes (Rank vs. Percent) to quantify regime-dependent survival probabilities.
*   **Stage 4: Causal Attribution:** Decomposing success into "Talent Alpha" and "Partner Beta" using Hierarchical Linear Mixed-Effects Models (LMM) and non-linear SHAP explanations.
*   **Stage 5: Mechanism Design:** Multi-objective Pareto optimization to find the optimal authority transfer curve (DAW System).
*   **Stage 6: Academic Harvesting:** Automated generation of LaTeX deliverables and key research punchlines.

## 3. High-Performance Computing (HPC)
To handle the high-dimensional state space of latent fan preferences, the core sampling kernel is optimized for industrial scalability:
- **Backend:** C++17 with **Eigen3** for vectorized linear algebra.
- **Parallelism:** **OpenMP** multi-threading achieving throughput >100,000 samples/sec.
- **JIT Acceleration:** **Numba**-optimized simulation kernels for real-time grid searching in the Pareto space.

## 4. Key Numerical Findings
Our large-scale simulation yielded significant empirical insights:
- **Inference Fidelity:** **84.3%** (measured 84.29% on platinum posterior artifacts, n=2686; reconstructed latent variables highly consistent with historical outcomes).
- **Systemic Dissonance Index:** **1.24** (artifact-measured 1.2389; indicating a fundamental negative correlation between expert criteria and populist sentiment).
- **Fairness Lift:** The proposed DAW mechanism provides a **27.1% improvement** in technical merit alignment over the historical baseline (absolute equity-delta metric; the paper separately reports **+53.4%** as a relative gain).
- **Survival Longevity:** Optimized regimes achieved an **infinite survival expectation** for top-30% technical talent within the experimental window.

## 5. Getting Started
### Prerequisites
Ensure your environment meets the specifications in `requirements.txt`. A C++17 compatible compiler (GCC 11+ / Clang 14+) is required for the MCMC kernel.

### Execution Sequence
1. **System Audit:** Verify API contracts and environment stability.
   ```bash
   python check_dependencies.py