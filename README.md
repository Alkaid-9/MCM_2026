# MCM 2026 Problem C: Fan Votes vs. Judge Scores in *Dancing with the Stars*

Solo entry in the 2026 COMAP Mathematical Contest in Modeling, Problem C. Result: **Meritorious Winner** (control number 2622651).

Submitted paper title: *The Invisible Hand of the Audience: A Bayesian Inverse Optimization Framework for Latent Preference Recovery and Incentive-Compatible Mechanism Design.* Several claims in that paper (including "incentive-compatible") are not supported by code in this repository; see [Limitations](#limitations).

This repository is the code and archived outputs from that contest, kept as a record of a multi-stage modeling and computation project built by one person during the contest window (about 103 hours from first to last commit). It is **not** a verified research package: see [Limitations](#limitations) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) before relying on any number.

## The problem

Audience votes for the show are never published. Given weekly judge scores and who was eliminated (34 seasons, two aggregation regimes: rank-based and percent-based), the project asks:

1. **Latent votes.** What audience vote shares are consistent with the observed scores and eliminations?
2. **Rule comparison.** If a season had been scored under the other rule, how would simulated outcomes differ?
3. **Mechanism design.** Could a time-varying judge/fan weighting (called DAW here) trade off fairness against audience influence?

Vote shares are inferred, not observed. Everything downstream depends on the priors, constraints and simulation rules chosen here. None of it is recovered ground truth, and none of it is a causal effect identified from experimental data.

## What was built

| Stage | What it does | Main code |
|---|---|---|
| 1. ETL | Cleans the raw table (421 rows × 53 columns) into a 2,686-row, 33-column analysis table, with data-quality checks | `src/etl/`, `conf/rules.yaml` |
| 2. Inference | Samples latent weekly vote shares with a Metropolis–Hastings kernel written in C++17 (Eigen, OpenMP, pybind11) | `cpp_kernel/`, `src/bridge/`, `src/core/` |
| 3. Counterfactuals | Week-by-week simulation of the Rank and Percent rules on historical contestant and score records, using the inferred vote shares; survival, signal-to-noise and noise-stress analyses. This is not a validated replay of whole alternative seasons (see Limitations) | `src/simulators/`, `src/analysis/` |
| 4. Attribution | Mixed-effects models (random intercept per ballroom partner) of judge scores and inferred fan shares; a divergence index between the two | `src/analysis/attribution_lmm.py`, `src/analysis/cognitive_divergence.py` |
| 5. Mechanism design | Grid search over the DAW parameters (k, t0) with two objectives and a utopia-distance selection rule | `src/solvers/` |
| 6. Deliverables | Tables and figure export | `src/utils/` |

Entry point: `main_pipeline.py` (`--stage 1..6` or `all`, `--season` for Stage 5). `main.py` is the earlier all-stages script. Python: NumPy, pandas, SciPy, Numba, statsmodels, scikit-learn. C++: about 1,700 lines in `cpp_kernel/` (sources and headers).

## What has been checked

An independent code-and-reproduction review of this repository was done on commit `20747c0`. What it established, and under what conditions:

- **Stage 1 reproduces exactly.** In a fresh Python 3.12 environment, `python main_pipeline.py --stage 1` takes about 4–5 seconds and rewrites `data/silver/cleaned_long_format.csv` and `data/gold/factor_library.csv` byte-identically (md5) to the committed files.
- **The C++ sampling kernel behaves correctly for its own target.** The kernel sources were compiled with g++ 13 and Eigen 3.4 plus a small test driver written for the check (the driver is **not** in this repository, and the pybind11 layer was **not** exercised). On a no-data Dirichlet target the sampler mean matched the analytic mean (L1 distance 0.001) and the acceptance rate was 0.234–0.237. Runs are bit-identical across repeats with a fixed seed.
- **Mixed-effects results recompute** from the committed artifacts with this repository's own `LMMAttributionEngine` and `DivergenceAnalyzer`, on the 2,686-row gold table joined to the committed sampler output (2,681 matched rows, all 34 seasons):
  - Dissonance index **D = 1.2398**: one minus the cosine similarity between the judge-model and fan-model coefficient vectors (four coefficients, intercept and variance excluded). It depends on the reference category (1.08–1.62 under other choices) and has no confidence interval here.
  - Intraclass correlation by ballroom partner: **judge 15.9%, fan 8.6%**. Every celebrity-season has exactly one partner, so partner effects and celebrity effects are not separated.
- **The DAW grid search recomputes**, with a narrower claim than "optimal": on Season 27 only (13 contestants, 8 weeks, 78 rows), a 20 × 20 grid of (k, t0) raises the code's fairness metric from 0.5085 (Percent) to 0.7800, i.e. **+53.4% relative, +27.1 points absolute**. The metric is an implementation-defined rank-alignment proxy (absolute Spearman correlation across contestants; the exact aggregation is in `docs/REPRODUCIBILITY.md`), not a measure of final standings versus total judge score. The parameters are tuned and scored on the same season. The 400 grid points collapse to 14 distinct outcomes and 34 points tie for the best distance, so the published (k, t0) is one point inside a tied region.
- **Failed and unflattering runs are kept in the repository**: two crashed log files, and a noise-stress experiment in which the Rank rule flips more often than Percent (the reverse of one claim in the paper).

## Limitations

Read these before citing a number from this repository.

- **Scale mismatch in the sampler.** The kernel's Percent rule adds the raw weekly judge average (about 2.7–13) to a vote share (at most 1); the counterfactual engine uses a normalized judge share. The kernel also penalizes any week where the eventual champion is not top-ranked. As a result, the Season 27 posterior mean for the eventual champion exceeds 0.99 in 6 of 8 weeks. That is more likely a consequence of the model setup than a recovered vote count.
- **Pseudo-posterior.** The sampled target is a Dirichlet prior times quadratic penalty terms, not a probabilistic observation model. The acceptance ratio has no Jacobian term, so the effective prior is Dir(s·p) rather than Dir(1 + s·p).
- **The Zipf shape is an assumption**, built into the prior (`conf/priors.yaml`, exponent 1.2). It has not been estimated from the data.
- **Counterfactuals use historical weekly records and posterior means.** The simulation removes one simulated loser per week but keeps reading each week's historical contestants and scores, so a contestant who was eliminated in history but survives in the simulation has no later-week data to draw on. Posterior uncertainty is not propagated. In the committed history file, simulated eliminations are not always followed by disappearance from later weeks. The committed outputs also disagree with each other about one headline case (Season 27, Bobby Bones under the Rank rule): `reports/mechanism_audit/multiverse_history.csv` has him surviving all eight weeks, while `reports/mechanism_audit/mechanism_audit_summary.json` records an elimination in Week 1. Treat that case as unresolved.
- **Two sets of Task 1 posterior estimates exist.** `data/gold/final_platinum_results.csv` is an earlier (2026-01-30) SLSQP/COBYLA point estimate; the C++ sampler output exists only as the `est_fan_vote_mu` column of `reports/mechanism_audit/multiverse_history.csv`. The Task 2–4 numbers above recompute only on the latter. The sampler's full output directory, `data/platinum/`, was not committed.
- **The pipeline has not been run end to end from this repository.** Stage 1 runs; Stages 2 and 3 fail for identified reasons; the rest was not tested through the pipeline. See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).
- **Parts of the submitted paper are not backed by code here.** A Gini-based circuit breaker (only threshold values are in `conf/rules.yaml`), the LogNormal prior-sensitivity experiment, a formal incentive-compatibility argument, and the Showmance interaction estimate have no implementation in this repository.
- **Several figures in the submitted paper do not come from pipeline outputs.** They come from `__main__` demo blocks or hard-coded arrays in `src/vis/`, `src/analysis/convergence.py` and `src/solvers/ic_simulator.py`. For that reason this README embeds no figures.
  - Checked against the code and each other: the `est_fan_vote_mu` column of `reports/mechanism_audit/multiverse_history.csv`, and the `flip_rate_rank`, `flip_rate_percent` and `champion_misalignment` fields of `reports/mechanism_audit/mechanism_audit_summary.json` (definitions and caveats in `docs/REPRODUCIBILITY.md`).
  - Not to be cited as experimental conclusions: the survival and comparison tables (the two `.tex` files and the JSON disagree with one another), the Bobby Bones case (above), and `reports/mechanism_audit/mechanism_stability_curve.png`, whose arrow label ("Percent System is more sensitive to noise") contradicts the curves drawn in the same figure.
- **Hard-coded to the original machine**: 23 chains and 23 threads (`conf/rules.yaml`, `main.py`, `main_pipeline.py`). `bin/*.so` is a build for CPython 3.10 on x86-64 Linux.

## Repository layout

```text
MCM_2026/
├── conf/                 # Rules, priors, column mappings
├── cpp_kernel/           # C++17 sampling kernel and CMake build (build/ output is also committed; see REPRODUCIBILITY)
├── bin/                  # Prebuilt CPython 3.10 extension (not portable)
├── data/
│   ├── bronze/           # Raw contest data
│   ├── silver/           # Cleaned long-format table
│   └── gold/             # Analysis table; earlier point-estimate posterior
├── src/
│   ├── etl/              # Stage 1
│   ├── bridge/, core/    # Stage 2 Python side
│   ├── simulators/       # Stage 3 counterfactual engines
│   ├── analysis/         # Stages 3–4
│   ├── solvers/          # Stage 5
│   ├── utils/            # Stage 6 and helpers
│   └── vis/              # Plotting scripts, several with demo `__main__` data
├── reports/mechanism_audit/   # Sampler output used downstream, summary JSON, LaTeX tables, figures
├── reports/final_design/      # Stage 5 figures
├── logs/                 # Archived run logs, including two crashed runs
├── main_pipeline.py      # Stage-oriented entry point
├── main.py               # Earlier entry point
├── check_dependencies.py
└── requirements.txt
```

About 16 of the 44 modules under `src/` are not reachable from either entry point (plotting scripts, an older SLSQP/COBYLA solver, `src/analysis/convergence.py`, `src/simulators/adversarial_test.py`).

## Reproducing

Short version: **Stage 1 runs and matches the committed outputs.** Stages 2 and 3 fail from a clean checkout for identified reasons (a CPython 3.10-only binary; a missing `data/platinum/` file), and the Stage 5 IC audit fails with an `AttributeError`. Stage 4, the rest of Stage 5 and Stage 6 were not run end to end through the pipeline. Exact environments, commands, what was recomputed with the help of extra scripts, and the status of each stage are in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Provenance and AI use

- Solo entry. All 22 commits inside the contest window are under one author name. They fall in the window (2026-01-29 22:30 to 2026-02-03 05:21, UTC+8, about 103 hours first to last). Later changes are small: a two-line class-name fix (2026-08-11) in a module no entry point imports, and README edits.
- AI use, as disclosed in the paper's AI-use report (p. 25):
  - Tools: ChatGPT (GPT-4o) for code syntax optimization and LaTeX formatting; GitHub Copilot for auto-completion during Python and C++ development.
  - Code: refactoring Python loops into vectorized NumPy/Numba operations, and generating C++ syntax for the MCMC kernel.
  - Document formatting: LaTeX for complex formulas (for example the hierarchical LMM equations) and tables.
  - Language polishing: grammar and flow of the paper's Abstract and Memo.
  - The report does not itemize which functions were AI-assisted, so no part of the code here should be assumed to be assistance-free.
- The competition paper is not included in this repository. Check COMAP's terms before redistributing competition materials.

## Status

Archived contest project. Reproducibility and several known modeling issues (listed above) are open; they have not been fixed in the contest code, and the committed outputs have not been regenerated.
