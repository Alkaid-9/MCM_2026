# Reproducibility notes

Status of what can and cannot be reproduced from this repository, based on an independent review of commit `20747c0`. Written for a reader who wants to know what to trust and what to expect if they try to run it.

**Summary.** Stage 1 reproduces exactly. The C++ sampling kernel and the downstream mixed-effects and DAW numbers were reproduced, but only with extra scripts or a test driver written during the review and not committed here. Stages 2 and 3 fail from a clean checkout for identified reasons, the Stage 5 IC audit fails with an `AttributeError`, and the rest of Stages 4–6 were not tested through the pipeline. The full pipeline has not been reproduced end to end.

Nothing below changes the contest code or its committed outputs. Where something was not run, it says so.

## 1. Verified environment

All checks ran on one machine:

| Item | Value |
|---|---|
| OS | Ubuntu 24.04, x86-64, 4 cores |
| Python | 3.12.3 (fresh virtual environment) |
| Key packages | pandas 2.3.3, numpy 2.5.4, scipy 1.18.1, numba 0.68.0, statsmodels 0.15.0, scikit-learn 1.9.1 (plus lifelines, xgboost, shap, seaborn, matplotlib, pyyaml, tqdm, joblib) |
| C++ (kernel test only) | g++ 13.3, Eigen 3.4.0, `-O2 -fopenmp` |

The versions pinned in `requirements.txt` (Python 3.10 era, for example numpy 1.26.4, pandas 2.3.3) were **not** tested. Do not treat them as verified. `requirements.txt` also lists `arviz`, which nothing imports, and omits `scikit-learn`, which four files import.

`main_pipeline.py` imports every stage at start-up, so all of the packages above are needed even to run Stage 1 alone.

## 2. What was reproduced

| Item | How | Result | Needs code not in this repo? |
|---|---|---|---|
| Stage 1 (ETL) | `python main_pipeline.py --stage 1` | 421 × 53 → 2,686 × 33. About 4–5 s. `data/silver/cleaned_long_format.csv` and `data/gold/factor_library.csv` are byte-identical (md5) to the committed files | No |
| C++ kernel, no-data target | Kernel sources compiled directly with a small test driver; Dirichlet target, 8 chains × 4×10⁵ draws | Sampler mean matches Dir(s·p) (L1 distance 0.001); acceptance 0.234–0.237. Bit-identical output over repeated runs with a fixed seed | **Yes**: the driver was written for the review |
| C++ kernel, real weeks | Same driver; 40-week random subset with eliminations, 23 chains × 10⁵ draws | Per-dimension mean split-R̂ 1.028; 94.6% of dimensions below 1.1 | **Yes**: same driver; a 40-week subset, not a full-dataset or end-to-end reproduction (the committed per-week configuration is `n_samples: 100000`, 23 chains) |
| LMM, ICC, dissonance index D | Repository classes `LMMAttributionEngine` and `DivergenceAnalyzer`, called from a short script | 2,686 gold rows, 2,681 matched to sampler output, 34 seasons. ICC judge 15.9%, fan 8.6%. D = 1.2398 | **Yes**: the script merges `est_fan_vote_mu` from `multiverse_history.csv` into the gold table, a step the pipeline does not do itself (see section 4) |
| DAW grid search, Season 27 | Repository classes `MultiverseEngine` and `MechanismEvaluator`, 20 × 20 grid of (k, t0) | "Equity" metric 0.5085 (Percent) → 0.7800: +53.4% relative, +27.1 points absolute (definition in section 2a). 14 distinct outcomes among 400 points; 34 points tie for the best distance | **Yes**: same merge script. Season 27 only, in-sample |

A rerun of the Stage 1, LMM and DAW rows in a clean copy under pandas 2.3.3 gave the same numbers listed here.

### 2a. What the DAW "fairness" and "engagement" numbers actually compute

From `src/solvers/objective_engine.py` (`MechanismEvaluator.evaluate_regime_performance`). For one simulated season it first collapses the history to one row per contestant:

- `sim_placement` aggregated with **`min`**: the contestant's best (lowest-numbered) simulated weekly placement, not a final standing;
- `cum_avg_tech_score` aggregated with **`last`**: the value in the contestant's last simulated week;
- `cum_avg_fan_vote` aggregated with **`last`**: likewise.

Then:

- **Equity** (called "fairness" in the paper) = absolute Spearman correlation between that `sim_placement` and `cum_avg_tech_score`;
- **Engagement** = absolute Spearman correlation between `sim_placement` and `cum_avg_fan_vote`.

Two details change how to read these:

- In the DAW search path (`pareto_optimizer.py:67-68`, also `design_pipeline.py:37-38, 71-72`), when the simulation output has no `cum_avg_tech_score` column it is filled from **`week_avg_score`** (that week's judge average, not a running mean), and `cum_avg_fan_vote` from `est_fan_vote_mu`. The simulation function does not emit these columns, so in the search path the "last" value is a last-week value. In the archived `multiverse_history.csv`, by contrast, `cum_avg_tech_score` is a running mean over weeks (checked by hand for one contestant), so archived columns and the code path that produced the 0.5085 → 0.7800 numbers are not the same quantity.
- The Season 27 run has 13 contestants. Checked on the Percent-rule run: the 0.5085 figure is the Spearman correlation over those 13 values of (best weekly placement, judge average in the contestant's own last simulated week), and that "last week" differs by contestant (it ranges from Week 1 to Week 8). Using a running mean of judge scores instead gives a different value (0.487 for the same run), so the choice of aggregation matters.

These are proxies defined by the implementation. They are not "final placement versus cumulative judge score". The recomputation above reproduces the numbers; it does not make the proxies a validated measure of fairness.

The C++ rows come from compiling the repository's own `.cpp` files with the headers unchanged. The `src/bridge/` Python wrapper and the pybind11 layer (`cpp_kernel/src/bindings.cpp`) were **not** run, and the committed `bin/*.so` was not checked against the sources.

## 3. Status of each stage

Two different things are recorded here, and they should not be confused: **failed when run** (an observed error) and **not tested** (nobody ran it through the pipeline). Everything below was observed in the same Python 3.12 environment.

| Stage | Status | Detail |
|---|---|---|
| 1 | **Ran, matches committed outputs** | See section 2 |
| 2 (inference) | **Failed when run** | `ImportError` when the wrapper loads the kernel. `bin/mcm_core_lib.cpython-310-x86_64-linux-gnu.so` is built for CPython 3.10 only. Building from source needs CMake, pybind11, Eigen ≥ 3.3, OpenMP and a C++17 compiler (`cpp_kernel/CMakeLists.txt`). The review environment had none of CMake, Eigen or pybind11, so the official CMake build was **never tried** |
| 3 (counterfactuals) | **Failed when run** | Exits with `断点数据缺失` ("breakpoint data missing"), raised at `main_pipeline.py:47`. It reads `data/platinum/final_posterior_results.csv`, the Stage 2 output. That directory is listed in `.gitignore` and was not committed. With the missing input worked around (section 4), the counterfactual engine itself was run for Season 27, but the full Stage 3 analyses (survival, signal-to-noise, noise stress) were **not** run |
| 4 (attribution) | **Not tested through the pipeline** | It also reads `data/platinum/`, so it could not start. The LMM, ICC and D parts were run separately (section 2). SHAP and XGBoost attribution were not run |
| 5 (design) | **Partly run; one step fails when run** | The DAW grid search was run separately (section 2). The Season-27 IC audit fails with `AttributeError`: `src/solvers/ic_simulator.py:187` calls `self._plot_ic_trajectory`, which is not defined anywhere in the final tree. It was deleted in the last contest-window commit (`6981e11`); the earlier commit `bc2ad30` still has it. The audit logic itself was run in the review with the plotting call removed |
| 6 (deliverables) | **Not tested** | Not run |

Also not run: the `--season` flag and `--stage all`. There is no code for the prior-sensitivity experiment the paper describes.

## 3a. What the counterfactual simulation does and does not establish

`MultiverseEngine.simulate_season` (`src/simulators/multiverse_engine.py`) is a week-by-week rule simulation. For each week it takes the contestants still alive **in the simulation**, reads their **historical** judge scores and inferred vote shares for that week, applies the Rank or Percent rule to decide who is eliminated, and removes that contestant from the simulated survivors. It does not generate scores or votes for situations that did not occur.

Consequences that were observed:

- **Contestants outlive their historical data.** A contestant who was eliminated in history but survives in the simulation has no historical record for later weeks, so they cannot be scored there. In the Season 27 reruns the number of contestants simulated per week differs from the historical number (for example 10 simulated against 11 historical in Week 3 under the Percent rule), and the simulation ends when fewer than two contestants have data.
- **Archived history has inconsistent elimination flags.** In `reports/mechanism_audit/multiverse_history.csv`, 245 of the 494 (season, universe, contestant) combinations flagged `is_simulated_loser = True` still have rows in later weeks, and some weeks flag more than one loser (44 weeks flag two, up to six in one week). The committed `simulate_season` does not write `is_simulated_loser`, `cum_avg_tech_score`, `cum_avg_fan_vote`, `judge_weight_applied` or `max_rank_this_week`, so the archived file was produced by some other version of the code, or post-processed. The cause was not traced.
- **`is_regime_anomaly` is not an elimination indicator.** It compares the simulated loser with the first historical `Eliminated` row for that week, not with the contestant actually eliminated that week. The `"Eliminated Week 1 (Regime Shift)"` entry in `mechanism_audit_summary.json` comes from this flag.
- **The survival analysis does not compare counterfactual universes.** `survival_analyst.py` groups by whether a season is in the Rank-regime list, so it compares historical Rank seasons with historical Percent seasons.
- **The result depends on the posterior version.** Under the earlier 2026-01-30 point estimate the same engine eliminates Bobby Bones in Week 5 (Rank) or Week 4 (Percent); under the sampler output he is never eliminated under Rank.

Everything simulated here is a rule illustration conditional on those inputs. It is not a validated alternative-season replay.

## 4. Two sets of posterior estimates

Task 1 has two sets of results in the repository, and downstream numbers depend on which one is used.

- **`data/gold/final_platinum_results.csv`** (column `est_fan_vote_pct`): an earlier point estimate from 2026-01-30, SLSQP/COBYLA based. The three committed `logs/mcm_full_run_*.log` files record this prototype, not the C++ sampler. Two of the three crashed.
- **`reports/mechanism_audit/multiverse_history.csv`** (column `est_fan_vote_mu`): the C++ sampler's posterior means, 5,362 rows (two rule regimes × 2,681 rows), 34 seasons. This is the only copy of the sampler output in the repository.

The LMM, ICC, D and DAW numbers in section 2 recompute on the second file, and only when it is joined onto the gold table (the review script does this on `season`, `week_num`, `celebrity_name`). The pipeline's own route to that join goes through `data/platinum/`, which is missing. The first file gives different results; for example the Season 27 champion's estimated share is about 2% there, against above 0.99 in 6 of 8 weeks in the second.

The paper's downstream numbers correspond to the second file. A reader trying to recompute them should start from `multiverse_history.csv`.

## 5. Other things to know before trying to run it

- **Parallelism is fixed to the original machine.** `OMP_NUM_THREADS` and related variables are set to 23 in `main.py` and `main_pipeline.py`, and `n_chains: 23` is in `conf/rules.yaml` and the C++ default. On a smaller machine that over-subscribes the CPU.
- **Committed build products.** `.gitignore` excludes `build/`, `*.so`, logs and data, but those files were force-added: `cpp_kernel/build/` (29 files, including CMake caches that contain the original author's local paths), `bin/*.so`, 14 `.pyc` files, a `.idea/` folder and the three run logs.
- **Unreachable code.** About 16 of 44 modules under `src/` are not imported from either entry point.
- **Sampling configuration** (`conf/rules.yaml`): seed 2026, 23 chains, 50% burn-in, thinning 20, prior strength 50.
- **Committed code may differ from the code behind the paper's numbers.** The C++ sources were last committed on 2026-02-02 00:13 and the `.so` on 2026-02-02 05:50 (UTC+8), while figures and reports were still being updated until 2026-02-03 05:21. The committed tree is therefore not guaranteed to be byte-identical to what produced every submitted result.

## 6. If you want to try

1. Create a Python 3.12 environment and install the packages listed in section 1 (not from `requirements.txt` as written).
2. `python main_pipeline.py --stage 1` and compare md5s with the committed `data/silver` and `data/gold` files.
3. For the downstream numbers, start from `reports/mechanism_audit/multiverse_history.csv` and join `est_fan_vote_mu` onto `data/gold/factor_library.csv` by `season`, `week_num`, `celebrity_name`. That join is not part of the repository's pipeline.
4. To test the sampler, install CMake, pybind11, Eigen 3.3+ and OpenMP, build `cpp_kernel/`, and expect to adjust paths and thread counts. This route has not been verified.

An end-to-end clean-environment reproduction, with the missing `data/platinum/` output regenerated and the Stage 5 method restored, has not been attempted and is not claimed.
