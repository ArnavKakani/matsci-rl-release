# Preregistration: E05, confirmatory replication of adaptive search vs screening

Written on 2026-10-06, before any E05 run was submitted. Commit this file before running `./submit.sh E05`; the commit
date is the registration date. Any change made after an E05 result exists is a deviation and must be reported as one.
Each E05 task refuses to start unless this file is committed and unchanged, and records the commit in
`$OUT_ROOT/E05/provenance/task<i>.txt`: the HEAD commits of the job scripts and the code, the PREREG.md blob sha, and the
commit and date that last touched PREREG.md. Comparing those dates with the job start times shows the order.

## Question

On rewards that contain CHGNet, does adaptive evolutionary search end with a lower judge-stable rate than best-of-N
screening with the same budget? The analysis uses no GGA+U exclusion rule.

The exploratory estimate comes from seeds 0-20 and the same three rewards, read from `seed_level_v2.csv.gz`:
−4.0 points, two-sided p = 0.019, with 16 of 21 seeds negative (REPORT A1).

## Design

- **Rewards.** These are the Table 1 configurations, unchanged (sources: `ga/README.md`).
  - `chgnet_proxy`: `expB.py --proxy chgnet --objective proxy`
  - `mace_ens`: `expB.py --proxy mace --objective ens`
  - `mace_ensmax`: `expB.py --proxy mace --objective ensmax`
- **Run settings.** All three use:
  - the MP reference hull;
  - 12 generations;
  - population 96, 192 children per generation and 32 elites;
  - 2,400 proxy relaxations per arm;
  - `--nproc 12` on one 24 GB GPU.
- **Arms.** Each run produces both arms, adaptive and screening, from the same 96 seed structures.
- **Seeds.** The seed list is 30, 31, 32, 33, 34, 35, 36, 37, 38, 39: ten seeds, fixed here. There is no extension and no
  replacement, whatever the result.
  - No existing run uses these seeds for these rewards.
  - Seeds 21-29 (E02) and seed 7 (E03) are not part of this analysis.
- **Judges.** ORB v3 conservative-inf-MPA and MatterSim v1.0.0-5M each relax the proxy structure.
  - Stability is measured against the MP+WBM+Alexandria hull on the formation-energy scale.
  - The code is `ood/audit_d0/audit2.py` with `rlenv/anD.py`, run through E04.
  - E04 writes the `rows2` and `seed_level_v2` files described in `ga/SCHEMA.md`.
- **Unit of analysis.** The seed.

## Primary endpoint

The primary endpoint is the rule-free paired deficit in the judge-stable rate, pooled over the three CHGNet rewards.

- **Rate.** For reward r, seed s and arm a at generation 12, `st` is the share of the 64 (elite, judge) rows with
  `true ≤ 0`: 32 elites times 2 judges. No GGA+U exclusion is applied. This is column `st` of `seed_level_v2`.
- **Per-reward difference.** d(r, s) = st(adaptive) − st(screening).
- **Seed-level difference.** D(s) is the mean of d(r, s) over the three rewards.
- **Test.** One-sided Wilcoxon signed-rank test on the D(s), run with scipy 1.18.1:
  `wilcoxon(D, alternative="less", zero_method="wilcox", method=PermutationMethod(n_resamples=np.inf))`.
  - Zeros are dropped (`"wilcox"`).
  - The p-value comes from enumerating all 2^n sign patterns, so it is exact even with tied |D(s)|. Ties are likely:
    every D(s) is a multiple of 1/192.
  - For n ≤ 13 this equals scipy 1.18.1's default `method="auto"`.
  - The alternative hypothesis is that the location of D is below 0; α = 0.05.
- **Decision.** The result is *confirmed* if p < 0.05, and *not confirmed* otherwise.
- **Reporting.** Report:
  - the mean of D in points, with a 95% percentile seed bootstrap (10,000 resamples, `numpy.random.default_rng(0)`, as
    in `assets2.py`);
  - how many seeds have D(s) < 0;
  - all ten D(s) and all 30 d(r, s).

```python
# uv run --no-project --with scipy==1.18.1 --with pandas --with numpy python
import glob, numpy as np, pandas as pd
from scipy.stats import wilcoxon, PermutationMethod
S = pd.concat(map(pd.read_csv, glob.glob(f"{OUT_ROOT}/E04/E05/*.seed_level_v2.csv")))
S = S[(S.gen == 12) & S.seed.between(30, 39) & S.run.isin(["chgnet_proxy", "mace_ens", "mace_ensmax"])]
d = S.pivot_table(index=["seed", "run"], columns="arm", values="st")
D = (d.adaptive - d.passive).unstack("run").mean(axis=1).dropna()   # available rewards per seed; a seed with none drops
p = wilcoxon(D, alternative="less", zero_method="wilcox", method=PermutationMethod(n_resamples=np.inf)).pvalue
```

## Secondary endpoints

Each secondary endpoint uses the same seeds, the same pooling over the three rewards, and the same one-sided test.

- **Status.** p-values are nominal, with no multiplicity correction.
- **Reporting.** All are reported, whatever they show.
- **Interpretation.** None of them can turn a "not confirmed" primary result into "confirmed".

| ID | Endpoint | Definition |
|---|---|---|
| S1 | W-only rule | `st_w`: an elite with W and O or F counts as not stable. |
| S2 | Full GGA+U rule | `st_strict`, the paper's metric: an elite with O or F plus any of Co, Cr, Fe, Mn, Mo, Ni, V or W counts as not stable. |
| S3 | Tolerance | Stable if `true ≤ τ`, for τ = 0.025 and τ = 0.05 eV/atom. Computed rule-free, and again with the full rule. |
| S4 | Both judges vs either judge | Rule-free, at τ = 0. An elite is stable only if both judges give `true ≤ 0` (both-judges), or if at least one does (either-judge). The per-seed rate is the mean over the 32 elites, computed from `rows2`. |
| S5 | Per reward | The paired test for each of the three rewards separately (n = 10). |
| S6 | Collapse | The share of runs with `st_strict < 0.05`, adaptive vs screening. Descriptive only. |

## Data handling, fixed in advance

- **Inclusion.** A (reward, seed, arm) cell is used if at least 60 of its 64 judge rows exist (`assets2.py:51`).
  - If either arm of a (reward, seed) pair fails this, d(r, s) is missing, and D(s) is the mean of the rewards that
    remain.
  - If no reward remains for a seed, the seed is dropped.
  - Every exclusion is reported.
- **Complete run file.** A run file is complete when it holds exactly 640 rows: 32 elites × 2 judges (orb, mattersim)
  × 2 arms × 5 checkpoints (0, 2, 4, 8, 12). Its job log must also end with `gold mattersim done`. A row whose judge
  relaxation failed (`gold_eh` null) still counts as a row; the inclusion rule above handles it. E04 additionally
  fails a run that yields fewer than the 10 seed-level cells a complete run gives.
- **Technical failures.** A task that ends without a complete run file is resubmitted with identical arguments. Causes
  include node or environment faults, the time limit and CUDA crashes. Only the node and `--nproc` may change.
  - After three failed attempts, the cell is missing.
- **No outcome-dependent reruns.** A run that completes is never rerun, replaced or discarded, whatever its result. The
  first completed run of each (reward, seed) is the one analysed. Combined-defense seeds 11-20 were once run twice and
  the earlier run discarded (REPORT B18); that must not happen here.
- **Code and data.** The code is the repository at the registration commit. The data are the `$DATA_ROOT` inputs that
  `ga/preflight.sh` checks.

## Power, and how a null result is read

A seed bootstrap of the 21 exploratory D(s) values (mean −4.0, sd 7.4 points) gives the power of this test:

| Seeds | Power |
|---|---|
| 10 | ≈ 0.51 |
| 15 | ≈ 0.71 |
| 20 | ≈ 0.81 |

With ten seeds, a non-significant result is weak evidence that there is no effect. It will be reported as "not
confirmed", together with the estimate and its interval. It will not be reported as "no difference".

## Outside this preregistration (exploratory)

- Pooling seeds 30-39 with seeds 0-20.
- Including the MACE reward (it is subject to the GGA+U artifact).
- Any analysis at a generation other than 12.
