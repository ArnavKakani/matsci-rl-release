"""Step 5: turn the judged rows (step 4) into per-seed numbers, paired statistics and the missing-phase depth.

Per (run, arm, generation, seed) we average over the final candidates and the held-out judges:
  claimed        fraction the optimizer's own reward calls stable (proxy_eh <= 0)
  verified       fraction the held-out judges call stable against the complete hull (stable_strict; `--lenient` uses stable)
  depth_median   median missing-phase depth (ref_depth) of the candidates, eV/atom
  deep_share     fraction of candidates whose reward hull is more than --deep eV/atom too high (default 0.1)
Then, for every run that has both arms: verified(adaptive) vs verified(passive), PAIRED BY SEED, two-sided Wilcoxon signed-rank test.
With --compare A B: runs A and B (e.g. a defense vs its baseline), adaptive arm, Mann-Whitney U test across seeds.

usage: python summarize.py [--rows FILE] [--gen N] [--compare RUN_A RUN_B]      (default rows: $MATSCI_SCRATCH/judged/rows.csv)
"""
import argparse, os
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon

S = os.environ.get("MATSCI_SCRATCH", "scratch")
ap = argparse.ArgumentParser()
ap.add_argument("--rows", default=f"{S}/judged/rows.csv")
ap.add_argument("--gen", type=int, default=None, help="generation (checkpoint) to compare; default: the last one in the data")
ap.add_argument("--min-rows", type=int, default=0, help="drop seed cells with fewer judged rows (a full cell is elite x judges)")
ap.add_argument("--deep", type=float, default=0.1)
ap.add_argument("--lenient", action="store_true", help="count +U-regime candidates as verified when the judge says stable")
ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"))
args = ap.parse_args()

R = pd.read_csv(args.rows)
R["claimed"] = R.proxy_eh <= 0
R["verified"] = R.stable if args.lenient else R.stable_strict
R["deep"] = R.ref_depth > args.deep
seed_lv = R.groupby(["run", "arm", "gen", "seed"]).agg(
    claimed=("claimed", "mean"), verified=("verified", "mean"), depth_median=("ref_depth", "median"), deep_share=("deep", "mean"),
    n=("verified", "size")).reset_index()
seed_lv = seed_lv[seed_lv.n >= args.min_rows]
gen = args.gen if args.gen is not None else int(seed_lv.gen.max())
F = seed_lv[seed_lv.gen == gen]
os.makedirs(os.path.dirname(args.rows) or ".", exist_ok=True)
seed_lv.to_csv(os.path.join(os.path.dirname(args.rows), "seed_level.csv"), index=False)
pd.set_option("display.width", 200)


def wilcoxon_p(a, b):
    return wilcoxon(a, b).pvalue if len(a) >= 5 and (np.asarray(a) != np.asarray(b)).any() else np.nan


print(f"=== generation {gen}: adaptive vs passive, paired by seed ===")
for run, g in F.groupby("run"):
    a = g[g.arm == "adaptive"].set_index("seed"); p = g[g.arm == "passive"].set_index("seed")
    both = a.index.intersection(p.index)
    if len(both) < 2:
        print(f"{run}: {len(a)} adaptive seeds, no passive arm to pair with"); continue
    print(f"{run}: {len(both)} paired seeds | claimed a/p {a.loc[both].claimed.mean():.3f}/{p.loc[both].claimed.mean():.3f}"
          f" | verified a/p {a.loc[both].verified.mean():.3f}/{p.loc[both].verified.mean():.3f}"
          f" | Wilcoxon p={wilcoxon_p(a.loc[both].verified, p.loc[both].verified):.4f}")

print(f"\n=== generation {gen}: missing-phase depth (reward hull above complete hull), mean over seeds ===")
print(F.groupby(["run", "arm"])[["depth_median", "deep_share"]].mean().round(3).to_string())

print("\n=== verified fraction by generation, mean over seeds ===")
print(seed_lv.groupby(["run", "arm", "gen"]).verified.mean().unstack("gen").round(3).to_string())

if args.compare:
    x, y = (F[(F.run == r) & (F.arm == "adaptive")].set_index("seed").verified for r in args.compare)
    print(f"\n=== {args.compare[0]} vs {args.compare[1]} (adaptive arm, gen {gen}) ===")
    print(f"n={len(x)}/{len(y)} verified {x.mean():.3f} vs {y.mean():.3f}  Mann-Whitney p={mannwhitneyu(x, y).pvalue:.4f}")
