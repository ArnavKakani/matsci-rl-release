"""Step 6: plain-PBE reaction energy of each candidate vs its decomposition products (all computed with run_qe.py).
dft_dE   = E(candidate) - sum_i x_i E(product_i), eV/atom, plain PBE (no +U, no empirical corrections). Verdict uses this.
dE_anion = dft_dE + net MP2020 anion correction (GGA run type, no U terms): matters only when a product is an elemental anion;
           reported to show whether the verdict is robust near the hull.
Verdict: dft_dE > +tol "above hull", < -tol "below hull", else "on hull (|dE| <= tol)".   usage: python analyze_dft.py [--tol 0.05]
Reads $MATSCI_SCRATCH/dft/{plan.json,results/*.json}; writes dft/verdicts.csv."""
import argparse, glob, json, os, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

ap = argparse.ArgumentParser(); ap.add_argument("--tol", type=float, default=0.05); args = ap.parse_args()
DFT = os.environ.get("MATSCI_SCRATCH", "scratch") + "/dft"
plan = json.load(open(f"{DFT}/plan.json"))
res = {r["id"]: r for r in (json.load(open(f)) for f in glob.glob(f"{DFT}/results/*.json"))}
C = MaterialsProject2020Compatibility(check_potcar=False)


def anion_corr_pa(formula):
    comp = Composition(Composition(formula).alphabetical_formula)   # canonical order (the MP2020 H check is order-sensitive)
    e = C.process_entry(ComputedEntry(comp, 0.0, parameters={"run_type": "GGA", "hubbards": {}}), clean=True)
    if e is None:   # GGA entry rejected for U-element oxides/fluorides: keep only the anion terms
        e = C.process_entry(ComputedEntry(comp, 0.0, parameters={"run_type": "GGA+U", "hubbards": {}}), clean=True)
        if e is None: return 0.0
        return sum(a.value for a in e.energy_adjustments if "U" not in a.name and "Hubbard" not in a.name) / comp.num_atoms
    return e.correction / comp.num_atoms


jobs = {j["id"]: j for j in plan["jobs"]}
rows = []
for c in plan["candidates"]:
    need = [c["id"]] + [p for p, _ in c["products"]]
    ok = all(i in res and "energy_per_atom" in res[i] for i in need)
    row = dict(cat=c["cat"], formula=c["formula"], proxy_eh=c["proxy_eh"], judge_eh=c["judge_eh"], complete=ok)
    if ok:
        dE = res[c["id"]]["energy_per_atom"] - sum(a * res[p]["energy_per_atom"] for p, a in c["products"])
        dcorr = anion_corr_pa(c["formula"]) - sum(a * anion_corr_pa(jobs[p]["formula"]) for p, a in c["products"])
        row.update(dft_dE=dE, dE_anion=dE + dcorr, converged=all(res[i]["bfgs_converged"] for i in need),
                   products=" + ".join(f"{a:.2f} {jobs[p]['formula']}" for p, a in c["products"]))
    rows.append(row)
R = pd.DataFrame(rows)
pd.set_option("display.width", 240); pd.set_option("display.max_colwidth", 55)
done = R[R.complete]
print(f"complete: {len(done)}/{len(R)}\n")
if not done.empty:
    done = done.assign(verdict=np.select([done.dft_dE > args.tol, done.dft_dE < -args.tol], ["above hull", "below hull"], f"on hull (|dE|<={args.tol})"))
    print(done[["cat", "formula", "proxy_eh", "judge_eh", "dft_dE", "dE_anion", "converged", "verdict"]].round(3).to_string(index=False))
    print("\n", done.groupby(["cat", "verdict"]).size().unstack(fill_value=0))
R.to_csv(f"{DFT}/verdicts.csv", index=False)
