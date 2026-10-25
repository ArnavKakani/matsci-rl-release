"""Step 4: re-judge the final candidates of every run against a MORE COMPLETE reference hull than the one the optimizer used.

For each record in a run file (written by step 3 or 7) the held-out MLIP (ORB-v3 or MatterSim, already evaluated in that step) gave
`gold_eh`, an E_hull against the hull the optimizer rewarded. Here we convert that back to an energy and measure it against the
complete hull = Materials Project + WBM + Alexandria, all on a formation-energy scale (each database referenced to its own elemental
ground states, which removes per-element offsets between databases).

  true       E_hull of the candidate under the held-out judge vs the complete hull (<= 0 means "stable")
  ref_depth  how far the optimizer's hull sits above the complete hull at this composition (missing-phase depth, eV/atom, >= 0)
  ureg       candidate has O or F plus a +U transition metal (Co Cr Fe Mn Mo Ni V W): the MP2020 +U correction applies, so every
             MLIP-only judgement there is unreliable; `stable_strict` counts such candidates as unverified

usage: python judge.py [--runs 'GLOB'] [--out FILE]
   defaults: $MATSCI_SCRATCH/runs/*.jsonl  ->  $MATSCI_SCRATCH/judged/rows.csv
needs $MATSCI_SCRATCH/data/{ppd-mp.pkl.gz, wbm-summary.csv.gz, alex_hull.parquet}  (step 1). Runs on CPU.
A run whose `<run>_fuzzlog.json` exists (step 7) is judged against the hull the fuzzer actually used (MP + the added competitors).
"""
import argparse, collections, glob, gzip, itertools, json, os, pickle, re
import numpy as np, pandas as pd
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram

S = os.environ.get("MATSCI_SCRATCH", "scratch")
ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=f"{S}/runs/*.jsonl")
ap.add_argument("--out", default=f"{S}/judged/rows.csv")
args = ap.parse_args()
UEL = {"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"}

# ---------------- databases ----------------
D = f"{S}/data"
MP = [PDEntry(e.composition, e.energy) for e in pickle.load(gzip.open(f"{D}/ppd-mp.pkl.gz", "rb")).all_entries]
s = pd.read_csv(f"{D}/wbm-summary.csv.gz"); s = s[s.uncorrected_energy.notna() & s.e_correction_per_atom_mp2020.notna()]
WB = [PDEntry(Composition(f), u + c * n) for f, u, c, n in zip(s.formula, s.uncorrected_energy, s.e_correction_per_atom_mp2020, s.n_sites)]
al = pd.read_parquet(f"{D}/alex_hull.parquet")
AL = [PDEntry(Composition(f), e) for f, e in zip(al.formula, al.energy_corrected)]


def elemental_refs(entries):
    mu = {}
    for e in entries:
        if len(e.composition.elements) == 1:
            x = str(e.composition.elements[0]); mu[x] = min(mu.get(x, 9e9), e.energy_per_atom)
    return mu


MU_MP, MU_AL = elemental_refs(MP), elemental_refs(AL)
key = lambda c: "-".join(sorted(str(x) for x in c.elements))


def index(entries):
    ix = collections.defaultdict(list)
    for e in entries: ix[key(e.composition)].append(e)
    return ix


def formation(entries, mu):
    return [PDEntry(e.composition, e.energy - sum(mu[str(el)] * a for el, a in e.composition.items()))
            for e in entries if all(str(el) in mu for el in e.composition.elements)]


def sub(els, ix):
    """all entries whose chemical system is a subset of `els`"""
    return [e for k in range(1, len(els) + 1) for c in itertools.combinations(sorted(els), k) for e in ix.get("-".join(c), [])]


IMP, IWB = index(MP), index(WB)                       # raw (MP2020-corrected) energies: what the optimizers' reward used
IMP_F, IWB_F, IAL_F = index(formation(MP, MU_MP)), index(formation(WB, MU_MP)), index(formation(AL, MU_AL))
_TRUE = {}


def hull_complete(c):
    """hull energy per atom on the formation scale, MP + WBM + Alexandria"""
    k = key(c)
    if k not in _TRUE:
        els = k.split("-"); _TRUE[k] = PhaseDiagram(sub(els, IMP_F) + sub(els, IWB_F) + sub(els, IAL_F))
    return _TRUE[k].get_hull_energy_per_atom(c)


# ---------------- judging ----------------
def run_name(path):
    return re.sub(r"_s\d+\.jsonl$", "", os.path.basename(path))


def judge_file(path):
    recs = [json.loads(l) for l in open(path) if l.strip()]
    flog = path.replace(".jsonl", "_fuzzlog.json")
    extra = index([PDEntry(Composition(x["formula"]), x["energy"]) for x in json.load(open(flog))]) if os.path.exists(flog) else None
    if "fuzz" in os.path.basename(path) and extra is None:
        return []                                      # fuzz run not finished (the fuzzlog is written last)
    cache, rows = {}, []
    for r in recs:
        if r.get("gold_eh") is None: continue
        c = Composition(r["formula"]); k = key(c); els = k.split("-"); ref = r.get("ref") or "mp"
        if (ref, k) not in cache:                      # the hull the optimizer's reward used
            cache[(ref, k)] = PhaseDiagram(sub(els, IMP) + (sub(els, extra) if extra is not None else []) + (sub(els, IWB) if ref == "mpwbm" else []))
        h_used = cache[(ref, k)].get_hull_energy_per_atom(c)
        mu = sum(MU_MP[str(el)] * a for el, a in c.fractional_composition.items())
        ef = r["gold_eh"] + h_used - mu                # judge's formation energy per atom
        h_true = hull_complete(c)
        e = set(els)
        rows.append(dict(run=run_name(path), seed=r["seed"], arm=r["arm"], gen=r["gen"], gold=r["gold"], formula=r["formula"],
                         proxy_eh=r["proxy_eh"], gold_eh=r["gold_eh"], ref=ref, n_eval=r.get("n_eval"), n_fuzz=r.get("n_fuzz"),
                         true=ef - h_true, ref_depth=(h_used - mu) - h_true,
                         ureg=bool(e & {"O", "F"}) and bool(e & UEL)))
    return rows


if __name__ == "__main__":
    rows = []
    for f in sorted(glob.glob(args.runs)):
        got = judge_file(f); rows += got; print("judged", os.path.basename(f), len(got), flush=True)
    d = pd.DataFrame(rows).drop_duplicates(["run", "seed", "arm", "gen", "gold", "formula"])
    d["stable"] = d.true <= 0
    d["stable_strict"] = d.stable & ~d.ureg
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    d.to_csv(args.out, index=False)
    print("wrote", args.out, len(d), "rows")
