"""Step 6: choose candidates from YOUR runs for a plain-PBE DFT check, plus the competing phases to compute with them.

Candidates come from the judged rows (step 4) and their structures from the run files (step 3/7). Three groups, `--per-cat` each,
all of them candidates the optimizer's own reward called stable (proxy_eh <= 0) at the last generation:
  judged_unstable   held-out judges say unstable against the complete hull (largest true E_hull first)
  judged_stable     held-out judges say stable (lowest true E_hull first)
  ureg              +U-regime candidates (O/F with Co Cr Fe Mn Mo Ni V W), where the correction scheme itself is in question
Competing phases = decomposition products on the Alexandria hull (structures from data/alex_hull.json.bz2). Only elements whose
SSSP pseudopotential fits the cutoffs below, cells up to --max-atoms, and decompositions with no gaseous element are eligible.
usage: python select_candidates.py [--rows FILE] [--per-cat 4] [--max-atoms 16]     writes $MATSCI_SCRATCH/dft/plan.json
"""
import argparse, bz2, collections, glob, itertools, json, os, re
import numpy as np, pandas as pd
from pymatgen.core import Composition, Element, Structure
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.io.ase import AseAtomsAdaptor

S = os.environ.get("MATSCI_SCRATCH", "scratch"); HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--rows", default=f"{S}/judged/rows.csv")
ap.add_argument("--runs", default=f"{S}/runs/*.jsonl")
ap.add_argument("--per-cat", type=int, default=4)
ap.add_argument("--max-atoms", type=int, default=16)
ap.add_argument("--ecut", type=float, nargs=2, default=[60.0, 480.0], help="wavefunction and density cutoffs (Ry)")
args = ap.parse_args()
sssp = json.load(open(f"{HERE}/sssp.json"))
OK_EL = {e for e, v in sssp.items() if v["cutoff_wfc"] <= args.ecut[0] and v["cutoff_rho"] <= args.ecut[1]
         and not (57 <= Element(e).Z <= 71 or Element(e).Z >= 89)}   # no lanthanides/actinides
els = lambda f: {str(x) for x in Composition(f).elements}

# ---- last-generation candidates: judge verdict (mean over judges) + proxy-relaxed structure ----
R = pd.read_csv(args.rows)
R = R[R.gen == R.groupby("run").gen.transform("max")]
e = R.groupby(["run", "seed", "arm", "formula"]).agg(proxy_eh=("proxy_eh", "mean"), judge_eh=("true", "mean"), ureg=("ureg", "max")).reset_index()
atoms = {}
for f in glob.glob(args.runs):
    run = re.sub(r"_s\d+\.jsonl$", "", os.path.basename(f))
    for l in open(f):
        r = json.loads(l)
        if r.get("atoms") is not None: atoms[(run, r["seed"], r["arm"], r["formula"])] = r["atoms"]
e = e[e.apply(lambda r: (r.run, r.seed, r.arm, r.formula) in atoms, axis=1)]
e = e[(e.proxy_eh <= 0) & e.formula.map(lambda f: els(f) <= OK_EL and Composition(f).num_atoms <= args.max_atoms and len(els(f)) >= 2)]
print("eligible candidates:", len(e))

# ---- competing phases from the Alexandria hull ----
GAS = {"H", "N", "O", "F", "Cl", "Br", "I"}
alex = json.load(bz2.open(f"{S}/data/alex_hull.json.bz2", "rt"))["entries"]
ix = collections.defaultdict(list)
for i, en in enumerate(alex): ix["-".join(sorted(en["data"]["elements"]))].append(i)
_dc = {}


def decomposition(formula):
    """[(entry index, structure, amount)] on the Alexandria hull; None if a gaseous element is a product or a cell is too large"""
    if formula in _dc: return _dc[formula]
    el = sorted(els(formula))
    idx = [i for k in range(1, len(el) + 1) for c in itertools.combinations(el, k) for i in ix.get("-".join(c), [])]
    pd_ = PhaseDiagram([PDEntry(Composition(alex[i]["composition"]), alex[i]["data"]["energy_corrected"], attribute=i) for i in idx])
    prods = []
    for ent, amt in pd_.get_decomposition(Composition(formula)).items():
        st = Structure.from_dict(alex[ent.attribute]["structure"])
        if len(st) > 24: st = st.get_primitive_structure()
        if (st.composition.is_element and str(st.composition.elements[0]) in GAS) or len(st) > 32:
            _dc[formula] = None; return None
        prods.append((ent.attribute, st, float(amt)))
    _dc[formula] = prods; return prods


def pick(df, n, key, ascending):
    out = []
    for _, r in df.sort_values(key, ascending=ascending).drop_duplicates("formula").iterrows():
        if decomposition(r.formula) is not None: out.append(r)
        if len(out) == n: break
    return pd.DataFrame(out)


cats = {"judged_unstable": pick(e[~e.ureg & (e.judge_eh > 0.05)], args.per_cat, "judge_eh", False),
        "judged_stable": pick(e[~e.ureg & (e.judge_eh <= 0)], args.per_cat, "judge_eh", True),
        "ureg": pick(e[e.ureg], args.per_cat, "judge_eh", True)}
structs, cands = {}, []
tolist = lambda a: {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in a.todict().items()}
for cat, df in cats.items():
    print(cat, len(df))
    for _, r in df.iterrows():
        prods = []
        for i, st, amt in decomposition(r.formula):
            key = f"alex_{alex[i]['data']['mat_id']}"
            structs.setdefault(key, {"id": key, "role": "phase", "formula": st.composition.reduced_formula, "atoms": tolist(AseAtomsAdaptor.get_atoms(st))})
            prods.append((key, amt))
        cid = f"cand_{cat}_{len(cands)}"
        structs[cid] = {"id": cid, "role": "candidate", "formula": r.formula, "atoms": atoms[(r.run, r.seed, r.arm, r.formula)]}
        cands.append({"id": cid, "cat": cat, "run": r.run, "arm": r.arm, "seed": int(r.seed), "formula": r.formula,
                      "proxy_eh": float(r.proxy_eh), "judge_eh": float(r.judge_eh), "products": prods})
jobs = sorted(structs.values(), key=lambda s: (s["role"] != "phase", len(s["atoms"]["numbers"])))   # small cells first
os.makedirs(f"{S}/dft", exist_ok=True)
json.dump({"ecut": args.ecut, "jobs": jobs, "candidates": cands}, open(f"{S}/dft/plan.json", "w"))
print(f"{len(cands)} candidates, {len(jobs)} DFT jobs -> {S}/dft/plan.json   (run_qe.py <0..{len(jobs) - 1}>)")
