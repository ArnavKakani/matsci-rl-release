"""Step 7 (defense): the step-3 search, but with a "fuzzing" verifier. Before the optimizer is rewarded for a chemical system, the
verifier relaxes up to --fuzzB competing structures in that system (prototypes from step 1 with neighbouring elements substituted)
and adds them to the reward hull, so a missing competing phase is likely to be found. The other defense, a more complete reward
hull, is step 3 with --ref mpwbm. Same inputs/outputs as step 3; files are named fuzz_<proxy>_<objective>_<verifier><tag>_s<seed>.jsonl,
with a _fuzzlog.json next to each (the competitors added to the hull). Needs GPUs."""
import argparse, gzip, json, os, pickle, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

D = os.environ.get("MATSCI_SCRATCH", "scratch")  # inputs: data/, seeds.jsonl (see top-level README)
OUT_DIR = f"{D}/runs"; os.makedirs(OUT_DIR, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument("--proxy", required=True, choices=["mace", "chgnet"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--gens", type=int, default=12)
ap.add_argument("--pop", type=int, default=96)
ap.add_argument("--children", type=int, default=192)
ap.add_argument("--elite", type=int, default=32)
ap.add_argument("--nproc", type=int, default=14)
ap.add_argument("--checkpoints", default="0,2,4,8,12")
ap.add_argument("--ref", default="mp", choices=["mp", "mpwbm"])  # reference hull used in the optimizer reward
ap.add_argument("--mincov", type=int, default=0)  # reject candidates whose chemsys has < K MP compounds
ap.add_argument("--tag", default="")
ap.add_argument("--verifier", default="fixed", choices=["fixed", "fuzz"])
ap.add_argument("--fuzzB", type=int, default=24)
ap.add_argument("--fuzzk", type=int, default=4)
ap.add_argument("--fuzzmode", default="pettifor", choices=["pettifor", "random"])  # random: arity-matched random prototypes, random species map (random-prototype baseline)
ap.add_argument("--stage", default="all", choices=["all", "proxy", "gold"])
ap.add_argument("--objective", default="proxy", choices=["proxy", "ens", "ensmax"])  # ens: mean, ensmax: max (pessimistic) of MACE and CHGNet E_hull
args = ap.parse_args()
CKPTS = [int(c) for c in args.checkpoints.split(",")]

# ---------------- workers ----------------
_calc = None; _calc2 = None
def _pick_gpu():
    """spread pool workers round-robin over the visible GPUs (no-op with one GPU); must run before torch import"""
    import os, multiprocessing as _mp
    devs = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",") if d != ""] or ["0"]
    ident = _mp.current_process()._identity
    os.environ["CUDA_VISIBLE_DEVICES"] = devs[((ident[0] - 1) if ident else 0) % len(devs)]

def _init(kind):
    global _calc, _calc2
    _pick_gpu()
    import torch; torch.set_num_threads(1)
    if kind == "mace+chgnet":
        from mace.calculators import mace_mp
        from chgnet.model.dynamics import CHGNetCalculator
        _calc = mace_mp(model="medium", device="cuda", default_dtype="float32")
        _calc2 = CHGNetCalculator(use_device="cuda")
    elif kind == "mace":
        from mace.calculators import mace_mp
        _calc = mace_mp(model="medium", device="cuda", default_dtype="float32")
    elif kind == "chgnet":
        from chgnet.model.dynamics import CHGNetCalculator
        _calc = CHGNetCalculator(use_device="cuda")
    elif kind == "orb":
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.calculator import ORBCalculator
        _calc = ORBCalculator(pretrained.orb_v3_conservative_inf_mpa(device="cuda", precision="float32-high", compile=False), device="cuda")
    elif kind == "mattersim":
        from mattersim.forcefield import MatterSimCalculator
        _calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cuda")

def _relax(atoms_dict, steps=200, fmax=0.05):
    from ase import Atoms
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    a = Atoms.fromdict(atoms_dict); a.calc = _calc
    try:
        opt = FIRE(FrechetCellFilter(a), logfile=None)
        opt.run(fmax=fmax, steps=steps)
        e = float(a.get_potential_energy())
        n = len(a)
        dmin = float(np.min(a.get_all_distances(mic=True)[np.triu_indices(n, 1)])) if n > 1 else 9.0
        if not np.isfinite(e) or e / n < -20 or a.get_volume() / n > 100:
            return None
        e2 = None
        if _calc2 is not None:
            b = a.copy(); b.calc = _calc2; e2 = float(b.get_potential_energy())
        out = a.copy(); out.calc = None
        return {"atoms": out.todict(), "energy": e, "energy2": e2, "dmin": dmin, "nsteps": opt.nsteps}
    except Exception:
        return None

def _tolist(d):
    return {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in d.items()}

# ---------------- thermodynamics (MP2020-corrected E_hull vs MP patched phase diagram) ----------------
from pymatgen.core import Composition, Element, Structure
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.io.ase import AseAtomsAdaptor
COMPAT = MaterialsProject2020Compatibility(check_potcar=False)
U = {"Co": 3.32, "Cr": 3.7, "Fe": 5.3, "Mn": 3.9, "Mo": 4.38, "Ni": 6.2, "V": 3.25, "W": 6.2}
PPD = None  # loaded in main only (spawned workers must not load it)
IX_MP = IX_REF = None; _HCACHE = {}
def _index(entries):
    import collections
    ix = collections.defaultdict(list)
    for e in entries:
        ix["-".join(sorted(map(str, e.composition.elements)))].append(e)
    return ix
def _sub(els, ix):
    import itertools
    return [e for k in range(1, len(els) + 1) for c in itertools.combinations(sorted(els), k) for e in ix.get("-".join(c), [])]
def hull_e(comp):
    """hull energy per atom vs the chosen reference; per-chemsys PhaseDiagram cache (validated vs PPD to 1e-4 eV/atom)"""
    from pymatgen.analysis.phase_diagram import PhaseDiagram
    els = [str(e) for e in comp.elements]; key = "-".join(sorted(els))
    if key not in _HCACHE:
        _HCACHE[key] = (PhaseDiagram(_sub(els, IX_REF)), sum(len(e.composition.elements) > 1 for e in _sub(els, IX_MP)))
    return _HCACHE[key][0].get_hull_energy_per_atom(comp)
def mp_cov(comp):
    hull_e(comp); return _HCACHE["-".join(sorted(str(e) for e in comp.elements))][1]

def e_hull(formula, energy, precorrected=False):
    comp = Composition(formula)
    if precorrected:  # CHGNet targets already include MP2020 corrections (Matbench Discovery note)
        return energy / comp.num_atoms - hull_e(comp)
    els = {str(e) for e in comp.elements}
    plus_u = bool(els & {"O", "F"}) and bool(els & set(U))
    params = {"run_type": "GGA+U" if plus_u else "GGA",
              "hubbards": {e: (U.get(e, 0.0) if plus_u else 0.0) for e in els}}
    ent = COMPAT.process_entry(ComputedEntry(comp, energy, parameters=params), clean=True)
    if ent is None:
        return None
    return ent.energy_per_atom - hull_e(comp)

# ---------------- search space ----------------
EXCL = {"He", "Ne", "Ar", "Kr", "Xe", "Tc", "Pm"}
ALLOWED = [Element.from_Z(z).symbol for z in range(1, 84) if Element.from_Z(z).symbol not in EXCL]
rng = np.random.default_rng(args.seed)

def mutate(atoms_dict):
    from ase import Atoms
    a = Atoms.fromdict(atoms_dict)
    syms = a.get_chemical_symbols(); present = sorted(set(syms))
    old = present[rng.integers(len(present))]
    new = ALLOWED[rng.integers(len(ALLOWED))]
    while new in present:
        new = ALLOWED[rng.integers(len(ALLOWED))]
    ro, rn = Element(old).atomic_radius or 1.5, Element(new).atomic_radius or 1.5
    frac = syms.count(old) / len(syms)
    a.set_chemical_symbols([new if s == old else s for s in syms])
    a.set_cell(a.cell * (1 + frac * (rn / ro - 1)), scale_atoms=True)
    return _tolist(a.todict())

def formula(atoms_dict):
    from ase import Atoms
    return Atoms.fromdict(atoms_dict).get_chemical_formula(mode="metal")

def score(results):
    out = []
    for r in results:
        if r is None: continue
        f = formula(r["atoms"])
        eh = e_hull(f, r["energy"], precorrected=(args.proxy == "chgnet"))
        eh_a, eh_b = eh, None
        if args.objective in ("ens", "ensmax") and eh is not None:
            eh_b = e_hull(f, r["energy2"], precorrected=True)
            eh = None if eh_b is None else (0.5 * (eh_a + eh_b) if args.objective == "ens" else max(eh_a, eh_b))
        if eh is not None and args.mincov and mp_cov(Composition(f)) < args.mincov:
            eh = None
        if eh is not None and r["dmin"] > 0.6:
            out.append({**r, "atoms": _tolist(r["atoms"]), "formula": f, "eh": eh, "eh_a": eh_a, "eh_b": eh_b})
    return out

def dedupe_top(cands, k):
    best = {}
    for c in cands:
        red = Composition(c["formula"]).reduced_formula
        if red not in best or c["eh"] < best[red]["eh"]:
            best[red] = c
    return sorted(best.values(), key=lambda c: c["eh"])[:k]

# ---------------- fuzzing verifier ----------------
FUZZED = set(); N_FUZZ = [0]; FUZZ_LOG = []
def chemsys(f): return "-".join(sorted(str(e) for e in Composition(f).elements))
def fuzz_setup():
    global BY_SYS, NBR, FRNG
    import itertools
    lib = pickle.load(open(f"{D}/data/protolib.pkl", "rb"))
    BY_SYS = {}
    for p in lib: BY_SYS.setdefault("-".join(p["els"]), []).append(p)
    els = sorted({e for p in lib for e in p["els"]} - {"He", "Ne", "Ar", "Kr", "Xe"})
    MN = {e: Element(e).mendeleev_no for e in els}
    NBR = {t: [t] + sorted([e for e in els if e != t], key=lambda e: abs(MN[e] - MN[t]))[:args.fuzzk] for t in els}
    NBR["_MN"] = MN
    global BY_ARITY
    BY_ARITY = {k: [p for p in lib if len(p["els"]) == k] for k in range(2, 9)}
    FRNG = np.random.default_rng(1000 + args.seed)
def fuzz_candidates(cs):
    import itertools
    from ase import Atoms
    S = cs.split("-"); MN = NBR["_MN"]; cands = {}
    for m in range(2, len(S) + 1):
        for T in itertools.combinations(S, m):
            if any(t not in NBR for t in T): continue
            for src in itertools.product(*[NBR[x] for x in T]):
                if len(set(src)) < m or src == T: continue
                for p in BY_SYS.get("-".join(sorted(src)), []):
                    sig = dict(zip(src, T))
                    cands[(p["mid"], tuple(sorted(sig.items())))] = (sum(abs(MN[a] - MN[b]) for a, b in sig.items()) + p["ehull"] + FRNG.random() * 1e-3, p, sig)
    if args.fuzzmode == "random":   # budget-matched: same number of competitors the Pettifor search would relax for this system
        nb = min(args.fuzzB, len(cands))
        subs = [T for m in range(2, len(S) + 1) for T in itertools.combinations(S, m) if BY_ARITY.get(m)]
        R = lambda e: Element(e).atomic_radius or 1.5; out = []
        for _ in range(nb if subs else 0):
            T = subs[FRNG.integers(len(subs))]; pool = BY_ARITY[len(T)]; p = pool[FRNG.integers(len(pool))]
            sig = dict(zip(p["els"], FRNG.permutation(list(T)))); sp = [str(sig[x]) for x in p["species"]]
            scale = (sum(R(x) ** 3 for x in sp) / sum(R(x) ** 3 for x in p["species"])) ** (1 / 3)
            out.append(_tolist(Atoms(sp, scaled_positions=p["frac"], cell=np.array(p["lattice"]) * scale, pbc=True).todict()))
        return out
    out = []
    R = lambda e: Element(e).atomic_radius or 1.5
    for _, p, sig in sorted(cands.values(), key=lambda x: x[0])[:args.fuzzB]:
        sp = [sig[x] for x in p["species"]]
        scale = (sum(R(x) ** 3 for x in sp) / sum(R(x) ** 3 for x in p["species"])) ** (1 / 3)
        out.append(_tolist(Atoms(sp, scaled_positions=p["frac"], cell=np.array(p["lattice"]) * scale, pbc=True).todict()))
    return out
def corr_epa(f, e, pre):
    comp = Composition(f); v = e_hull(f, e, precorrected=pre)
    return None if v is None else v + hull_e(comp)   # corrected energy per atom (e_hull plus hull energy)
def fuzz_systems(systems, relax_raw):
    from pymatgen.analysis.phase_diagram import PDEntry
    jobs = [a for cs in systems for a in fuzz_candidates(cs)]
    FUZZED.update(systems); N_FUZZ[0] += len(jobs)
    added = []
    for r in relax_raw(jobs):
        if r is None or r["dmin"] <= 0.6 or r["energy2"] is None: continue
        f = formula(r["atoms"]); comp = Composition(f)
        ea = corr_epa(f, r["energy"], args.proxy == "chgnet"); eb = corr_epa(f, r["energy2"], True)
        if ea is None or eb is None: continue
        added.append(PDEntry(comp, max(ea, eb) * comp.num_atoms))   # pessimistic: both models must agree the competitor is low
    for e in added:
        k = "-".join(sorted(str(x) for x in e.composition.elements)); IX_REF.setdefault(k, []).append(e)
        FUZZ_LOG.append({"formula": e.composition.formula, "energy": e.energy})
        for key in list(_HCACHE):
            if set(k.split("-")) <= set(key.split("-")): del _HCACHE[key]
def rescore(pop):
    out = []
    for c in pop:
        eh_a = e_hull(c["formula"], c["energy"], precorrected=(args.proxy == "chgnet")); eh = eh_a
        if args.objective in ("ens", "ensmax"):
            eh_b = e_hull(c["formula"], c["energy2"], precorrected=True)
            eh = None if (eh_a is None or eh_b is None) else (0.5 * (eh_a + eh_b) if args.objective == "ens" else max(eh_a, eh_b))
        if eh is not None: out.append({**c, "eh": eh, "eh_a": eh_a})
    return out
def verify(pop, relax_raw):
    if args.verifier != "fuzz": return pop
    for _ in range(3):
        new = {chemsys(c["formula"]) for c in pop[:args.elite]} - FUZZED
        if not new: break
        fuzz_systems(sorted(new), relax_raw); pop = dedupe_top(rescore(pop), args.pop)
    return pop

# ---------------- main ----------------
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    PPD = pickle.load(gzip.open(f"{D}/data/ppd-mp.pkl.gz", "rb"))
    ALLOWED = [e for e in ALLOWED if e in {str(el) for el in PPD.elements}]
    from pymatgen.analysis.phase_diagram import PDEntry
    import pandas as pd
    mp_entries = [PDEntry(e.composition, e.energy) for e in PPD.all_entries]
    IX_MP = _index(mp_entries)
    if args.verifier == "fuzz": fuzz_setup()
    if args.ref == "mpwbm":
        s = pd.read_csv(f"{D}/data/wbm-summary.csv.gz"); s = s[s.uncorrected_energy.notna() & s.e_correction_per_atom_mp2020.notna()]
        IX_REF = _index(mp_entries + [PDEntry(Composition(f), u + c * n) for f, u, c, n in zip(s.formula, s.uncorrected_energy, s.e_correction_per_atom_mp2020, s.n_sites)])
    else:
        IX_REF = IX_MP
    t0 = time.time()
    seeds = [json.loads(l) for l in open(f"{D}/seeds.jsonl")]
    idx = rng.choice(len(seeds), args.pop, replace=False)
    seeds = [seeds[i] for i in idx]
    out_path = f"{OUT_DIR}/fuzz_{args.proxy}_{args.objective}_{args.verifier}{args.tag}_s{args.seed}.jsonl"
    fout = open(out_path, "w") if args.stage != "proxy" else None
    def log(rec): fout.write(json.dumps(rec) + "\n"); fout.flush()

    snap_path = out_path.replace(".jsonl", "_snaps.pkl")
    if args.stage == "gold":
        st = pickle.load(open(snap_path, "rb")); snaps = st["snaps"]; N_FUZZ[0] = st["n_fuzz"]; FUZZ_LOG.extend(st["fuzzlog"])
        from pymatgen.analysis.phase_diagram import PDEntry as _PDE
        for x in FUZZ_LOG:
            e = _PDE(Composition(x["formula"]), x["energy"]); IX_REF.setdefault("-".join(sorted(str(y) for y in e.composition.elements)), []).append(e)
        _HCACHE.clear()
    ppool = None if args.stage == "gold" else ProcessPoolExecutor(args.nproc, initializer=_init, initargs=("mace+chgnet" if args.objective in ("ens", "ensmax") else args.proxy,))
    relax_raw = lambda ds: list(ppool.map(_relax, ds, chunksize=2))
    relax = lambda ds: score(relax_raw(ds))

    # adaptive arm (mu + lambda evolution on proxy E_hull, unique reduced formulas)
    if args.stage != "gold": pop = verify(dedupe_top(relax(seeds), args.pop), relax_raw)
    n_eval = args.pop
    if args.stage != "gold": snaps = {}
    for g in (range(args.gens + 1) if args.stage != "gold" else []):
        if g in CKPTS:
            snaps[("adaptive", g)] = (n_eval, pop[:args.elite])
            if args.stage == "proxy": pickle.dump({"snaps": snaps, "fuzzlog": FUZZ_LOG, "n_fuzz": N_FUZZ[0]}, open(snap_path, "wb"))  # checkpoint in case of timeout
        print(f"[{time.time()-t0:.0f}s] gen {g} n_eval={n_eval} best={pop[0]['eh']:.3f} "
              f"elite_mean={np.mean([c['eh'] for c in pop[:args.elite]]):.3f} n_fuzz={N_FUZZ[0]} fuzzed_sys={len(FUZZED)}", flush=True)
        if g == args.gens: break
        parents = [min(rng.choice(len(pop), 2), key=lambda i: pop[i]["eh"]) for _ in range(args.children)]
        kids = relax([mutate(pop[i]["atoms"]) for i in parents])
        n_eval += args.children
        pop = verify(dedupe_top(pop + kids, args.pop), relax_raw)

    # passive arm: one-shot random substitutions of the seeds, same total proxy budget
    budget = args.pop + args.gens * args.children
    passive_raw = []
    allp = []
    for g in []:
        n = args.pop + g * args.children
        snaps[("passive", g)] = (n, dedupe_top(allp[: n], args.elite))
    if ppool is not None: ppool.shutdown()
    print(f"[{time.time()-t0:.0f}s] proxy phase done", flush=True)
    if args.stage == "proxy":
        pickle.dump({"snaps": snaps, "fuzzlog": FUZZ_LOG, "n_fuzz": N_FUZZ[0]}, open(snap_path, "wb")); raise SystemExit

    # gold evaluation: relax each elite with each gold, starting from the proxy-relaxed structure
    for gold in ["orb", "mattersim"]:
        gpool = ProcessPoolExecutor(min(args.nproc, 8), initializer=_init, initargs=(gold,))
        for (arm, g), (n, elites) in snaps.items():
            res = list(gpool.map(_relax, [c["atoms"] for c in elites], chunksize=1))
            for c, r in zip(elites, res):
                geh = e_hull(c["formula"], r["energy"]) if r else None
                log({"proxy": args.proxy, "objective": args.objective, "eh_a": c["eh_a"], "eh_b": c["eh_b"], "ref": args.ref, "mincov": args.mincov, "seed": args.seed, "arm": arm, "gen": g, "n_eval": n,
                     "gold": gold, "formula": c["formula"], "proxy_eh": c["eh"], "gold_eh": geh, "verifier": args.verifier, "fuzzmode": args.fuzzmode, "fuzzB": args.fuzzB, "n_fuzz": N_FUZZ[0],
                     "dmin": c["dmin"], "gold_dmin": r["dmin"] if r else None,
                     "atoms": c["atoms"] if gold == "orb" else None})
        gpool.shutdown()
        print(f"[{time.time()-t0:.0f}s] gold {gold} done", flush=True)
    fout.close()
    json.dump(FUZZ_LOG, open(out_path.replace(".jsonl", "_fuzzlog.json"), "w"))
