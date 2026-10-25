"""Evaluate CrystalFormer-RL generations over training with independent golds + hole taxonomy.
Reward (from CSV): ORB-v3 single-point E_hull vs Alexandria hull (MP2020-corrected), as in CrystalFormer-RL.
Golds: MatterSim (independent MLIP) and CHGNet (trained on corrected energies => immune to the +U locality hole),
single-point on the same generated structure. Hulls: Alexandria (reward ref) and MP+WBM+Alexandria (more complete)."""
import argparse, glob, gzip, json, os, pickle, re, itertools, collections, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from pymatgen.core import Composition, Structure
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.io.ase import AseAtomsAdaptor

ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--every", type=int, default=5); args = ap.parse_args()
ROOT = os.environ.get("MATSCI_SCRATCH", "scratch"); D = f"{ROOT}/data"
csvs = sorted(glob.glob(f"{args.run}/*/relaxed_structures_ehull_*.csv"), key=lambda f: int(re.search(r"_(\d+)\.csv$", f).group(1)))
ep0 = int(re.search(r"_(\d+)\.csv$", csvs[0]).group(1))
csvs = [f for f in csvs if (int(re.search(r"_(\d+)\.csv$", f).group(1)) - ep0) % args.every == 0 or f == csvs[-1]]
print("epochs to evaluate:", [int(re.search(r"_(\d+)\.csv$", f).group(1)) for f in csvs], flush=True)

# reference hulls
ppd = pickle.load(gzip.open(f"{D}/ppd-mp.pkl.gz", "rb"))
s = pd.read_csv(f"{D}/wbm-summary.csv.gz"); s = s[s.uncorrected_energy.notna() & s.e_correction_per_atom_mp2020.notna()]
alex = pd.read_parquet(f"{D}/alex_hull.parquet")
OFF = json.load(open(f"{D}/alex_mp_elem_offsets.json"))   # mu_Alex - mu_MP per element
shift = lambda comp: sum(OFF[str(el)] * n for el, n in Composition(comp).items())
def index(entries):
    ix = collections.defaultdict(list)
    for e in entries: ix["-".join(sorted(map(str, e.composition.elements)))].append(e)
    return ix
IX_A = index([PDEntry(Composition(f), e) for f, e in zip(alex.formula, alex.energy_corrected)])            # raw (what the published reward uses)
IX_AA = index([PDEntry(Composition(f), e - shift(f)) for f, e in zip(alex.formula, alex.energy_corrected)])  # aligned to MP scale
IX_O = index([PDEntry(e.composition, e.energy) for e in ppd.all_entries] +
             [PDEntry(Composition(f), u + c * n) for f, u, c, n in zip(s.formula, s.uncorrected_energy, s.e_correction_per_atom_mp2020, s.n_sites)])
sub = lambda els, ix: [e for k in range(1, len(els) + 1) for c in itertools.combinations(sorted(els), k) for e in ix.get("-".join(c), [])]
HC = {}
def hulls(comp):
    els = [str(e) for e in comp.elements]; k = "-".join(sorted(els))
    if k not in HC:
        aa = sub(els, IX_AA)
        HC[k] = (PhaseDiagram(sub(els, IX_A)), PhaseDiagram(aa), PhaseDiagram(aa + sub(els, IX_O)))
    return tuple(p.get_hull_energy_per_atom(comp) for p in HC[k])
COMPAT = MaterialsProject2020Compatibility(check_potcar=False)
U = {"Co": 3.32, "Cr": 3.7, "Fe": 5.3, "Mn": 3.9, "Mo": 4.38, "Ni": 6.2, "V": 3.25, "W": 6.2}
def corr_epa(comp, energy, precorrected=False):
    if precorrected: return energy / comp.num_atoms
    els = {str(e) for e in comp.elements}; pu = bool(els & {"O", "F"}) and bool(els & set(U))
    ent = COMPAT.process_entry(ComputedEntry(comp, energy, parameters={"run_type": "GGA+U" if pu else "GGA",
                               "hubbards": {x: (U.get(x, 0) if pu else 0) for x in els}}), clean=True)
    return None if ent is None else ent.energy_per_atom

import torch
from mattersim.forcefield import MatterSimCalculator
from chgnet.model.dynamics import CHGNetCalculator
dev = "cuda" if torch.cuda.is_available() else "cpu"
MS = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device=dev); CG = CHGNetCalculator(use_device=dev)
def sp(atoms, calc):
    a = atoms.copy(); a.calc = calc
    try: return float(a.get_potential_energy())
    except Exception: return np.nan

rows = []
rng = np.random.default_rng(0)
for f in csvs:
    ep = int(re.search(r"_(\d+)\.csv$", f).group(1)); df = pd.read_csv(f)
    df = df[np.isfinite(df.relaxed_ehull) & (df.relaxed_ehull.abs() < 19.9)]            # drop failures / clipped
    take = df.sample(min(args.n, len(df)), random_state=int(rng.integers(1e9)))
    for _, r in take.iterrows():
        try:
            st = Structure.from_dict(eval(r.relaxed_cif) if isinstance(r.relaxed_cif, str) else r.relaxed_cif)
        except Exception:
            continue
        comp = st.composition; at = AseAtomsAdaptor.get_atoms(st)
        hA, hAA, hO = hulls(comp)
        orb_epa = r.relaxed_ehull + hA                                                  # invert published reward -> ORB MP2020 E/atom
        ems, ecg = sp(at, MS), sp(at, CG)
        ms_epa = corr_epa(comp, ems) if np.isfinite(ems) else None
        cg_epa = corr_epa(comp, ecg, precorrected=True) if np.isfinite(ecg) else None
        els = {str(e) for e in comp.elements}
        dmin = float(np.min(at.get_all_distances(mic=True)[np.triu_indices(len(at), 1)])) if len(at) > 1 else 9.0
        rows.append(dict(epoch=ep - ep0, formula=comp.reduced_formula, nel=len(els), natoms=len(at), dmin=dmin,
                         ureg=bool(els & {"O", "F"}) and bool(els & set(U)),
                         reward_eh=r.relaxed_eh if False else r.relaxed_ehull,
                         orb_eh_full=orb_epa - hO, ref_shift=hAA - hO, scale_artifact=hA - hAA,
                         hasF="F" in els, hasCl="Cl" in els, hasN="N" in els,
                         ms_eh_full=None if ms_epa is None else ms_epa - hO,
                         cg_eh_full=None if cg_epa is None else cg_epa - hO))
    print(f"epoch {ep - ep0}: evaluated {sum(1 for x in rows if x['epoch'] == ep - ep0)}", flush=True)
R = pd.DataFrame(rows); R.to_csv(f"{args.run}/eval_rl.csv", index=False)
# note: reward_stable uses the published (scale-mismatched) reward; *_stable columns use the MP-aligned full hull
R["model_err"] = R.ms_eh_full - R.orb_eh_full
R["short"] = R.dmin < 1.0
g = R.groupby("epoch").agg(n=("formula", "size"), reward_eh=("reward_eh", "mean"),
    reward_stable=("reward_eh", lambda x: np.mean(x <= 0)),
    ms_stable=("ms_eh_full", lambda x: np.mean(x <= 0)),
    true_stable=("ms_eh_full", lambda x: np.nan),
    cg_stable=("cg_eh_full", lambda x: np.mean(x <= 0)),
    model_err=("model_err", "median"), ref_shift=("ref_shift", "mean"), scale_art=("scale_artifact", "mean"),
    fracF=("hasF", "mean"), fracCl=("hasCl", "mean"), fracN=("hasN", "mean"), ureg=("ureg", "mean"), short=("short", "mean"), nel=("nel", "mean"))
ts = R.assign(t=(R.ms_eh_full <= 0) & (R.cg_eh_full <= 0.05) & ~R.ureg).groupby("epoch").t.mean()
g["true_stable"] = ts
pd.set_option("display.width", 220)
print(g.round(3).to_string())
