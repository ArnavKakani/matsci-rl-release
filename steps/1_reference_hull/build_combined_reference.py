"""Step 1/8: combined reference on the MP (MP2020) energy scale: MP + WBM as-is; Alexandria entries shifted by per-element
reference offsets (mu_MP - mu_Alex), which removes the scale mismatch between the two databases.
Writes data/combined_ref_aligned.json.bz2 (reward hull for step 8) and data/alex_mp_elem_offsets.json (read by step 8 eval)."""
import bz2, gzip, json, os, pickle
import pandas as pd
from pymatgen.core import Composition
S = os.environ.get("MATSCI_SCRATCH", "scratch"); D = S + "/data"
ppd = pickle.load(gzip.open(f"{D}/ppd-mp.pkl.gz", "rb"))
alex = pd.read_parquet(f"{D}/alex_hull.parquet")
def elem_refs(pairs):
    mu = {}
    for comp, e in pairs:
        c = Composition(comp)
        if c.is_element:
            el = str(c.elements[0]); mu[el] = min(mu.get(el, 9e9), e / c.num_atoms)
    return mu
mu_mp = elem_refs((e.composition, e.energy) for e in ppd.all_entries)
mu_al = elem_refs(zip(alex.formula, alex.energy_corrected))
off = {el: mu_al[el] - mu_mp[el] for el in mu_al if el in mu_mp}
out, skipped = [], 0
def add(comp, e):
    c = Composition(comp)
    out.append({"@module": "pymatgen.entries.computed_entries", "@class": "ComputedEntry", "energy": float(e),
                "composition": {str(k): float(v) for k, v in c.items()}, "correction": 0.0,
                "data": {"elements": sorted(str(x) for x in c.elements)}})
for f, e in zip(alex.formula, alex.energy_corrected):
    c = Composition(f)
    if not all(str(el) in off for el in c.elements): skipped += 1; continue
    add(c, e - sum(off[str(el)] * n for el, n in c.items()))
for e in ppd.all_entries: add(e.composition, e.energy)
s = pd.read_csv(f"{D}/wbm-summary.csv.gz"); s = s[s.uncorrected_energy.notna() & s.e_correction_per_atom_mp2020.notna()]
for f, u, c, n in zip(s.formula, s.uncorrected_energy, s.e_correction_per_atom_mp2020, s.n_sites): add(f, u + c * n)
json.dump({"entries": out}, bz2.open(f"{D}/combined_ref_aligned.json.bz2", "wt"))
json.dump(off, open(f"{D}/alex_mp_elem_offsets.json", "w"))
print("entries:", len(out), "skipped alex (no MP elemental ref):", skipped)
