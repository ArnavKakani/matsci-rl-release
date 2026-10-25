"""Step 2 (optional diagnostic): is the reward local where the correction is global? Builds bcc W supercells with one O interstitial
of growing size, scores each with 4 MLIPs through the MP2020-corrected E_hull reward. The correction depends only on composition,
while a local MLIP only sees the O within its cutoff, so the two can disagree as the cell grows. Run on CPU; writes locality.csv here."""
import os
import gzip, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
from ase.build import bulk
from ase import Atom
from ase.optimize import FIRE
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
ppd = pickle.load(gzip.open(os.environ.get("MATSCI_SCRATCH", "scratch") + "/data/ppd-mp.pkl.gz", "rb")); C = MaterialsProject2020Compatibility(check_potcar=False)
def ehull(comp, e, pre):
    if pre: return e / comp.num_atoms - ppd.get_hull_energy_per_atom(comp)
    ent = C.process_entry(ComputedEntry(comp, e, parameters={"run_type": "GGA+U", "hubbards": {"W": 6.2, "O": 0}}), clean=True)
    return ent.energy_per_atom - ppd.get_hull_energy_per_atom(comp)
from mace.calculators import mace_mp
from chgnet.model.dynamics import CHGNetCalculator
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator
from mattersim.forcefield import MatterSimCalculator
calcs = {"mace": mace_mp(model="medium", device="cpu", default_dtype="float64"), "chgnet": CHGNetCalculator(use_device="cpu"),
         "orb": ORBCalculator(pretrained.orb_v3_conservative_inf_mpa(device="cpu", compile=False), device="cpu"),
         "mattersim": MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth", device="cpu")}
rows = []
for rep in [(1,1,1), (2,1,1), (2,2,1), (2,2,2), (3,2,2), (3,3,2), (3,3,3), (4,3,3), (4,4,4)]:
    a = bulk("W", "bcc", a=3.17, cubic=True).repeat(rep)
    a.append(Atom("O", position=(1.585, 1.585, 0.0)))        # octahedral interstitial
    d = a.get_distances(len(a) - 1, range(len(a) - 1), mic=True)
    frac_far = np.mean(d > 6.0)
    comp = Composition(a.get_chemical_formula(mode="metal"))
    out = {}
    for k, c in calcs.items():
        b = a.copy(); b.calc = c
        FIRE(b, logfile=None).run(fmax=0.05, steps=100)
        out[k] = ehull(comp, b.get_potential_energy(), pre=(k == "chgnet"))
    rows.append(dict(n_W=len(a) - 1, x_O=1 / len(a), frac_far=frac_far, **out)); print(rows[-1], flush=True)
import pandas as pd; pd.DataFrame(rows).to_csv("locality.csv", index=False)
