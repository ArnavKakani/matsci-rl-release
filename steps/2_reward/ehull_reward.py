"""Step 2: the reward the optimizers are given, in a stand-alone readable form (steps 3 and 7 contain the same logic inline).

reward = E_hull = corrected energy per atom of the candidate  -  energy per atom of the reference hull at its composition (eV/atom).
Lower is better, <= 0 is called "stable". Pieces:
  * energy       total energy of the relaxed candidate from an MLIP (MACE-MP-0, CHGNet, ...).
  * correction   MP2020 compatibility scheme applied to raw (non-CHGNet) energies, because the reference hull is MP2020-corrected.
                 +U transition metals (Co Cr Fe Mn Mo Ni V W) with O or F are treated as GGA+U. CHGNet is trained on corrected
                 energies, so it is NOT corrected again (precorrected=True).
  * hull         pymatgen PhaseDiagram of the reference database restricted to the candidate's chemical system.
usage (demo): python ehull_reward.py        prints E_hull of a relaxed MACE-MP-0 rock-salt structure vs the Materials Project hull
"""
import gzip, os, pickle
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedEntry
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility

COMPAT = MaterialsProject2020Compatibility(check_potcar=False)
U = {"Co": 3.32, "Cr": 3.7, "Fe": 5.3, "Mn": 3.9, "Mo": 4.38, "Ni": 6.2, "V": 3.25, "W": 6.2}


def e_hull(formula, total_energy, hull, precorrected=False):
    """hull: a pymatgen PhaseDiagram containing the candidate's chemical system. Returns eV/atom, or None if MP2020 rejects the entry."""
    comp = Composition(formula)
    if precorrected:
        return total_energy / comp.num_atoms - hull.get_hull_energy_per_atom(comp)
    els = {str(e) for e in comp.elements}
    plus_u = bool(els & {"O", "F"}) and bool(els & set(U))
    params = {"run_type": "GGA+U" if plus_u else "GGA", "hubbards": {e: (U.get(e, 0.0) if plus_u else 0.0) for e in els}}
    ent = COMPAT.process_entry(ComputedEntry(comp, total_energy, parameters=params), clean=True)
    return None if ent is None else ent.energy_per_atom - hull.get_hull_energy_per_atom(comp)


if __name__ == "__main__":
    from ase.build import bulk
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    from mace.calculators import mace_mp
    hull = pickle.load(gzip.open(os.environ.get("MATSCI_SCRATCH", "scratch") + "/data/ppd-mp.pkl.gz", "rb"))   # Matbench-Discovery MP phase diagram
    a = bulk("NaCl", "rocksalt", a=5.6, cubic=True); a.calc = mace_mp(model="medium", device="cpu", default_dtype="float64")
    FIRE(FrechetCellFilter(a), logfile=None).run(fmax=0.05, steps=200)
    print(a.get_chemical_formula(), "E_hull =", e_hull(a.get_chemical_formula(), a.get_potential_energy(), hull))
