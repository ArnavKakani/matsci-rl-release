"""Two robustness guards for the defended reward in CrystalFormer's reward.py. usage: python patch_degenerate.py <CrystalFormer checkout>

1. The defended reward scores raw (unrelaxed) samples with CHGNet. CHGNet's neighbour search never returns for a near-singular
   generated cell. A cell with a perpendicular height below 0.1 A or more than 5e7 expected neighbour pairs within 6 A gets c = inf
   (what the code already does when CHGNet raises), then is clipped like any other inf. Guarded samples are listed in
   <run>/degenerate_samples.csv.
2. If the ORB batch relaxation raises on a singular cell, redo that batch one structure at a time and give a failing structure
   energy inf (worst reward) instead of killing the run.
Requires make_defended_ehull_reward_fn in reward.py (see README.md)."""
import os, sys

GUARD = '''
def _degenerate_cell(st, rcut=6.0, min_height=0.1, max_pairs=5e7):
    """True if CHGNet's neighbour search would effectively never finish on this cell (see patch_degenerate.py)."""
    m = np.array(st.lattice.matrix, float); v = abs(float(np.linalg.det(m))); n = len(st)
    if not np.isfinite(v) or v < 1e-6:
        return True, v, 0.0, float("inf")
    h = min(v / np.linalg.norm(np.cross(m[(i + 1) % 3], m[(i + 2) % 3])) for i in range(3))
    pairs = n * n / v * 4.0 / 3.0 * np.pi * rcut ** 3
    return (h < min_height or pairs > max_pairs), v, h, pairs
'''

OLD = '''            try:
                a = ase_adaptor.get_atoms(st); a.calc = chg_calc
                c = a.get_potential_energy() / len(a) - hull_epa(st.composition)
            except Exception:
                c = np.inf'''
NEW = '''            bad, vol, hmin, pairs = _degenerate_cell(st)
            if bad:   # CHGNet would never return on this cell: same value as a CHGNet exception
                c = np.inf
                if path is not None:
                    _f = os.path.join(path, "degenerate_samples.csv"); _new = not os.path.exists(_f)
                    with open(_f, "a") as _h:
                        if _new: _h.write("epoch,formula,natoms,volume,min_height,pairs\\n")
                        _h.write(f"{epoch},{st.composition.reduced_formula},{len(st)},{vol:.6g},{hmin:.6g},{pairs:.6g}\\n")
                print(f"[degenerate cell] {st.composition.reduced_formula}: V={vol:.3g} A^3, min height {hmin:.3g} A, ~{pairs:.3g} pairs -> CHGNet skipped (inf)", flush=True)
            else:
              try:
                a = ase_adaptor.get_atoms(st); a.calc = chg_calc
                c = a.get_potential_energy() / len(a) - hull_epa(st.composition)
              except Exception:
                c = np.inf'''

def patch(cf_src):
    p = os.path.join(cf_src, "crystalformer/reinforce/reward.py")
    s = open(p).read()
    if "_degenerate_cell" in s:
        print("already patched:", p); return
    assert s.count(OLD) == 1, "reward.py changed: defended CHGNet block not found"
    anchor = "def make_defended_ehull_reward_fn("
    assert s.count(anchor) == 1
    s = s.replace(anchor, GUARD.lstrip("\n") + "\n\n" + anchor).replace(OLD, NEW)
    compile(s, p, "exec")
    with open(p + ".tmp", "w") as h:
        h.write(s)
    os.replace(p + ".tmp", p)   # atomic: running jobs keep their already-imported module
    print("patched:", p)

# ---- 2. ORB batch relaxation fallback. Failed structures get energy = inf and a placeholder Structure (same species and fractional
# coordinates, zero forces) so the epoch CSV keeps one row per sample; eval_rl.py drops |reward| >= 19.9 rows, so placeholders are never judged.
RELAX_OLD = '''    ase_adaptor = AseAtomsAdaptor()

    relaxation_trajectories = relaxer.relax(atoms_list)

    # Extract the relaxed structures and corresponding energies
    final_structures = [traj[-1] for traj in relaxation_trajectories.values()]
    final_energies = [structure.info['total_energy'] for structure in final_structures]
    final_structures = [ase_adaptor.get_structure(struct) for struct in final_structures]
    return final_structures, final_energies'''
RELAX_NEW = '''    ase_adaptor = AseAtomsAdaptor()

    try:
        relaxation_trajectories = relaxer.relax(atoms_list)
    except Exception as err:   # e.g. a singular cell; see cfrl/patch_degenerate.py
        print(f"[relax_structures] batch failed ({type(err).__name__}: {err}); relaxing one at a time", flush=True)
        out_s, out_e = [], []
        for at in atoms_list:
            try:   # one structure, on a FRESH relaxer: BatchRelaxer keeps the failed batch's optimizer instances, so
                   # reusing it makes every later call fail too
                r1 = BatchRelaxer(relaxer.potential, device=relaxer.device, fmax=relaxer.fmax, max_n_steps=relaxer.max_n_steps,
                                  max_natoms_per_batch=relaxer.max_natoms_per_batch, filter=relaxer.filter, optimizer=relaxer.optimizer)
                t1 = list(r1.relax([at]).values())[0][-1]
                out_s.append(ase_adaptor.get_structure(t1)); out_e.append(t1.info['total_energy'])
            except Exception as err1:
                cell = np.array(at.cell[:], float)
                frac = np.linalg.lstsq(cell.T, np.array(at.positions, float).T, rcond=None)[0].T
                ph = Structure(Lattice(cell), at.get_chemical_symbols(), frac, coords_are_cartesian=False,
                               site_properties={"forces": np.zeros((len(at), 3)).tolist()})
                print(f"[relax_structures] {ph.composition.reduced_formula}: {type(err1).__name__}; energy = inf (placeholder row)", flush=True)
                out_s.append(ph); out_e.append(np.inf)
        nfail = sum(1 for e in out_e if not np.isfinite(e))
        if nfail > max(2, len(atoms_list) // 10):   # systematic failure, not one bad cell: crash (the saved state stays clean)
            raise RuntimeError(f"relax_structures fallback: {nfail}/{len(atoms_list)} structures failed one at a time")
        return out_s, out_e

    # Extract the relaxed structures and corresponding energies
    final_structures = [traj[-1] for traj in relaxation_trajectories.values()]
    final_energies = [structure.info['total_energy'] for structure in final_structures]
    final_structures = [ase_adaptor.get_structure(struct) for struct in final_structures]
    return final_structures, final_energies'''


def patch_relax(cf_src):
    p = os.path.join(cf_src, "crystalformer/reinforce/reward.py")
    s = open(p).read()
    if "relaxing one at a time" in s:
        print("relax already patched:", p); return
    assert s.count(RELAX_OLD) == 1, "reward.py changed: relax_structures body not found"
    s = s.replace(RELAX_OLD, RELAX_NEW)
    compile(s, p, "exec")
    with open(p + ".tmp", "w") as h:
        h.write(s)
    os.replace(p + ".tmp", p)
    print("relax patched:", p)


if __name__ == "__main__":
    patch(sys.argv[1]); patch_relax(sys.argv[1])
