"""Step 6: relax ONE structure from plan.json (written by select_candidates.py) with plain PBE (no +U, no empirical corrections)
in Quantum ESPRESSO. usage: python run_qe.py <job_index>   (writes $MATSCI_SCRATCH/dft/results/<id>.json and <id>.pw.out)
Env: KSPACING (max k-point spacing 1/Angstrom, default 0.25), CALC (vc-relax or scf, default vc-relax), QE_BIN (folder with pw.x and
mpirun; default: found on PATH), QE_WORKDIR (scratch for QE), NSTEP, SLURM_CPUS_PER_TASK or NCPU (MPI ranks, default 8).
Pseudopotentials: $MATSCI_SCRATCH/dft/sssp/<file listed in sssp.json>."""
import json, os, subprocess, sys, time
import numpy as np
from ase import Atoms
from ase.io import write, read

HERE = os.path.dirname(os.path.abspath(__file__))
DFT = os.environ.get("MATSCI_SCRATCH", "scratch") + "/dft"
QE = os.environ.get("QE_BIN", "")
QE = QE + "/" if QE else ""
plan = json.load(open(f"{DFT}/plan.json")); sssp = json.load(open(f"{HERE}/sssp.json"))
job = plan["jobs"][int(sys.argv[1])]
KSP = float(os.environ.get("KSPACING", "0.25"))   # max k-point spacing, 1/Angstrom, 2*pi included
RES = f"{DFT}/results"
out_json = f"{RES}/{job['id']}.json"
if os.path.exists(out_json):
    sys.exit(f"done already: {out_json}")
os.makedirs(RES, exist_ok=True)
ad = job["atoms"]
cell = ad["cell"]["array"] if isinstance(ad["cell"], dict) else ad["cell"]
a = Atoms(numbers=ad["numbers"], positions=ad["positions"], cell=np.array(cell, dtype=float), pbc=True)
from ase.build import niggli_reduce
niggli_reduce(a)   # compact cell => sensible k-mesh
CALC = os.environ.get("CALC", "vc-relax")
ecutwfc, ecutrho = job.get("ecut", plan["ecut"])   
MAG = {"V", "Cr", "Mn", "Fe", "Co", "Ni"}
species = sorted(set(a.get_chemical_symbols()))
magnetic = bool(set(species) & MAG)
work = f"{os.environ.get('QE_WORKDIR', '/tmp/qe')}/{job['id']}"; os.makedirs(work, exist_ok=True)
input_data = {
    "control": {"calculation": CALC, "prefix": "x", "pseudo_dir": f"{DFT}/sssp", "outdir": work,
                "tprnfor": True, "tstress": True, "etot_conv_thr": 1e-4, "forc_conv_thr": 1e-3, "nstep": int(os.environ.get("NSTEP", "80")), "disk_io": "none"},
    "system": {"ecutwfc": ecutwfc, "ecutrho": ecutrho, "occupations": "smearing", "smearing": "mv", "degauss": 0.01,
               "nspin": 2 if magnetic else 1},
    "electrons": {"conv_thr": 1e-8 * len(a), "mixing_beta": 0.3, "electron_maxstep": 200},
    "ions": {"ion_dynamics": "bfgs"},
    "cell": {"cell_dynamics": "bfgs", "press_conv_thr": 0.5},
}
if magnetic:
    a.set_initial_magnetic_moments([2.0 if s in MAG else 0.0 for s in a.get_chemical_symbols()])
pseudos = {s: sssp[s]["filename"] for s in species}
rec = a.cell.reciprocal()  # rows b_i without 2*pi
kpts = tuple(max(1, int(np.ceil(2 * np.pi * np.linalg.norm(b) / KSP))) for b in rec)
write(f"{work}/pw.in", a, format="espresso-in", input_data=input_data, pseudopotentials=pseudos, kpts=kpts)
n = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("NCPU", "8"))
t0 = time.time()
nkp = int(np.prod(kpts)); npool = max(1, min(n // 2, nkp))
while n % npool: npool -= 1
cmd = f"{QE}mpirun -np {n} --bind-to none --oversubscribe {QE}pw.x -nk {npool} -in {work}/pw.in > {work}/pw.out 2>&1"
rc = subprocess.call(cmd, shell=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
txt = open(f"{work}/pw.out").read()
res = {"id": job["id"], "role": job["role"], "formula": job["formula"], "natoms": len(a), "magnetic": magnetic, "kpts": kpts, "kspacing": KSP, "calc": CALC, "ecut": [ecutwfc, ecutrho],
       "returncode": rc, "seconds": time.time() - t0,
       "bfgs_converged": ("bfgs converged" in txt or "End final coordinates" in txt) if CALC != "scf" else ("convergence has been achieved" in txt),
       "scf_failed": "convergence NOT achieved" in txt}
try:
    fin = read(f"{work}/pw.out", format="espresso-out", index=-1)
    res.update(energy=float(fin.get_potential_energy()), energy_per_atom=float(fin.get_potential_energy()) / len(fin),
               final_atoms={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in fin.todict().items() if k in ("numbers", "positions", "cell", "pbc")})
except Exception as ex:
    res["error"] = repr(ex)[:300]
json.dump(res, open(out_json, "w"))
subprocess.call(f"cp {work}/pw.out {RES}/{job['id']}.pw.out", shell=True)
print(json.dumps({k: v for k, v in res.items() if k != "final_atoms"}))
