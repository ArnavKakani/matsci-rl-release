"""Step 1: draw the starting structures (random WBM prototypes, <= 12 atoms) the search starts from. Writes <MATSCI_SCRATCH>/seeds.jsonl."""
import gzip, json, numpy as np, pandas as pd
from pymatgen.core import Structure, Element
from pymatgen.io.ase import AseAtomsAdaptor
import os
S = os.environ.get("MATSCI_SCRATCH", "scratch")   # set MATSCI_SCRATCH, see top-level README
D = S
summ = pd.read_csv(f"{D}/data/wbm-summary.csv.gz", index_col="material_id")
ok_ids = set(summ.index[summ.unique_prototype & (summ.n_sites <= 12)])
EXCL = {"He", "Ne", "Ar", "Kr", "Xe", "Tc", "Pm"}
rng = np.random.default_rng(0); out = []
with gzip.open(f"{D}/data/wbm-init-structs.jsonl.gz", "rt") as f:
    for line in f:
        if rng.random() > 0.08: continue                 # subsample while streaming
        d = json.loads(line)
        if d["material_id"] not in ok_ids: continue
        s = Structure.from_dict(d["initial_structure"])
        if any(e.Z > 83 or e.symbol in EXCL for e in s.composition.elements): continue
        a = AseAtomsAdaptor.get_atoms(s)
        out.append({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in a.todict().items()})
with open(f"{S}/seeds.jsonl", "w") as f:
    for a in out: f.write(json.dumps(a) + "\n")
print("seeds:", len(out))
