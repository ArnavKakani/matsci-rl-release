"""Step 1: turn the Alexandria PBE download into the two files the other steps read.
usage: python build_alexandria_tables.py <alexandria entries .json[.bz2]>      (ComputedStructureEntry dicts; the file may be
either {"entries": [...]} or a plain list; each entry needs composition, structure, data.energy_corrected, data.mat_id)
writes $MATSCI_SCRATCH/data/alex_hull.parquet   (formula, energy_corrected: total MP2020-style corrected energy of the entry)
       $MATSCI_SCRATCH/data/alex_hull.json.bz2  ({"entries": [...]}: the same entries, with structures; step 6 and 8 read this)"""
import bz2, gzip, json, os, sys
import pandas as pd
from pymatgen.core import Composition
D = os.environ.get("MATSCI_SCRATCH", "scratch") + "/data"
src = sys.argv[1]
opener = bz2.open if src.endswith(".bz2") else gzip.open if src.endswith(".gz") else open
raw = json.load(opener(src, "rt"))
entries = raw["entries"] if isinstance(raw, dict) else raw
pd.DataFrame({"formula": [Composition(e["composition"]).formula for e in entries],
              "energy_corrected": [float(e["data"]["energy_corrected"]) for e in entries]}).to_parquet(f"{D}/alex_hull.parquet")
json.dump({"entries": entries}, bz2.open(f"{D}/alex_hull.json.bz2", "wt"))
print("alexandria entries:", len(entries))
