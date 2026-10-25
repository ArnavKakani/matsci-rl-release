"""Step 1: prototype library for the fuzzing defense (step 7): Materials Project structures with <= 16 atoms, 2-4 elements and
E_above_hull <= 0.1 eV/atom. Reads $MATSCI_SCRATCH/data/{mp-energies.csv.gz,mp-cse.json.gz}; writes data/protolib.pkl."""
import os, pickle
import pandas as pd
D = os.environ.get("MATSCI_SCRATCH", "scratch") + "/data"
en = pd.read_csv(f"{D}/mp-energies.csv.gz").set_index("material_id")   # columns: material_id, energy_above_hull
df = pd.read_json(f"{D}/mp-cse.json.gz")                               # columns: material_id, entry (ComputedStructureEntry dict)
lib = []
for mid, ent in zip(df.material_id, df.entry):
    st = ent["structure"]; n = len(st["sites"])
    if n > 16 or mid not in en.index or en.at[mid, "energy_above_hull"] > 0.1: continue
    els = sorted({sp["element"] for s in st["sites"] for sp in s["species"]})
    if len(els) < 2 or len(els) > 4: continue
    lib.append({"mid": mid, "els": els, "ehull": float(en.at[mid, "energy_above_hull"]),
                "lattice": st["lattice"]["matrix"], "frac": [s["abc"] for s in st["sites"]],
                "species": [s["species"][0]["element"] for s in st["sites"]]})
print("prototype library:", len(lib), "by arity:", pd.Series([len(x["els"]) for x in lib]).value_counts().to_dict())
pickle.dump(lib, open(f"{D}/protolib.pkl", "wb"))
