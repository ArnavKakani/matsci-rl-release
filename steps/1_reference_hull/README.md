# Step 1: Reference data

Builds everything the later steps read, under `$MATSCI_SCRATCH`. Run once.

MP is the usual reward hull. WBM and Alexandria add many more DFT-computed phases, and together the three form the more
complete hull that the judge in step 4 uses.

## Commands

```bash
bash download_data.sh
python build_alexandria_tables.py <alexandria entries file>
python make_seeds.py
python make_prototypes.py          # only needed for step 7
python build_combined_reference.py # only needed for step 8
```

| Command | Reads | Writes |
|---|---|---|
| `bash download_data.sh` | public Matbench-Discovery figshare files | `data/{ppd-mp.pkl.gz, wbm-summary.csv.gz, wbm-init-structs.jsonl.gz}` |
| `python build_alexandria_tables.py <alexandria entries file>` | an Alexandria PBE entries file (see below) | `data/alex_hull.parquet`, `data/alex_hull.json.bz2` |
| `python make_seeds.py` | WBM summary and starting structures | `seeds.jsonl` (random unique-prototype structures with at most 12 atoms; the search population is drawn from these) |
| `python make_prototypes.py` | `data/mp-cse.json.gz`, `data/mp-energies.csv.gz` | `data/protolib.pkl` (prototype structures for the fuzzing defense, step 7) |
| `python build_combined_reference.py` | MP, WBM, Alexandria | `data/combined_ref_aligned.json.bz2`, `data/alex_mp_elem_offsets.json` (step 8 only) |

## Input files you provide

- Alexandria file: ComputedStructureEntry dictionaries (pymatgen JSON) whose `data` holds `energy_corrected` and `mat_id`, as in
  the Alexandria PBE database or the hull file distributed with CrystalFormer-RL. If your file differs, adapt
  `build_alexandria_tables.py`; nothing else touches the raw file.
- `data/mp-cse.json.gz`: a table with columns `material_id` and `entry` (pymatgen ComputedStructureEntry dict).
- `data/mp-energies.csv.gz`: columns `material_id` and `energy_above_hull`.
