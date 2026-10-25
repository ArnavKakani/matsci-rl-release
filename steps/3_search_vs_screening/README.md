# Step 3: Evolutionary search vs equal-budget screening

`run_search_vs_screening.py` runs one seed. Both arms spend the same number of proxy relaxations.

1. Adaptive arm: start from 96 random seed structures; each generation mutates parents (swap one element, rescale the cell),
   relaxes the children with the proxy MLIP, scores E_hull, and keeps the best unique formulas.
2. Passive arm: same starting structures; spends the same total number of proxy relaxations on one-shot random substitutions
   and keeps the best. This is the equal-budget baseline.
3. Judging: the top `--elite` candidates of each arm at generations `--checkpoints` are relaxed with ORB-v3 and MatterSim,
   starting from the proxy structure, and scored against the same reward hull (`gold_eh`).

## Inputs

`seeds.jsonl` and `data/` from step 1. Needs GPUs; workers are spread over the visible GPUs (`CUDA_VISIBLE_DEVICES`).

## Command

```bash
python run_search_vs_screening.py --proxy mace --objective ensmax --seed 0 --nproc 12
```

## Outputs

`runs/search_<proxy>_<objective><tag>_s<seed>.jsonl`, one line per (arm, generation, judge, candidate), including the relaxed
structure of ORB-judged candidates. Use many seeds, and the same seeds across variants, so step 5 can pair them.

## Options

| Option | Meaning |
|---|---|
| `--proxy {mace,chgnet}` | proxy model (required) |
| `--objective {proxy,ens,ensmax}` | `proxy`: the single model; `ens`: mean of MACE and CHGNet E_hull; `ensmax`: their maximum, a pessimistic reward |
| `--ref {mp,mpwbm}` | reward hull: MP, or MP + WBM (defense 1 of step 7) |
| `--mincov K` | reject chemical systems with fewer than K MP compounds |
| `--select {reward,random}` | `random` is a control with no selection on the reward |
| `--gens --pop --children --elite --checkpoints --nproc --seed` | search size and parallelism |
| `--tag` | suffix that keeps variants apart |
