# Step 4: Judge with a more complete reference

`judge.py` converts every judged record from steps 3 and 7 into an E_hull against MP + WBM + Alexandria (formation-energy
scale) and writes one row per (candidate, judge).

## Inputs

Runs from `runs/*.jsonl` and `data/` from step 1. CPU only; the first pass builds one phase diagram per chemical system, so it
can take a while. Fuzzing runs (step 7) are judged against the hull the fuzzer used (MP + its added competitors) before
comparing with the complete hull.

## Command

```bash
python judge.py [--runs 'GLOB'] [--out FILE]
```

## Output

`judged/rows.csv` (default), with these columns:

| Column | Meaning |
|---|---|
| `true` | E_hull under the held-out judge vs the complete hull; <= 0 is stable |
| `stable` | `true` <= 0 |
| `stable_strict` | stable, excluding the +U regime (`ureg`) |
| `ref_depth` | how far above the complete hull the reward's hull sat at that composition: the depth of the missing phases |
| `proxy_eh` | what the optimizer was told |
| `run`, `seed`, `arm`, `gen`, `gold`, `formula` | which run, seed, arm, generation, judge and formula the row is for |
