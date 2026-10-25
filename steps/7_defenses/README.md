# Step 7: Defenses

Two ways to make the reward harder to exploit. Run each with the same seeds and objective as your step-3 baseline, then judge
(step 4) and compare (step 5, `--compare`).

## 1. More complete reward hull

Adds WBM entries to the hull the optimizer is scored against.

```bash
python ../3_search_vs_screening/run_search_vs_screening.py --proxy mace --objective ensmax --ref mpwbm --tag _refwbm --seed 0 --nproc 12
```

## 2. Hull fuzzing

```bash
python run_fuzzing.py --proxy mace --objective ensmax --verifier fuzz --seed 0 --nproc 12
```

Before a chemical system is rewarded, the verifier relaxes up to `--fuzzB` competing structures there (prototypes from
`data/protolib.pkl`, elements swapped for chemically close neighbours, `--fuzzk` neighbours each) and adds them to the reward
hull. A competitor is only added at the higher of the two models' energies.

Only the adaptive arm is run. Output: `runs/fuzz_<proxy>_<objective>_<verifier><tag>_s<seed>.jsonl` and `_fuzzlog.json` (the
added competitors; step 4 needs it, and it is written when the run finishes).

| Option | Meaning |
|---|---|
| `--verifier {fixed,fuzz}` | `fixed` reproduces step 3 without selection variants |
| `--fuzzmode {pettifor,random}` | `random` is a budget-matched baseline with random prototypes |
| `--fuzzB`, `--fuzzk` | competitors per system, neighbours per prototype |
| `--stage {all,proxy,gold}` | which part of the run to execute |

The remaining options match step 3.
