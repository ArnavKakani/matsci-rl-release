# Step 2: The reward

`ehull_reward.py` is the reward the optimizers receive: E_hull of a relaxed candidate, MLIP energy plus the MP2020 correction
against a finite database hull (CHGNet is not corrected again, since it is trained on corrected energies). Read it first;
steps 3 and 7 contain the same logic inline.

## Commands

```bash
python ehull_reward.py      # one-structure smoke test (needs data/ppd-mp.pkl.gz and MACE)
python locality_probe.py    # optional, CPU
```

## Locality probe

`locality_probe.py` scores growing supercells of one metal with a single interstitial atom, using four MLIPs. The empirical
correction depends only on composition and a local MLIP only on nearby atoms, so any disagreement that grows with cell size
shows where the corrected reward stops being a property of the structure. It writes `locality.csv` in the current folder.
