# Step 6: Plain-PBE DFT check

An MLIP-independent and correction-independent check of a few candidates, using Quantum ESPRESSO with SSSP pseudopotentials.

## Steps

1. Pick candidates from your own runs:

   ```bash
   python select_candidates.py [--per-cat 4] [--max-atoms 16]
   ```

   Takes final candidates the reward called stable and picks a few the held-out judges reject, a few they accept, and a few in
   the +U regime, plus their competing phases (decomposition products on the Alexandria hull). Reads `judged/rows.csv`,
   `runs/*.jsonl` and `data/alex_hull.json.bz2`; writes `dft/plan.json`.

2. Put the SSSP pseudopotentials in `$MATSCI_SCRATCH/dft/sssp/` (file names as in `sssp.json`).

3. Run every job:

   ```bash
   bash run_all_local.sh
   ```

   On a cluster, run one task per index of `plan.json` `jobs` instead: `python run_qe.py <i>`. Set `QE_BIN` and `NCPU`;
   `KSPACING` and `CALC=scf` are optional. Small cells first. Writes `dft/results/<id>.json` and `.pw.out`.

4. Compute each candidate's reaction energy to its products:

   ```bash
   python analyze_dft.py [--tol 0.05]
   ```

   Writes `dft/verdicts.csv`.

## Notes

The magnetic elements V-Ni are run spin-polarised with a ferromagnetic start; check `converged` and the output files before
trusting a number. Candidates whose products need very large cells or gaseous elements are skipped by the selector.
