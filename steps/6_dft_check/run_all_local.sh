#!/bin/bash
# Run every job in plan.json one after another on this machine (set NCPU to the MPI ranks to use).
# On a cluster, submit run_qe.py as an array job instead: one task per index 0..N-1 (N is printed by select_candidates.py).
set -e
S=${MATSCI_SCRATCH:?set MATSCI_SCRATCH}
N=$(python -c "import json; print(len(json.load(open('$S/dft/plan.json'))['jobs']))")
for i in $(seq 0 $((N - 1))); do python "$(dirname "$0")/run_qe.py" "$i"; done
