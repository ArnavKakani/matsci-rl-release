#!/bin/bash
# Download the public Matbench-Discovery files used as reference data. Usage: MATSCI_SCRATCH=/your/dir bash download_data.sh
set -e
S=${MATSCI_SCRATCH:?set MATSCI_SCRATCH to a directory with ~20 GB free}
mkdir -p "$S/data"; cd "$S/data"
g() { curl -sL -A "Mozilla/5.0" --retry 3 -o "$2" "$1" && echo "ok $2 $(wc -c < "$2") bytes"; }
g https://ndownloader.figshare.com/files/64706751 wbm-summary.csv.gz          # WBM energies + MP2020 corrections
g https://ndownloader.figshare.com/files/48241624 ppd-mp.pkl.gz               # Materials Project phase diagram (MP2020-corrected)
g https://ndownloader.figshare.com/files/53161835 wbm-init-structs.jsonl.gz   # WBM starting structures
# Not downloadable by one stable URL; get these yourself (see README.md, step 1):
#   alexandria_pbe.json.bz2    Alexandria PBE entries (ComputedStructureEntry JSON with data.energy_corrected)
#   mp-cse.json.gz, mp-energies.csv.gz   Materials Project computed structure entries and energy_above_hull
