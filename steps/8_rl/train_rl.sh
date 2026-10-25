#!/bin/bash
# Fine-tune CrystalFormer with PPO against an E_hull reward.
# usage: bash train_rl.sh <published|defended> <run-name> <seed> [extra train_ppo args]
#   published : ORB-v3 E_hull vs the plain Alexandria hull   (--reward ehull)
#   defended  : pessimistic max(ORB-v3, CHGNet) E_hull vs the combined hull from step 1 (--reward ehull_def, needs patch_train_ppo.py)
# Set CF_SRC (your CrystalFormer checkout, patched), MATSCI_SCRATCH, and put the converted checkpoint at $MATSCI_SCRATCH/cf/ckpt/epoch_044000.pkl
# and the ORB-v3 checkpoint at $MATSCI_SCRATCH/cf/orb-v3-conservative-inf-mpa-20250404.ckpt.
# Set CF_BLOCK_RAD=1 to give samples with radioactive elements (patch_mask.py) the worst reward.
set -e
ARM=$1; NAME=$2; SEED=$3; shift 3
S=${MATSCI_SCRATCH:?set MATSCI_SCRATCH}; CF=$S/cf
case $ARM in
  published) REWARD=ehull;     HULL=$S/data/alex_hull.json.bz2 ;;
  defended)  REWARD=ehull_def; HULL=$S/data/combined_ref_aligned.json.bz2 ;;
  *) echo "arm must be published or defended"; exit 2 ;;
esac
mkdir -p "$CF" && cd "$CF"
PYTHONPATH=${CF_SRC:?set CF_SRC to your patched CrystalFormer checkout} XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 \
python -m crystalformer.cli.train_ppo --folder "./rl_$NAME/" --restore_path "$CF/ckpt/epoch_044000.pkl" \
  --reward $REWARD --convex_path "$HULL" \
  --mlff_model orb-v3-conservative-inf-mpa --mlff_path "$CF/orb-v3-conservative-inf-mpa-20250404.ckpt" \
  --lr 1e-05 --dropout_rate 0.0 --K 40 --batchsize 100 --epochs 60 --num_io_process 16 --seed "$SEED" "$@"
