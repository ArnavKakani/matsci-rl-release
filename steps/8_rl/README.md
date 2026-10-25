# Step 8: RL fine-tuning of CrystalFormer

Repeats the audit on a generative model: CrystalFormer is fine-tuned with PPO against an E_hull reward, then its samples are judged.

## 1. Set up checkpoints

Clone [CrystalFormer](https://github.com/deepmodeling/CrystalFormer). Convert its pretrained checkpoint if your JAX is newer:

```bash
python convert_ckpt.py <epoch_044000.pkl> $MATSCI_SCRATCH/cf/ckpt
```

Put the ORB-v3 checkpoint at `$MATSCI_SCRATCH/cf/orb-v3-conservative-inf-mpa-20250404.ckpt`.

## 2. Patch your checkout

Each script takes the path to your checkout, `<CF>`:

```bash
python patch_train_ppo.py <CF>    # adds --reward ehull_def
python patch_mask.py <CF>         # optional; CF_BLOCK_RAD=1 gives radioactive-element samples the worst reward
python patch_degenerate.py <CF>   # guards against degenerate generated cells
```

## 3. Write the reward function

You must write `make_defended_ehull_reward_fn` yourself in `crystalformer/reinforce/reward.py`; it is not included in this kit.
The signature used by the patch is `make_defended_ehull_reward_fn(orb_calc, chgnet_calc, ref_data, clip_value)`, which returns
`(_, batch_reward_fn)`. The reward of a sample is the pessimistic (maximum) of the ORB-v3 and CHGNet E_hull against the
combined hull in `ref_data`, clipped to `clip_value`, with failed evaluations mapped to the worst reward, in the same sign
convention as the existing `ehull` reward. Copy the structure of that function.

## 4. Train

Step 1's `build_combined_reference.py` makes the combined hull. Then:

```bash
CF_SRC=<CF> bash train_rl.sh published myrun 0
CF_SRC=<CF> bash train_rl.sh defended myrun 0
```

Writes `cf/rl_<name>/`.

## 5. Judge

```bash
python eval_rl.py --run $MATSCI_SCRATCH/cf/rl_<name> --n 100 --every 5
```

Re-scores samples every 5 epochs with MatterSim and CHGNet against the aligned complete hull and prints claimed-vs-verified
stability per epoch. Writes `eval_rl.csv` in the run folder. Use several seeds, and the same seeds for both arms.
