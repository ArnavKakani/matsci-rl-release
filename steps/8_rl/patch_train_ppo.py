"""Add the --reward ehull_def option to CrystalFormer's train_ppo.py. usage: python patch_train_ppo.py <CrystalFormer checkout>
NOTE: it imports make_defended_ehull_reward_fn from crystalformer/reinforce/reward.py, which you must write yourself (see README.md)."""
import sys
p = sys.argv[1] + "/crystalformer/cli/train_ppo.py"; s = open(p).read()
if "ehull_def" not in s:
    s = s.replace("choices=['ehull', 'prop', 'dielectric', 'density']", "choices=['ehull', 'ehull_def', 'prop', 'dielectric', 'density']")
    anchor = '    elif args.reward == "prop":'
    add = '''    elif args.reward == "ehull_def":  # added: pessimistic ORB/CHGNet reward vs combined hull
        import json, bz2
        from crystalformer.reinforce.reward import make_defended_ehull_reward_fn
        from chgnet.model.dynamics import CHGNetCalculator
        with bz2.open(args.convex_path) as fh:
            ref_data = json.loads(fh.read().decode("utf-8"))
        for entry in ref_data["entries"]:
            entry.pop("structure", None)
        chg = CHGNetCalculator(use_device="cuda")
        _, batch_reward_fn = make_defended_ehull_reward_fn(calc, chg, ref_data, clip_value=args.ehull_clip)

'''
    assert s.count(anchor) == 1 and "'ehull_def'" in s, "anchor/choices not found"
    s = s.replace(anchor, add + anchor); open(p, "w").write(s)
print("ok", s.count("ehull_def"))
