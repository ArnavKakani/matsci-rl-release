"""Add an optional element mask to CrystalFormer's reward.py (env CF_BLOCK_RAD=1). usage: python patch_mask.py <CrystalFormer checkout>"""
import sys
p = sys.argv[1] + "/crystalformer/reinforce/reward.py"; s = open(p).read()
if "CF_BLOCK_RAD" not in s:
    helper = '''
# ---- added: optional element mask (env CF_BLOCK_RAD=1): samples containing radioactive elements (Z>=84, Tc, Pm)
# receive the worst (clipped) reward. Applied identically to the published and defended rewards.
def _mask_radioactive(structures, out, clip_value):
    if os.environ.get("CF_BLOCK_RAD") != "1":
        return out
    from pymatgen.core import Element
    bad = np.array([st is None or any(Element(str(e)).Z >= 84 or str(e) in ("Tc", "Pm") for e in st.composition.elements) for st in structures])
    out = np.array(out, dtype=float); out[bad] = clip_value
    return out
'''
    s = s.replace("from BatchRelaxer import BatchRelaxer\n", "from BatchRelaxer import BatchRelaxer\n" + helper, 1)
    # published reward: mask before device_put
    a = "        output = map_reward_fn(structures, energies)\n        output = jnp.array(output)\n"
    assert s.count(a) == 1, "published anchor"
    s = s.replace(a, "        output = map_reward_fn(structures, energies)\n        output = jnp.array(_mask_radioactive(structures, output, clip_value))\n")
    # defended reward
    b = "        out = np.clip(np.maximum(orb_eh, chg_eh), -clip_value, clip_value)\n"
    assert s.count(b) == 1, "defended anchor"
    s = s.replace(b, b + "        out = _mask_radioactive(structures, out, clip_value)\n")
    open(p, "w").write(s)
print("ok", s.count("_mask_radioactive("))
