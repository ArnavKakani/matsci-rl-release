"""Convert a CrystalFormer checkpoint saved with an older JAX so a current JAX can load it (drops the old array 'aval' state).
usage: python convert_ckpt.py <epoch_044000.pkl> <out dir>      writes <out dir>/epoch_044000.pkl with {"params": ...}"""
import os, pickle, sys
import numpy as np
import jax


def _reconstruct_array(fun, args, arr_state, aval_state):
    v = fun(*args); v.__setstate__(arr_state); return np.asarray(v)


class U(pickle.Unpickler):
    def find_class(self, mod, name):
        return _reconstruct_array if name == "_reconstruct_array" else super().find_class(mod, name)


src, out = sys.argv[1], sys.argv[2]
d = U(open(src, "rb")).load()
p = jax.tree_util.tree_map(np.asarray, d["params"])
print("param leaves", len(jax.tree_util.tree_leaves(p)))
os.makedirs(out, exist_ok=True)
pickle.dump({"params": p}, open(os.path.join(out, os.path.basename(src)), "wb")); print("saved")
