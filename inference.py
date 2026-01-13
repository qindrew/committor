import jax
import jax.numpy as np
import flax.serialization as serialization
from train import CommittorNN_Dist_Lip, make_forward_eval  # reuse your definitions
import pysages

# --- reconstruct same model structure ---
indices = np.arange(50)
tri_idx1 = np.asarray([44, 47, 47, 6, 5, 47, 7, 5])
tri_idx2 = np.asarray([46, 46, 49, 49, 44, 49, 46, 7])

model = CommittorNN_Dist_Lip(indices=indices,
                             tri_idx1=tri_idx1,
                             tri_idx2=tri_idx2,
                             h1=16, h2=16, h3=8, out_dim=1, sig_k=3.0)

# --- initialize params to get structure ---
rng = jax.random.PRNGKey(0)
dummy_pos = np.zeros((1, 50, 3), dtype=np.float32)
params = model.init(rng, dummy_pos, training=False)

# --- load trained parameters ---
with open("distance_flax.params", "rb") as f:
    params = serialization.from_bytes(params, f.read())
params = jax.tree.map(
    lambda x: x.astype(np.float64) if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating) else x,
    params,
)
# --- create jitted inference function ---
forward_eval = make_forward_eval(model)

# --- example usage ---
# load new positions (B, N, 3)
from ase.io import read, write
pos = read('interp.xyz',index=0).get_positions()[None, :, :]
pos = np.asarray(pos, dtype=np.float64)

outputs = np.squeeze(forward_eval(params, pos))
print("Predicted committor values:", outputs)
