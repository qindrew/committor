import os
# Respect external setting; uncomment the next line to force CPU:
# os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "0")  # float32 for speed

from typing import Any, Dict, Tuple, Iterable, Sequence
import numpy as onp
import jax
import jax.numpy as np
from flax import linen as nn
from flax.training import train_state
import optax

# ------------------------------
# Layers
# ------------------------------
class NormalizedLinear(nn.Module):
    in_features: int
    out_features: int

    @nn.compact
    def __call__(self, x: np.ndarray) -> np.ndarray:
        kernel = self.param(
            "kernel", nn.initializers.lecun_normal(), (self.out_features, self.in_features)
        )
        bias = self.param("bias", nn.initializers.zeros, (self.out_features,))
        ci = self.param("ci", nn.initializers.ones, ())  # trainable scalar

        absrowsum = np.sum(np.abs(kernel), axis=1, keepdims=True)
        scale = np.minimum(1.0, nn.softplus(ci) / (absrowsum + 1e-12))
        w_norm = kernel * scale
        return np.dot(x, w_norm.T) + bias

def cartesian(idx1, idx2):
    return np.stack(np.broadcast_arrays(idx1[:, None], idx2[None, :]), axis=-1).reshape(-1, 2)

def get_block_dist(pos, block):
    return np.sort(np.linalg.norm(pos[:,block[:,0]] - pos[:,block[:,1]], axis=-1), axis=-1)

def get_x(pos, blocks):
    return np.concatenate([get_block_dist(pos, block) for block in blocks], axis=-1)

class CommittorNN_Dist_Lip(nn.Module):
    indices: np.ndarray   # (M,)
    blocks: Sequence
    h1: int = 32
    h2: int = 16
    h3: int = 8
    out_dim: int = 1
    sig_k: float = 4.0

    @nn.compact
    def __call__(self, pos: np.ndarray, training: bool = False) -> np.ndarray:
        # pos: (B, N, 3)
        sig_shift1 = 1e-3*self.param("sig_shift1", nn.initializers.constant(0.0), ())
        sig_shift2 = 1e-3*self.param("sig_shift2", nn.initializers.constant(0.0), ())

        x = get_x(pos[:, self.indices, :], self.blocks)
        x = NormalizedLinear(x.shape[-1], self.h1)(x); x = np.tanh(x)
        x = NormalizedLinear(self.h1, self.h2)(x);       x = np.tanh(x)
        x = NormalizedLinear(self.h2, self.h3)(x);       x = np.tanh(x)
        x = NormalizedLinear(self.h3, self.out_dim)(x)              # (B,1)
        x = np.squeeze(x, axis=-1)                                  # (B,)
        q = 0.5*jax.nn.sigmoid(self.sig_k * (x-sig_shift1) ) + 0.5*jax.nn.sigmoid(self.sig_k * (x-sig_shift2) )
        if training:
            return q, x
        else:
            return x

class CommittorNN_Dist_Lip_old(nn.Module):
    indices: np.ndarray   # (M,)
    blocks: Sequence
    h1: int = 32
    h2: int = 16
    h3: int = 8
    out_dim: int = 1
    sig_k: float = 3.0

    @nn.compact
    def __call__(self, pos: np.ndarray, training: bool = False) -> np.ndarray:
        # pos: (B, N, 3)
        x = get_x(pos[:, self.indices, :], self.blocks)
        x = NormalizedLinear(x.shape[-1], self.h1)(x); x = np.tanh(x)
        x = NormalizedLinear(self.h1, self.h2)(x);	 x = np.tanh(x)
        x = NormalizedLinear(self.h2, self.h3)(x);	 x = np.tanh(x)
        x = NormalizedLinear(self.h3, self.out_dim)(x)              # (B,1)
        x = np.squeeze(x, axis=-1)                                  # (B,)
        return jax.nn.sigmoid(self.sig_k * x) if training else x

# ------------------------------
# Losses
# ------------------------------
@jax.jit
def lipschitz_loss_from_params(params: Dict[str, Any]) -> np.ndarray:
    prod = 1.0

    def walk(p):
        nonlocal prod
        if isinstance(p, dict):
            if "ci" in p:
                prod = prod * nn.softplus(p["ci"])  # multiply scalar
            for v in p.values():
                walk(v)

    walk(params)
    return prod

def make_forward_eval(model):
    @jax.jit
    def _forward_eval(params, pos_b):
        return model.apply(params, pos_b, training=False)
    return _forward_eval


if __name__ == "__main__":
    import MDAnalysis as mda
    import pysages
    from ase.io import read as ase_read
    import matplotlib.pyplot as plt
    import flax.serialization as serialization

    # masses shape (1, N, 1)
    atoms0 = ase_read('../A.xyz', index=0)
    masses = onp.asarray(atoms0.get_masses(), dtype=onp.float32)[None, :, None]

    # ----- Model config (indices / triplets)
    indices = np.arange(50)
    block1 = [np.array([6,44,47]), np.array([45,46,48,49])]
    block2 = [np.array([5]), np.array([6,44,47])]
    block3 = [np.array([5]), np.array([7])]
    block4 = [np.array([7]), np.array([45,46,48,49])]
    blocks = [block1, block2, block3, block4]
    blocks = tuple([cartesian(block[0],block[1]) for block in blocks])

    model = CommittorNN_Dist_Lip(indices=indices, blocks=blocks, h1=64, h2=32, h3=16, out_dim=1, sig_k=4.0)

    forward_eval = make_forward_eval(model)
    rng = jax.random.PRNGKey(0)
    dummy_pos = np.zeros((1, 50, 3), dtype=np.float32)
    params = model.init(rng, dummy_pos, training=False)

    # --- load trained parameters ---
    with open("distance_flax.params", "rb") as f:
        params = serialization.from_bytes(params, f.read())
    def soft(x):
        return np.log(1+np.exp(x))
    print(soft(params["params"]["NormalizedLinear_0"]['ci']))
    print(soft(params["params"]["NormalizedLinear_1"]['ci']))
    print(soft(params["params"]["NormalizedLinear_2"]['ci']))
    print(soft(params["params"]["NormalizedLinear_3"]['ci']))
    print(params["params"]["sig_shift1"])
    print(params["params"]["sig_shift2"])
