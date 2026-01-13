from jax import numpy as np
from jax import jit

def cartesian(idx1, idx2):
    return np.stack(np.broadcast_arrays(idx1[:, None], idx2[None, :]), axis=-1).reshape(-1, 2)

from ase.io import read, write
pos = read('A.xyz', index=0).get_positions()
pos = np.stack([pos, pos], axis=0)
print(pos.shape)

block1 = [np.array([6,44,47]), np.array([45,46,48,49])]
block2 = [np.array([5]), np.array([6,44,47])]
block3 = [np.array([5]), np.array([7])]
blocks = [block1, block2, block3]
blocks = [cartesian(block[0],block[1]) for block in blocks]

def get_block_dist(pos, block):
    return np.sort(np.linalg.norm(pos[:,block[:,0]] - pos[:,block[:,1]], axis=-1), axis=-1)
@jit
def get_x(pos, blocks):
    return np.concatenate([get_block_dist(pos, block) for block in blocks], axis=-1)

print(get_x(pos,blocks).shape)
