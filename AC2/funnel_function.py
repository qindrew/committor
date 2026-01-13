# funnel functions
from functools import partial

import jax.numpy as np
from jax import jit, grad
from jax.numpy import linalg

def distance(r, cell_size):
    diff = r[1:] - r[0]
    #diff = diff - np.round(diff / cell_size) * cell_size
    return np.linalg.norm(diff,axis=1)

def switch(r, r0, k,):
    return (1+np.exp(k*(r-r0)))**-1

def coordnum_energy(r, cell_size, r0s, ks, locations, c_mins, k2, idx = np.asarray([ [0,1],[0,2],[0,3],[1,2],[1,3],[2,3] ])):
    total = 0 #all O somewhat close to C
    c_min = c_mins[0]
    dist = distance(r[:5],cell_size)
    total += 0.5*(np.where(dist > c_min, dist - c_min, 0.0)**2).sum()

    c_min = c_mins[1] #Os are all close to each other
    rO = r[1:5]
    dists = np.linalg.norm( rO[idx[:,0]] - rO[idx[:,1]],axis=1 )
    total += 0.5*(np.where(dists > c_min, dists - c_min, 0.0)**2).sum()

    c_min = c_mins[2]
    dist = np.linalg.norm(r[0] - r[4])
    total += 2*np.where(dist < c_min, dist - c_min, 0.0)**2

    c_min = c_mins[3]
    dist = np.linalg.norm(r[5] - r[4])
    total += 0.5*np.where(dist < c_min, dist - c_min, 0.0)**2

    c_min = c_mins[4]
    dist = np.linalg.norm(r[1] - r[6])
    total += 2*np.where(dist < c_min, dist - c_min, 0.0)**2

    c_min = c_mins[5]
    dist = np.linalg.norm(r[2] - r[6])
    total += 2*np.where(dist < c_min, dist - c_min, 0.0)**2

    c_min = c_mins[6]
    dist = np.linalg.norm(r[1] - r[7])
    total += 2*np.where(dist < c_min, dist - c_min, 0.0)**2

    c_min = c_mins[6]
    dist = np.linalg.norm(r[1] - r[5])
    total += 2*np.where(dist < c_min, dist - c_min, 0.0)**2

    c_min = c_mins[6]
    dist = np.linalg.norm(r[2] - r[8])
    total += 2*np.where(dist < c_min, dist - c_min, 0.0)**2

    c_min = c_mins[6]
    dist = np.linalg.norm(r[3] - r[6])
    total += 2*np.where(dist < c_min, dist - c_min, 0.0)**2

    c_min = c_mins[6]
    dist = np.linalg.norm(r[3] - r[7])
    total += 2*np.where(dist < c_min, dist - c_min, 0.0)**2

#    c_min = c_mins[1] #O7 close to OH2
#    rO7=r[1]
#    rO=r[2:5]
#    rH=r[5:]
#    val = switch(np.linalg.norm(rO[:,None,:] - rH, axis=2), r0=1.6,k=8) #(3,4) coordination of each O
#    maxval = np.max(val,axis=0) #(4)
#    val = np.where(val < maxval, 0.0, val).sum(axis=1) #val is coordination number of each O. Each H only coordinates with one O due to maxval
#    dists = np.linalg.norm(rO7-rO,axis=1) #distance of o7 from all o
#    mindist = np.min(np.where(val > 1.95, dists, 4)) #distance from closest water
#    total += 0.5*np.where(mindist > c_min, mindist - c_min, 0.0)**2

#    c_min = c_mins[2] #2h2o close to each other
#    waters = rO[np.argsort(val)[-2:]] #coordinates of h2os (maximally coordinated Hs)
#    dist = np.linalg.norm(waters[1]-waters[0]) #distance between waters
#    total += 0.5*np.where(np.sort(val)[1] > 1.95, np.where(dist > c_min, dist-c_min, 0), 0)**2 #if second most coordinated O > 1.95
    #if dist too large condition nested within if 2 h2os condition
    return k2 * total


def intermediate_funnel(pos, ids, indexes, cell_size, r0s, ks, locations, c_mins, k2):
    r = pos[ids[indexes]]
    return coordnum_energy(r, cell_size, r0s, ks, locations, c_mins, k2)

def log_funnel():
    return 0.0

def external_funnel(data, indexes, cell_size, r0s, ks, locations, c_mins, k2):
    pos = data.positions[:, :3]
    ids = data.indices
    bias = grad(intermediate_funnel)(pos, ids, indexes, cell_size, r0s, ks, locations, c_mins, k2)
    proj = log_funnel()
    return bias, proj

def get_funnel_force(indexes, cell_size, r0s, ks, locations, c_mins, k2):
    funnel_force = partial(
        external_funnel,
        indexes=indexes, cell_size=cell_size, r0s=r0s, ks=ks,
        locations=locations, c_mins=c_mins, k2=k2,)
    return jit(funnel_force)
