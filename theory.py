
import numpy as np
import scipy.stats as sts
import scipy.optimize as sio
import scipy.special as sps
import functools as ft
import itertools as it
import sklearn.decomposition as skd
import sklearn.cross_decomposition as skcd

import disentangled.aux as da

def generate_2d_rep(n_pts, stds, rows=1, std_factor=1):
    vert_std, horiz_std = stds
    columns = int(n_pts/rows)
    horiz_extent = std_factor*horiz_std*columns
    vert_extent = std_factor*vert_std*rows
    vert_cents = np.linspace(-vert_extent, vert_extent, rows)        
    horiz_cents = np.linspace(-horiz_extent, horiz_extent, columns)
    if rows == 1:
        vert_cents = [0]
    if columns == 1:
        horiz_cents = [0]
    centers = it.product(horiz_cents, vert_cents)
    dkl = 0
    std_vec = np.array([horiz_std, vert_std])
    for c in centers:
        c = np.array(c)
        kd = -np.sum(1 + np.log(std_vec**2) - c**2 - std_vec**2)
        dkl = dkl + kd
    return kd
        
def optimize_2d_rep(n_pts, rows=1, eps=.00001, std_factor=1):
    std_rep = ft.partial(generate_2d_rep, n_pts, rows=rows,
                         std_factor=std_factor)
    x = sio.minimize(std_rep, (1., 1.), bounds=((eps, None), (eps, None)))
    return x

def pr_bound_diff(d, n=None):
    if n is None:
        num = sps.zeta(1 + 2/d)**2
        denom = sps.zeta(2 + 4/d)
    else:
        ns = np.arange(1, n + 1)
        num = np.sum(1/(ns**(1 + 2/d)))**2
        denom = np.sum(1/ns**(2 + 4/d))
    out = num/denom
    return out

def pr_bound(n=1000):
    ns = np.arange(1, n + 1)
    num = np.sum(1/(ns**1))**2
    denom = np.sum(1/ns**2)
    out = num/denom
    return out

def rd_bound(dims, parts, sigma=1):
    sig2 = sigma**2
    distortion = dims*sig2*(2*np.pi*sig2)**(1/(2*parts) - 1)
    rate = .5*(dims - dims/(2*parts))*np.log2(2*np.pi*sig2)
    return rate, distortion
    
def generate_binary_map(dims, n_parts, n_samps=1000, source_distrib=None,
                        ones=True, **kwargs):
    if source_distrib is None:
        source_distrib = sts.multivariate_normal(np.zeros(dims), 1)
    else:
        dims = source_distrib.dim
    out = da.generate_partition_functions(dims, n_funcs=n_parts, **kwargs)
    funcs, planes, _ = out
    samps = source_distrib.rvs(n_samps)
    targs = da.generate_target(samps, funcs)
    if ones:
        targs[targs == 0] = -1
    targ_scal = da.generate_scalar(samps, planes)
    return samps, targs, targ_scal, planes

def compute_partition_cov(dims, n_parts, **kwargs):
    samps, targs, targ_scal, planes = generate_binary_map(dims, n_parts,
                                                          **kwargs)
    big_targ = np.dot(targs.T, targs)/targs.shape[0]
    big_targ_scal = np.dot(targ_scal.T, targ_scal)/targ_scal.shape[0]
    planes_outer = np.dot(planes, planes.T)
    planes_outer[planes_outer > 1] = 1
    targ_theor = 1 - 4*np.arccos(planes_outer)/(np.pi*2)
    scal_theor = np.dot(planes, planes.T)
    return big_targ, big_targ_scal, targ_theor, scal_theor

def compute_partition_rank(dims, n_parts, **kwargs):
    samps, targs, targ_scal, _ = generate_binary_map(dims, n_parts,
                                                     **kwargs)
    p = skd.PCA()
    p.fit(targs)
    ps = skd.PCA()
    ps.fit(targ_scal)
    return p.explained_variance_, ps.explained_variance_

def compute_binary_diags(dims, n_parts, **kwargs):
    samps, targs, _, _ = generate_binary_map(dims, n_parts, **kwargs)
    p = skcd.PLSCanonical(n_components=min(dims, n_parts))
    trs = p.fit_transform(targs, samps)
    return trs
