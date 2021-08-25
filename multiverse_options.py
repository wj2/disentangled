import itertools as it
import pickle
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import disentangled.aux as da

def make_common_multi_dict(multi_dict=None):
    if multi_dict is None:
        multi_dict = {}
    multi_dict['input_dims'] = (2, 5, 8)
    multi_dict['latent_dims'] = (10, 50, 100)
    multi_dict['n_reps'] = (2,)
    multi_dict['n_train_bounds'] = ((3, 4),)
    multi_dict['n_train_diffs'] = (2,)
    multi_dict['dg_dim'] = (500,)
    multi_dict['use_prf_dg'] = (True,)

    multi_dict['use_tanh'] = (True, False)
    multi_dict['layer_spec'] = ((250, 100, 100), (250, 150, 100, 100),
                                (250, 150, 100, 100, 100))
    multi_dict['source_distr'] = ('normal', 'uniform')
    return multi_dict

def make_fd_multi_dict():
    multi_dict = {}
    ps = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
    multi_dict['partitions'] = tuple((x,) for x in ps)
    multi_dict['use_orthog_partitions'] = (False,)
    multi_dict['offset_distr_var'] = (.4,)
    multi_dict['contextual_partitions'] = (True,)
    multi_dict['no_autoencoder'] = (True,)
    multi_dict['nan_salt'] = (0,)
    multi_dict = make_common_multi_dict(multi_dict)
    return multi_dict

def make_bvae_multi_dict():
    multi_dict = {}
    bs = (0, .5, 1, 1.5, 2, 2.5, 3, 5, 10, 15, 20)
    multi_dict['betas'] = tuple((x,) for x in bs)
    multi_dict = make_common_multi_dict(multi_dict)
    return multi_dict

def get_n_options(d):
    p = it.product(*list(d.values()))
    return len(list(p))

def save_option_dicts(d, folder, file_base='mvo_{}.pkl',
                      manifest_name='manifest.pkl'):
    entries = list(d.items())
    keys = list(e[0] for e in entries)
    values = list(e[1] for e in entries)
    p = it.product(*values)
    path = os.path.join(folder, file_base)
    dict_dict = {}
    for i, pi in enumerate(p):
        pi_dict = dict(list(zip(keys, pi)))
        pickle.dump(pi_dict, open(path.format(i), 'wb'))
        dict_dict[i] = pi_dict
    manifest_path = os.path.join(folder, manifest_name)
    pickle.dump(dict_dict, open(manifest_path, 'wb'))
    return dict_dict

def _find_fl(fls, fl):
    candidates = []
    inds = []
    for i, fli in enumerate(fls):
        m = re.match(fl, fli)
        if m is not None:
            ind = int(m.group(1))
            candidates.append(fli)
            inds.append(ind)
    use_ind = np.argmax(inds)
    return candidates[use_ind]

def load_multiverse(folder, manifests, f_template='{abbrev}-mv_{ind}_([0-9]+)',
                    min_regr=0):
    fls = os.listdir(folder)
    for abbrev, v in manifests.items():
        ind_dict = pickle.load(open(v, 'rb'))
        for i, (k, args) in enumerate(ind_dict.items()):
            fl = f_template.format(abbrev=abbrev, ind=k)
            fl_use = _find_fl(fls, fl)
            full_folder = os.path.join(folder, fl_use)
            dat = da.load_generalization_output(full_folder)
            _, _, _, pk, _, _, lk, _, args_ns = dat
            train_egs = np.logspace(*(args_ns.n_train_bounds
                                      + (args_ns.n_train_diffs,)))
            print(args_ns)
            for j, te in enumerate(train_egs):
                args['train_eg'] = te
                args['class_std'] = np.mean(pk[j, ..., 0])
                args['class_gen'] = np.mean(pk[j, ..., 1])
                args['regr_gen'] = np.max((np.mean(lk[0][j]), min_regr))
                if i == 0 and j == 0:
                    args_all = {k:[] for k in args.keys()}
                for k in args.keys():
                    ak = args[k]
                    if k == 'partitions' or k == 'betas':
                        ak = ak[0]
                    args_all[k].append(ak)
    df = pd.DataFrame(data=args_all)            
    return df

def plot_multiverse_split(data, plot_x, plot_y, split, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    split_cats = np.unique(data[split])
    for i, sc in enumerate(split_cats):
        data_i = data[data[split] == sc]
        plot_multiverse(data_i, plot_x, plot_y, ax=ax)
    return ax

def plot_multiverse(data, plot_x, plot_y, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    l = ax.plot(data[plot_x], data[plot_y], 'o')
    col = l[0].get_color()
    xs_u = np.unique(data[plot_x])
    ys_m = []
    for xi in xs_u:
        ys_m.append(np.mean(data[data[plot_x] == xi][plot_y]))
    ax.plot(xs_u, ys_m, color=col)

def generate_and_save_dicts(folder, file_base='mvo_{}.pkl'):
    fd = save_option_dicts(make_fd_multi_dict(), folder, 'fd-' + file_base,
                           manifest_name='fd_manifest.pkl')
    bd = save_option_dicts(make_bvae_multi_dict(), folder, 'bv-' + file_base,
                           manifest_name='bv_manifest.pkl')
    return fd, bd
