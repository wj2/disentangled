import itertools as it
import pickle
import os

import numpy as np

def make_common_multi_dict(multi_dict=None):
    if multi_dict is None:
        multi_dict = {}
    multi_dict['input_dims'] = (2, 5, 8)
    multi_dict['latent_dims'] = (10, 50, 100)
    multi_dict['n_reps'] = (2,)
    multi_dict['n_train_bounds'] = ((3, 4),)
    multi_dict['n_train_diffs'] = (2,)
    multi_dict['dg_dim'] = (500,)

    multi_dict['use_tanh'] = (True, False)
    multi_dict['layer_spec'] = ((100, 50), (200, 100, 100),
                                (200, 100, 100, 50))
    multi_dict['train_dg'] = (True, False)
    multi_dict['source_distr'] = ('normal', 'uniform')
    return multi_dict

def make_fd_multi_dict():
    multi_dict = {}
    ps = (0, 2, 4, 6, 8, 10, 12, 14, 20)
    multi_dict['partitions'] = tuple((x,) for x in ps)
    multi_dict['use_orthog_partitions'] = (False,)
    multi_dict['offset_distr_var'] = (.4,)
    multi_dict['contextual_partitions'] = (True,)
    multi_dict['no_autoencoder'] = (False,)
    multi_dict['nan_salt'] = (0,)
    multi_dict = make_common_multi_dict(multi_dict)
    return multi_dict

def make_bvae_multi_dict():
    multi_dict = {}
    bs = (0, .5, 1, 1.5, 2, 2.5, 3, 5, 10)
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

def generate_and_save_dicts(folder, file_base='mvo_{}.pkl'):
    save_option_dicts(make_fd_multi_dict(), folder, 'fd-' + file_base,
                      manifest_name='fd_manifest.pkl')
    save_option_dicts(make_bvae_multi_dict(), folder, 'bv-' + file_base,
                      manifest_name='bv_manifest.pkl')
