import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import os
import itertools as it
import scipy.stats as sts
import numpy as np
import functools as ft
import re

import general.utility as u

float_cast_center = lambda x: tf.cast(x, tf.float32) - .5
negloglik = lambda x, rv_x: -rv_x.log_prob(x)
dummy = lambda x, rv_x: tf.zeros_like(x)

class TFModel(object):
    
    def _save_wtf(self, path, tf_entries):
        paths = {}
        ndict = dict(**self.__dict__)
        for i, tfe in enumerate(tf_entries):
            x = ndict.pop(tfe)
            p, ext = os.path.splitext(path)
            path_i = p + '_{}'.format(i) + ext
            tf.keras.models.save_model(x, path_i)
            paths[tfe] = path_i
        ndict['__paths__'] = paths
        pickle.dump(ndict, open(path, 'wb'))

    @staticmethod
    def _load_model(dummy_object, path, use_new_head=True, skip=None):
        if use_new_head:
            new_folder, _ = os.path.split(path)
        ndict = pickle.load(open(path, 'rb'))
        tf_items = ndict.pop('__paths__')
        if skip is not None:
            list(tf_items.pop(skip_item) for skip_item in skip)
        for attr, tf_path in tf_items.items():
            if use_new_head:
                _, targ_file = os.path.split(tf_path)
                tf_path = os.path.join(new_folder, targ_file)
            m_attr = tf.keras.models.load_model(tf_path)
            setattr(dummy_object, attr, m_attr)
        for attr, value in ndict.items():
            setattr(dummy_object, attr, value)
        return dummy_object

class InputGenerator(object):

    def __init__(self, distribution, proc_func):
        self.source = distribution
        self.proc = proc_func

    def gen(self):
        while True:
            yield self.proc(self.source.rvs())

    def rvs(self, n):
        out = self.proc(self.source.rvs(n))
        return out
    
def _binary_classification(x, plane=None, off=0):
    return np.sum(plane*x, axis=1) - off > 0
    
def generate_partition_functions(dim, offset_distribution=None, n_funcs=100,
                                 orth_vec=None, orth_off=None):
    if orth_vec is not None:
        orth_vecs = u.get_orthogonal_basis(orth_vec)[:, 1:]
        rand_inds = np.random.choice(orth_vecs.shape[1], n_funcs)
        plane_vec = orth_vecs[:, rand_inds].T
    else:
        direction = np.random.randn(n_funcs, dim)
        norms = np.expand_dims(np.sqrt(np.sum(direction**2, axis=1)), 1)
        plane_vec = direction/norms
    if offset_distribution is not None:
        offsets = offset_distribution.rvs(n_funcs)
    else:
        offsets = np.zeros(n_funcs)
    if orth_off is not None:
        offsets = np.ones(n_funcs)*orth_off
    funcs = np.zeros(n_funcs, dtype=object)
    
    for i in range(n_funcs):
        funcs[i] = ft.partial(_binary_classification, plane=plane_vec[i:i+1],
                              off=offsets[i])
    return funcs, plane_vec, offsets
    
class HalfMultidimensionalNormal(object):

    def __init__(self, *args, set_partition=None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.distr = sts.multivariate_normal(*args, **kwargs)
        if set_partition is None:
            out = generate_partition_functions(len(self.distr.mean),
                                               n_funcs=1)
            pfs, vecs, offs = out
            set_partition = pfs[0]
            self.partition = vecs[0]
            self.offset = offs[0]
        else:
            self.partition = None
            self.offset = None
        self.partition_func = set_partition

    def rvs(self, rvs_shape):
        rvs = self.distr.rvs(rvs_shape)
        while not np.all(self.partition_func(rvs)):
            mask = np.logical_not(self.partition_func(rvs))
            sample = self.distr.rvs(rvs_shape)
            rvs[mask] = sample[mask]
        return rvs

    def flip(self):
        set_part = lambda x: np.logical_not(self.partition_func(x))
        new = HalfMultidimensionalNormal(*self.args, set_partition=set_part,
                                         **self.kwargs)
        new.partition = self.partition
        new.offset = self.offset
        return new

def save_object_arr(path, model_arr, save_method):
    path_arr = np.zeros_like(model_arr, dtype=object)
    inds = it.product(*(range(x) for x in model_arr.shape))
    p, ext = os.path.splitext(path)
    for ind in inds:
        ind_str = '{}'*len(ind)
        path_ind = p + ind_str.format(*ind) + ext
        m = model_arr[ind]
        save_method(path_ind, m)
        path_arr[ind] = path_ind
    pickle.dump(path_arr, open(path, 'wb'))
    
def save_models(path, model_arr):
    save_object_arr(path, model_arr, lambda p, m: m.save(p))

def save_histories(path, hist_arr):
    def hist_save_func(path_ind, hist):
        hist.model = None
        pickle.dump(hist, open(path_ind, 'wb'))
    save_object_arr(path, hist_arr, hist_save_func)

def load_objects(path, load_method, replace_head=True): 
    path_arr = pickle.load(open(path, 'rb'))
    if replace_head:
        replace_head_path, _ = os.path.split(path)
    model_arr = np.zeros_like(path_arr, dtype=object)
    inds = it.product(*(range(x) for x in model_arr.shape))
    
    for ind in inds:
        path_ind = path_arr[ind]
        if replace_head:
            _, targ_file = os.path.split(path_ind)
            path_ind = os.path.join(replace_head_path, targ_file)
        m = load_method(ind, path_ind)
        model_arr[ind] = m
    return model_arr
   
def load_histories(path):
    load_method = lambda ind, p: pickle.load(open(p, 'rb'))
    history_arr = load_objects(path, load_method)
    return history_arr

def load_models(path, model_type=None, model_type_arr=None, replace_head=True):
    if model_type is None and model_type_arr is None:
        raise IOError('one of model_type or model_type_arr must be specified')

    path_arr = pickle.load(open(path, 'rb'))
    if model_type_arr is None:
        model_type_arr = np.zeros_like(path_arr, dtype=object)
        model_type_arr[:] = model_type
    load_method = lambda ind, path_ind: model_type_arr[ind].load(path_ind)
    model_arr = load_objects(path, load_method, replace_head=replace_head)
    return model_arr

def save_generalization_output(folder, dg, models, th, p, c, lr=None, sc=None,
                               seed_str='genout_{}.tfmod', save_tf_models=True):
    os.mkdir(folder)
    path_base = os.path.join(folder, seed_str)

    if save_tf_models:
        dg_file = seed_str.format('dg')
        dg.save(os.path.join(folder, dg_file))

        models_file = seed_str.format('models')
        save_models(os.path.join(folder, models_file), models)

        history_file = seed_str.format('histories')
        save_histories(os.path.join(folder, history_file), th)

    p_file = seed_str.format('p')
    pickle.dump(p, open(os.path.join(folder, p_file), 'wb'))

    c_file = seed_str.format('c')
    pickle.dump(c, open(os.path.join(folder, c_file), 'wb'))

    if lr is not None:
        lr_file = seed_str.format('lr')
        pickle.dump(lr, open(os.path.join(folder, lr_file), 'wb'))

    if sc is not None:
        sc_file = seed_str.format('sc')
        pickle.dump(sc, open(os.path.join(folder, sc_file), 'wb'))

    if save_tf_models:
        manifest = (dg_file, models_file, history_file)
    else:
        manifest = ()
    manifest = manifest + (p_file, c_file)
    if lr is not None:
        manifest = manifest + (lr_file,)
    if sc is not None:
        manifest = manifest + (sc_file,)
    pickle.dump(manifest, open(os.path.join(folder, 'manifest.pkl'), 'wb'))
    
def load_generalization_output(folder, manifest='manifest.pkl',
                               dg_type=None, analysis_only=False,
                               model_type=None, model_type_arr=None):
    fnames = pickle.load(open(os.path.join(folder, manifest), 'rb'))
    fnames_full = list(os.path.join(folder, x) for x in fnames)
    if len(fnames_full) == 4:
        dg_file, models_file, p_file, c_file = fnames_full
        history_file = None
        ld_file, sc_file = None, None
    elif len(fnames_full) == 5:
        dg_file, models_file, history_file, p_file, c_file = fnames_full
        ld_file, sc_file = None, None
    else:
        dg_file, models_file, history_file, p_file, c_file = fnames_full[:-2]
        ld_file, sc_file = fnames_full[-2:]

    if analysis_only:
        dg = None
        models = None
        th = None
    else:
        dg = dg_type.load(dg_file)
        models = load_models(models_file, model_type=model_type,
                             model_type_arr=model_type_arr)
        if history_file is not None:
            th = load_histories(history_file)
        else:
            th = None
    p = pickle.load(open(p_file, 'rb'))
    c = pickle.load(open(c_file, 'rb'))
    if ld_file is not None:
        ld = pickle.load(open(ld_file, 'rb'))
    else:
        ld = None
    if sc_file is not None:
        sc = pickle.load(open(sc_file, 'rb'))
    else:
        sc = None

    return dg, models, th, p, c, ld, sc

def load_full_run(folder, run_ind, merge_axis=1,
                  file_template='bvae-n_([0-9])_{run_ind}',
                  analysis_only=False, **kwargs):
    tomatch = file_template.format(run_ind=run_ind)
    fls = os.listdir(folder)
    targ_inds = []
    outs = []
    for fl in fls:
        x = re.match(tomatch, fl)
        if x is not None:
            print(fl)
            ti = int(x.group(1))
            targ_inds.append(ti)
            fp = os.path.join(folder, fl)
            out = load_generalization_output(fp, analysis_only=analysis_only,
                                             **kwargs)
            outs.append(out)
    sort_inds = np.argsort(targ_inds)
    out_inds = []
    for i, si in enumerate(sort_inds):
        out_inds.append(targ_inds[si])
        if i == 0:
            dg_all, models_all, th_all, p_all, c_all, ld_all, sc_all = outs[si]
        else:
            _, models, th, p, c, ld, sc = outs[si]
            if not analysis_only:
                models_all = np.concatenate((models_all, models),
                                            axis=merge_axis)
                if th_all is not None:
                    th_all = np.concatenate((th_all, th), axis=merge_axis)
            p_all = np.concatenate((p_all, p), axis=merge_axis)
            ch_all = np.concatenate((c_all, c), axis=merge_axis)
            ld_all = np.concatenate((ld_all, ld), axis=merge_axis)
            sc_all = np.concatenate((sc_all, sc), axis=merge_axis)
    return dg_all, models_all, th_all, p_all, c_all, ld_all, sc_all
