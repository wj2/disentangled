import tensorflow as tf
import tensorflow_probability as tfp
import pickle
import os
import itertools as it
import scipy.stats as sts
import numpy as np

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
    def _load_model(dummy_object, path):
        ndict = pickle.load(open(path, 'rb'))
        tf_items = ndict.pop('__paths__')
        for attr, tf_path in tf_items.items():
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
        funcs[i] = lambda x: np.sum(plane_vec[i:i+1]*x, axis=1) - offsets[i] > 0
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

def load_objects(path, load_method): 
    path_arr = pickle.load(open(path, 'rb'))

    model_arr = np.zeros_like(path_arr, dtype=object)
    inds = it.product(*(range(x) for x in model_arr.shape))
    for ind in inds:
        path_ind = path_arr[ind]
        m = load_method(ind, path_ind)
        model_arr[ind] = m
    return model_arr
   
def load_history(path):
    load_method = lambda ind, p: pickle.load(open(p, 'rb'))
    history_arr = load_objects(path, load_method)
    return history_arr
    
def load_models(path, model_type=None, model_type_arr=None):
    if model_type is None and model_type_arr is None:
        raise IOError('one of model_type or model_type_arr must be specified')

    path_arr = pickle.load(open(path, 'rb'))
    if model_type_arr is None:
        model_type_arr = np.zeros_like(path_arr, dtype=object)
        model_type_arr[:] = model_type
    load_method = lambda ind, path_ind: model_type_arr[ind].load(path_ind)
    model_arr = load_objects(path, load_method)
    return model_arr

def save_generalization_output(folder, dg, models, th, p, c,
                               seed_str='genout_{}.tfmod'):
    os.mkdir(folder)
    path_base = os.path.join(folder, seed_str)

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

    manifest = (dg_file, models_file, p_file, c_file)
    pickle.dump(manifest, open(os.path.join(folder, 'manifest.pkl'), 'wb'))
    
def load_generalization_output(folder, manifest='manifest.pkl',
                               dg_type=None,
                               model_type=None, model_type_arr=None):
    fnames = pickle.load(open(os.path.join(folder, manifest), 'rb'))
    fnames_full = (os.path.join(folder, x) for x in fnames)
    dg_file, models_file, history_file, p_file, c_file = fnames_full

    dg = dg_type.load(dg_file)
    models = load_models(models_file, model_type=model_type,
                         model_type_arr=model_type_arr)
    th = load_histories(history_file)
    p = pickle.load(open(p_file, 'rb'))
    c = pickle.load(open(c_file, 'rb'))

    return dg, models, th, p, c
    
    
