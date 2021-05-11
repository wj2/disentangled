import tensorflow as tf
import pickle
import os
import itertools as it
import scipy.stats as sts
import numpy as np
import functools as ft
import re
import PIL.Image as pImage
import pandas as pd
import h5py

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
            m_attr = tf.keras.models.load_model(tf_path, compile=False)
            setattr(dummy_object, attr, m_attr)
        for attr, value in ndict.items():
            setattr(dummy_object, attr, value)
        dummy_object._compile()
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

def _contextual_binary_classification(x, plane=None, off=0, context=None,
                                      context_off=0):
    ret = (np.sum(plane*x, axis=1) - off > 0).astype(float)
    if context[0] is not None:
        mask = np.sum(context*x, axis=1) - context_off < 0
        ret[mask] = np.nan
    return ret

def generate_partition_functions(dim, offset_distribution=None, n_funcs=100,
                                 orth_vec=None, orth_off=None,
                                 orth_basis=False, contextual=False,
                                 smaller=False, context_offset=False):
    if orth_basis:
        orth_vecs = u.generate_orthonormal_basis(dim)
        seq_inds = np.arange(n_funcs, dtype=int) % dim
        plane_vec = orth_vecs[:, seq_inds].T
    elif orth_vec is not None:
        plane_vec = u.generate_orthonormal_vectors(orth_vec, n_funcs)
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
    offsets_context = np.zeros(n_funcs)
    if contextual:
        direction_c = np.random.randn(n_funcs, dim)
        norms_c = np.expand_dims(np.sqrt(np.sum(direction_c**2, axis=1)), 1)
        plane_vec_context = direction_c/norms_c
    else:
        plane_vec_context = (None,)*n_funcs
    if context_offset:
        offsets_context = offset_distribution.rvs(n_funcs)
        
    funcs = np.zeros(n_funcs, dtype=object)
    
    for i in range(n_funcs):
        funcs[i] = ft.partial(_contextual_binary_classification,
                              plane=plane_vec[i:i+1], off=offsets[i],
                              context=plane_vec_context[i:i+1],
                              context_off=offsets_context[i])
    return funcs, plane_vec, offsets
    
class HalfMultidimensionalNormal(object):

    def __init__(self, *args, set_partition=None, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.distr = sts.multivariate_normal(*args, **kwargs)
        self.dim = self.distr.dim
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

    @classmethod
    def partition(cls, norm):
        m = norm.mean
        s = norm.cov
        return cls(m, s)
    
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
        new.partition = -self.partition
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
                               gd=None, seed_str='genout_{}.tfmod',
                               save_tf_models=True, save_args=None):
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

    if gd is not None:
        gd_file = seed_str.format('gd')
        pickle.dump(gd, open(os.path.join(folder, gd_file), 'wb'))

    if save_tf_models:
        manifest = (dg_file, models_file, history_file)
    else:
        manifest = ()
    if save_args is not None:
        args_file = seed_str.format('args')
        pickle.dump(save_args, open(os.path.join(folder, args_file), 'wb'))
    manifest = manifest + (p_file, c_file)
    if lr is not None:
        manifest = manifest + (lr_file,)
    if sc is not None:
        manifest = manifest + (sc_file,)
    if gd is not None:
        manifest = manifest + (gd_file,)
    if save_args is not None:
        manifest = manifest + (args_file,)
    pickle.dump(manifest, open(os.path.join(folder, 'manifest.pkl'), 'wb'))
    
def load_generalization_output(folder, manifest='manifest.pkl',
                               dg_type=None, analysis_only=False,
                               model_type=None, model_type_arr=None,
                               key_template='.*_([a-z]+)\.tfmod',
                               skip_gd=True):
    fnames = pickle.load(open(os.path.join(folder, manifest), 'rb'))
    fnames_full = list(os.path.join(folder, x) for x in fnames)
    key_str = (re.match(key_template, fn).group(1) for fn in fnames)
    fnames_dict = dict(zip(key_str, fnames_full))
    
    dg_file = fnames_dict.get('dg')
    models_file = fnames_dict.get('models')
    history_file = fnames_dict.get('histories')
    sc_file = fnames_dict.get('sc')
    ld_file = fnames_dict.get('lr')
    args_file = fnames_dict.get('args')
    p_file = fnames_dict.get('p')
    c_file = fnames_dict.get('c')
    gd_file = fnames_dict.get('gd')
    if models_file is None:
        analysis_only = True
        
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
        try:
            ld = pickle.load(open(ld_file, 'rb'))
        except ModuleNotFoundError:
            ld = None
    else:
        ld = None
    if sc_file is not None:
        sc = pickle.load(open(sc_file, 'rb'))
    else:
        sc = None
    if gd_file is not None and not skip_gd:
        gd = pickle.load(open(gd_file, 'rb'))
    else:
        gd = None
    if args_file is not None:
        args = pickle.load(open(args_file, 'rb'))
    else:
        args = None
        
    return dg, models, th, p, c, ld, sc, gd, args

def _interpret_foldername(fl, td_pattern='-td([0-9]+)-',
                          ld_pattern='-ld([0-9]+)-',
                          bd_pattern='-bd([0-9.]+)-([0-9.]+)-([0-9]+)-',
                          orth_pattern='-orth-'):
    td_out= re.search(td_pattern, fl)
    if td_out is not None:
        td_out = int(td_out.group(1))

    ld_out = re.search(ld_pattern, fl)
    if ld_out is not None:
        ld_out = int(ld_out.group(1))

    bd_out = re.search(bd_pattern, fl)
    if bd_out is not None:
        bd_out = (float(bd_out.group(1)), float(bd_out.group(2)),
                  int(bd_out.group(3)))

    orth_out = re.search(orth_pattern, fl)
    if orth_out is not None:
        orth_out = True
    return dict(input_dimensions=td_out, latent_dimensions=ld_out,
                training_eg_args=bd_out, orthogonal_partitions=orth_out)

def load_full_run(folder, run_ind, merge_axis=1,
                  file_template='bvae-n_([0-9])_{run_ind}',
                  analysis_only=False, **kwargs):
    tomatch = file_template.format(run_ind=run_ind)
    fls = os.listdir(folder)
    targ_inds = []
    outs = []
    have_info = False
    info = None
    for fl in fls:
        x = re.match(tomatch, fl)
        if x is not None:
            ti = int(x.group(1))
            if not have_info:
                info = _interpret_foldername(fl)
            targ_inds.append(ti)
            fp = os.path.join(folder, fl)
            out = load_generalization_output(fp, analysis_only=analysis_only,
                                             **kwargs)
            outs.append(out)
    if len(outs) == 0:
        raise IOError('no files found')
    sort_inds = np.argsort(targ_inds)
    out_inds = []
    args_all = []
    for i, si in enumerate(sort_inds):
        out_inds.append(targ_inds[si])
        if i == 0:
            dg_all, models_all, th_all, p_all = outs[si][:4]
            c_all, ld_all, sc_all, gd_all, args = outs[si][4:]
            if args is not None:
                args_all.append(vars(args))
            else:
                args_all.append(args)
            if gd_all is None:
                ls_all = None
                dgs_all = None
                rs_all = None
            else:
                ls_all, dgs_all, rs_all = gd_all
                ls_all = np.expand_dims(ls_all, merge_axis)
                dgs_all = np.expand_dims(dgs_all, merge_axis)
            try:
                sc_all.shape
            except AttributeError:
                sc_all, _ = sc_all
        else:
            _, models, th, p, c, ld, sc, gd, args = outs[si]
            if args is not None:
                args_all.append(vars(args))
            else:
                args_all.append(args)
            if gd is None:
                ls, dgs, rs = None, None, None
            else:
                ls, dgs, rs = gd
                ls = np.expand_dims(ls, merge_axis)
                dgs = np.expand_dims(dgs, merge_axis)
            if not analysis_only:
                models_all = _concatenate_none((models_all, models),
                                            axis=merge_axis)
                if th_all is not None:
                    th_all = _concatenate_none((th_all, th), axis=merge_axis)
            try:
                sc.shape
            except AttributeError:
                sc, _ = sc
            p_all = _concatenate_none((p_all, p), axis=merge_axis)
            ch_all = _concatenate_none((c_all, c), axis=merge_axis)
            ld_all = _concatenate_none((ld_all, ld), axis=merge_axis)
            sc_all = _concatenate_none((sc_all, sc), axis=merge_axis)
            ls_all = _concatenate_none((ls_all, ls), axis=merge_axis)
            dgs_all = _concatenate_none((dgs_all, dgs), axis=merge_axis)
            rs_all = _concatenate_none((rs_all, rs), axis=merge_axis)
            
    data = (out_inds, dg_all, models_all, th_all, p_all, c_all, ld_all, sc_all,
            (ls_all, dgs_all, rs_all))
    info['args'] = args_all
    return data, info

def _concatenate_none(arrs, axis=0):
    if np.any(list(arr is None for arr in arrs)):
        out = None
    else:
        out = np.concatenate(arrs, axis=axis)
    return out

def get_circle_pts(n, inp_dim, r=1):
    angs = np.linspace(0, 2*np.pi, n)
    pts = np.stack((np.cos(angs), np.sin(angs),) +
                   (np.zeros_like(angs),)*(inp_dim - 2), axis=1)
    return r*pts

def _load_archive(archive, img_key, param_vals, param_names,
                  norm_params=False, img_size=(64, 64),
                  max_load=np.inf, norm_pixels=False, convert_color=True):
    images = archive[img_key]
    images_list = []
    need_resize = not np.all(np.array(img_size) == images.shape[1:3])
    for i, img in enumerate(images):
        if i >= max_load:
            break
        if need_resize:
            im = pImage.fromarray(img)
            im = im.resize(img_size)
            img = np.asarray(im)
        if convert_color:
            img = pImage.fromarray(img).convert(mode='RGB')
            img = np.asarray(img)
        if len(images.shape) == 3:
            img = np.expand_dims(img, -1)
        if norm_pixels:
            img = img/255.
        images_list.append(img)
    if max_load == np.inf:
        max_load = archive[param_vals].shape[0]
    vals = archive[param_vals][:max_load]
    if norm_params:
        vals = u.demean_unit_std(vals)
    d = {param_names[i]:vals[:, i] for i in range(vals.shape[1])}
    d['images'] = images_list
    data = pd.DataFrame(data=d)
    return data    

twod_shape_params = ('color', 'shape', 'scale','orientation', 'x_pos', 'y_pos')
def load_2d_shapes(file_, img_key='imgs', param_vals='latents_values',
                   param_names=twod_shape_params, img_size=(64, 64),
                   **kwargs):
    archive = np.load(file_)
    return _load_archive(archive, img_key, param_vals, param_names,
                         img_size=img_size, **kwargs)

threed_shape_params = ('floor_hue', 'wall_hue', 'object_hue','scale', 'shape',
                       'orientation')
def load_3d_shapes(file_, img_key='images', param_vals='labels',
                   param_names=threed_shape_params, img_size=(64, 64),
                   norm_pixels=True, **kwargs):
    archive = h5py.File(file_, 'r')
    return _load_archive(archive, img_key, param_vals, param_names,
                         img_size=img_size, norm_pixels=norm_pixels,
                         **kwargs)    

chair_temp = 'image_([0-9]{3})_p([0-9]{3})_t([0-9]{3})_r([0-9]{3})\.png'
def load_chair_images(folder, file_template=chair_temp, mid_folder='renders',
                      img_size=(64, 64), max_load=np.inf, norm_pixels=True,
                      norm_params=False, grayscale=False, filter_edges=None,
                      edge_keys=('rotation',), read_specific_chairs=None):
    subfolders = filter(lambda x: os.path.isdir(os.path.join(folder, x)),
                                                os.listdir(folder))
    names = []
    nums = []
    pitchs = []
    rots = []
    dists = []
    ids = []
    imgs = []
    loaded = 0
    for sfl in subfolders:
        if ((read_specific_chairs is not None and sfl in read_specific_chairs)
            or read_specific_chairs is None):
            p = os.path.join(folder, sfl, mid_folder)
            img_fls = os.listdir(p)
            for ifl in img_fls:
                m = re.match(file_template, ifl)
                if m is not None:
                    names.append(ifl)
                    nums.append(int(m.group(1)))
                    pitchs.append(int(m.group(2)))
                    rots.append(int(m.group(3)))
                    dists.append(int(m.group(4)))
                    ids.append(sfl)
                    img = pImage.open(os.path.join(p, ifl))
                    if grayscale:
                        img = img.convert('L')
                    img_rs = img.resize(img_size)
                    img_arr = np.asarray(img_rs)
                    if grayscale:
                        img_arr = np.expand_dims(img_arr, -1)
                    if norm_pixels:
                        img_arr = img_arr/255
                    imgs.append(img_arr)
                    img.close()
                    
                    loaded = loaded + 1
                if loaded >= max_load:
                    break
        if loaded >= max_load:
            break
    if norm_params:
        pitchs = u.demean_unit_std(np.array(pitchs))
        rots = u.demean_unit_std(np.array(rots))
        dists = u.demean_unit_std(np.array(dists))
    d = {'names':names, 'chair_id':ids, 'img_nums':nums, 'pitch':pitchs,
         'rotation':rots, 'distances':dists, 'images':imgs}
    data = pd.DataFrame(data=d)
    if filter_edges is not None:
        mask = np.zeros(len(pitchs), dtype=bool)
        for fk in edge_keys:
            min_fk = np.min(d[fk])
            max_fk = np.max(d[fk])
            extent = max_fk - min_fk
            sub = filter_edges*extent/2
            mask_fk = np.logical_or(d[fk] < min_fk + sub,
                                    d[fk] > max_fk - sub)
            mask = np.logical_or(mask, mask_fk)
        data = data[np.logical_not(mask)]
    return data
