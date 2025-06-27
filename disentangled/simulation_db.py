
import re
import os
import pandas as pd
import numpy as np
import pickle

import disentangled.auxiliary as da
import general.utility as u

def read_in_files(folders, pd_db=None):
    if pd_db is None:
        pd_db = {}
    else:
        pd_db = pd_db.to_dict()
    for fold in folders:
        db_dict = read_in_files_from_folder(fold)
        pd_db = append_dicts(pd_db, db_dict)
    db = pd.DataFrame.from_dict(pd_db)
    return db

file_templ = '{}.*_(?P<ind>[0-9]+)_(?P<run>[0-9]+)'
def read_in_files_from_folder(folder, templ=file_templ, pre_str='',
                              load_results=True):
    templ = templ.format(pre_str)

    db = {'category':np.array([]), 'run':np.array([]),
          'ind':np.array([])}
    folders = os.listdir(folder)
    for fl in folders:
        m = re.match(templ, fl)
        if m is not None:
            run_ind = int(m.group('ind'))
            run_id = int(m.group('run'))
            mask = np.array(db['run']) == run_id
            fullpath = os.path.join(folder, fl)
            if np.any(db['ind'][mask] == run_ind):
                pass
            elif len(os.listdir(fullpath)) > 0:
                db = add_file_to_db(fullpath, run_ind, run_id,
                                    db, category=folder,
                                    load_results=load_results)
    return db

def complex_field_func(db_field, func=np.mean, pre_ind=None):
    new_field = []
    for db_f in db_field:
        if pre_ind is not None:
            db_f = db_f[pre_ind]
        repl = np.zeros_like(db_f)
        for ind in u.make_array_ind_iterator(db_f.shape):
            repl[ind] = func(db_f[ind])
        new_field.append(repl)
    return np.array(new_field, dtype=object)

def add_file_to_db(fl, run_ind, run_id, db, mn='manifest.pkl',
                   args_patt='.*_args.tfmod', category=None,
                   load_results=True):
    d_fl = da._interpret_foldername(fl)
    d_fl['category'] = category
    d_fl['run_id'] = run_id
    d_fl['run_ind'] = run_ind
    man_path = os.path.join(fl, mn)
    if os.path.isfile(man_path):
        manifest = pickle.load(open(os.path.join(fl, mn), 'rb'))
        args_l = list(filter(lambda x: re.match(args_patt, x) is not None,
                             manifest))
        if len(args_l) > 0:
            args = pickle.load(open(os.path.join(fl, args_l[0]), 'rb'))
            d_fl.update(vars(args))
        if load_results:
            out = da.load_generalization_output(fl, analysis_only=True)
            _, _, _, cgen, _, _, lin, _, _ = out
            d_fl['class_gen'] = cgen
            d_fl['regr_gen'] = lin
    db = append_dicts(db, d_fl, single_entry=True)
    return db

def construct_matrix(db, instru_key, *dim_keys, instru_ind=None):
    if instru_ind is None:
        instru_shape = db[instru_key][0].shape
    else:
        instru_shape = db[instru_key][0][instru_ind].shape
    n_entries = len(db[instru_key])
    dk_sizes = np.zeros(len(dim_keys), dtype=int)
    uk_vals = []
    for i, dk_i in enumerate(dim_keys):
        uk_i = np.unique(db[dk_i])
        dk_sizes[i] = len(uk_i)
        uk_vals.append(uk_i)
    out_mat = np.zeros(tuple(dk_sizes) + instru_shape)
    for ind in u.make_array_ind_iterator(dk_sizes):
        masks = np.zeros((len(dim_keys), n_entries))
        for i, dk_i in enumerate(dim_keys):
            val = uk_vals[i][ind[i]]
            mask_i = db[dk_i] == val
            masks[i] = mask_i
        full_mask = np.product(masks, axis=0, dtype=bool)
        if np.sum(full_mask) == 0:
            out_mat[ind] = np.nan
        else:
            if instru_ind is None:
                insert = db[instru_key][full_mask][0]
            else:
                insert = db[instru_key][full_mask][0][instru_ind]
            out_mat[ind] = insert
    return uk_vals, out_mat
        
    

def append_dicts(dlong, db, ref_field='run', single_entry=False):
    dl_set = set(dlong.keys())
    db_set = set(db.keys())
    dl_only = list(dl_set.difference(db))
    for dlo in dl_only:
        db[dlo] = np.nan
    db_only = db_set.difference(dl_set)
    if ref_field in dlong.keys():
        n_entries = len(dlong[ref_field])
    else:
        n_entries = 0
    for dbo in db_only:
        dlong[dbo] = np.ones(n_entries, dtype=object)*np.nan
    for k, v in db.items():
        if single_entry:
            v_arr = np.empty(1, dtype=object)
            v_arr[0] = v
        else:
            v_arr = v
        dlong[k] = np.append(dlong[k], v_arr, axis=0)
    return dlong

"""
work to connect back to ML results
come up with some kind of task related to Salzmann task
use standard dataset and show disentangling emerges from 
  classification



"""
