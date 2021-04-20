
import re
import os
import pandas as pd
import numpy as np
import pickle

import disentangled.aux as da

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

file_templ = '.*_(?P<ind>[0-9]+)_(?P<run>[0-9]+)'
def read_in_files_from_folder(folder, templ=file_templ):
    db = {'category':np.array([]), 'run':np.array([]),
          'ind':np.array([])}
    folders = os.listdir(folder)
    for fl in folders:
        m = re.match(file_templ, fl)
        if m is not None:
            run_ind = int(m.group('ind'))
            run_id = int(m.group('run'))
            mask = np.array(db['run']) == run_id
            fullpath = os.path.join(folder, fl)
            if np.any(db['ind'][mask] == run_ind):
                pass
            elif len(os.listdir(fullpath)) > 0:
                db = add_file_to_db(fullpath, run_ind, run_id,
                                    db, category=folder)
    return db

def add_file_to_db(fl, run_ind, run_id, db, mn='manifest.pkl',
                   args_patt='.*_args.tfmod', category=None):
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
    db = append_dicts(db, d_fl, single_entry=True)
    return db

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
