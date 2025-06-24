
import numpy as np

def get_default_confs():
    conf = {}

    # defaults
    conf['input_dims'] = 5
    conf['dg_dim'] = 500
    conf['dg_layers'] = None
    conf['dg_noise'] = .2
    conf['dg_regweight'] = (0, .3)
    conf['dg_train_egs'] = 100000

    conf['latent_dim'] = 50
    conf['layer_spec'] = 50, 50, 50
    conf['batch_size'] = 100
    conf['n_train_bounds'] = (3, 4)
    conf['n_train_diffs'] = 2
    conf['n_reps'] = 5
    conf['no_autoencoder'] = True
    conf['model_epochs'] = 200

def make_fig2_dicts(folder=''):
    f2a = get_default_confs()
    pickle.dump(f2a, open(os.path.join(folder, 'f2e.pkl'), 'wb'))

def make_fig3_dicts(folder=''):
    offset_vars = (0, .2, .4)
    for i, ov in enumerate(offset_vars):
        f3b_i = get_default_confs()
        f3b_i['offset_distr_var'] = ov
    
