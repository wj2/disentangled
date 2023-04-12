import numpy as np
import scipy.stats as sts

import tensorflow as tf

import functools as ft
import itertools as it

tfk = tf.keras
tfkl = tf.keras.layers

def _task_func(x, vec=None, off=0):
    val = x @ vec.T + off
    cat = val > 0
    return cat

def make_simple_tasks(true_learn_dim, n_partitions, offset_var=.4):
    offs = sts.norm(0, offset_var).rvs((n_partitions))
    vecs = sts.norm(0, 1).rvs((n_partitions, true_learn_dim))
    vecs = vecs/np.linalg.norm(vecs, axis=1, keepdims=True)

    p_funcs = []
    for i in range(n_partitions):
        pf_i = ft.partial(_task_func, vec=vecs[i:i+1], off=offs[i:i+1])
        p_funcs.append(pf_i)
    return p_funcs, vecs, offs

def make_simple_model(inp_dim, hl_widths, rep_dim, out_dim,
                      layer_type=tfkl.Dense, act_func=None,
                      rep_act=None, out_act=tf.nn.sigmoid):
    layers = []
    inp_layer = tfkl.InputLayer(input_shape=(inp_dim,))
    layers.append(inp_layer)

    for wid in hl_widths:
        l_w = layer_type(wid, activation=act_func)
        layers.append(l_w)

    layers.append(layer_type(rep_dim, activation=rep_act))
    rep_model = tfk.Sequential(layers)

    layers.append(layer_type(out_dim, activation=out_act))
    out_model = tfk.Sequential(layers)
    return rep_model, out_model

def make_data_dict(fdg, n_training_samples=10**4, n_val_samples=10**3,
                   n_grid_sides=30, grid_range=2, off_grid=0):
    data_dict = {}

    lv_tr, irs_tr = fdg.sample_reps(n_training_samples)
    data_dict['training'] = (irs_tr, lv_tr)

    lv_val, irs_val = fdg.sample_reps(n_val_samples)
    data_dict['validation'] = (irs_val, lv_val)

    n_samps = n_grid_sides**2
    lv_grid, _ = fdg.sample_reps(n_samps)
    grid_pts = np.linspace(-grid_range, grid_range, n_grid_sides)
    lv_grid[:, :2] = np.array(list(it.product(grid_pts, repeat=2)))
    lv_grid[:, 2:] = off_grid
    irs_grid = fdg.get_representation(lv_grid)
    data_dict['grid'] = (irs_grid, lv_grid)
    return data_dict

class LinearDisentangler:

    def __init__(self, inp_dim, layers, rep_dim, underlying_dim, n_tasks,
                 offset_var=.4, expansion_model=None, **kwargs):

        self.inp_dim = inp_dim
        self.rep_dim = rep_dim
        self.out_dim = n_tasks
        self.underlying_dim = underlying_dim

        out = make_simple_tasks(underlying_dim, n_tasks,
                                offset_var=offset_var)
        self.p_funcs, self.p_vectors, self.p_offsets = out

        out = make_simple_model(inp_dim, layers, rep_dim, n_tasks)
        self.rep_model, self.out_model = out

        self.compiled = False
        self.expansion_model = expansion_model
        

    def get_representation(self, x):
        return self.rep_model(x)

    def get_output(self, x):
        return self.out_model(x)

    def get_target(self, lvs):
        out = np.concatenate(list(pf(lvs) for pf in self.p_funcs), axis=1)
        return out

    def _compile(self, optimizer=None,
                 loss=tf.losses.MeanSquaredError()):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)

        self.out_model.compile(optimizer, loss)
        self.compiled = True

    def _make_training_data(self, fdg, n_samps=10**4):
        lvs, inp_rs = fdg.sample_reps(n_samps)
        if self.expansion_model is not None:
            inp_rs = self.expansion_model.get_representation(inp_rs)
        return self.get_training_targets(inp_rs, lvs)

    def get_training_targets(self, inp_rs, lvs):
        targs = self.get_target(lvs)
        return inp_rs, targs
                
    def fit(self, fdg, n_training_samples=10**4, n_validation_samples=10**3,
            **kwargs):
        training_data = self._make_training_data(
            fdg, n_samps=n_training_samples,
        )
        validation_data = self._make_training_data(
            fdg,
            n_samps=n_validation_samples,
        )
        return self.fit_data(training_data, validation_data=validation_data,
                             **kwargs)
        
    def fit_data(self, training_data, validation_data=None, batch_size=200,
                 epochs=10, **kwargs):
        if not self.compiled:
            self._compile()

        out = self.out_model.fit(x=training_data[0], y=training_data[1],
                                 validation_data=validation_data,
                                 epochs=epochs, batch_size=batch_size,
                                 **kwargs)
        return out


