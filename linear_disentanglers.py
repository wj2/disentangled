import numpy as np
import scipy.stats as sts

import tensorflow as tf
import sklearn.linear_model as sklm
import sklearn.svm as skm

import functools as ft
import itertools as it

import general.utility as u

tfk = tf.keras
tfkl = tf.keras.layers


def _task_func(x, vec=None, off=0):
    val = x @ vec.T + off
    cat = val > 0
    return cat


def generate_random_task_samples(n_dims, n_tasks, n_samps=10000):
    ts = u.make_unit_vector(sts.norm(0, 1).rvs((n_tasks, n_dims)))
    samps = sts.norm(0, 1).rvs((n_samps, n_dims))
    out = np.sign(ts @ samps.T).T
    if len(out.shape) == 1:
        out = np.expand_dims(out, 1)
    return samps, out


def label_abstraction(n_dims, n_tasks, n_samps=10000, dec_dim=0, gen_dim=1,
                      gap=0, n_reps=20, fdg=None):
    out_discrete = np.zeros(n_reps)
    out_cont = np.zeros_like(out_discrete)
    out_dim = np.zeros_like(out_discrete)
    for i in range(n_reps):
        samps, targs = generate_random_task_samples(n_dims, n_tasks, n_samps)
        if fdg is not None:
            reps = fdg.get_representation(samps)
            trs, _, _, _ = np.linalg.lstsq(reps, targs, rcond=None)
            targs = reps @ trs
        mask_tr = samps[:, gen_dim] > gap/2
        mask_te = samps[:, gen_dim] < -gap/2
        out = compute_abstraction(targs, samps[:, dec_dim], mask_tr, mask_te)
        out_discrete[i], out_cont[i] = out
        out_dim[i] = u.participation_ratio(targs)
    return out_discrete, out_cont, out_dim


def empirical_task_dimensionality(n_dims, n_tasks, n_samps=10000):
    if n_tasks == 0:
        dim = 0
    else:
        out = generate_random_task_samples(n_dims, n_tasks, n_samps=n_samps)
        dim = u.participation_ratio(out)
    return dim


def task_dimensionality(n_dims, n_tasks, n_samps=1000):
    v1 = u.make_unit_vector(sts.norm(0, 1).rvs((n_samps, n_dims)))
    v2 = u.make_unit_vector(sts.norm(0, 1).rvs((n_samps, n_dims)))
    correls = np.sum(v1*v2, axis=1)
    c_ij = 1 - (2/np.pi)*np.arccos(correls)
    dim = n_tasks/(1 + (n_tasks - 1)*np.mean(c_ij**2))
    return dim


def ols_statistics(task_vecs, partition_dim=0, nov_dim=1, eps=1e-5):
    p_vec, n_vec = np.identity(task_vecs.shape[1])[:2]
    tv_outer = task_vecs @ task_vecs.T
    mask = np.identity(tv_outer.shape[0], dtype=bool)
    tv_outer[mask] = 1

    M = 1 - (2/np.pi)*np.arccos(tv_outer)
    P_A = 1 - (2/np.pi)*np.arccos(task_vecs @ p_vec)
    P_B = 1 - (2/np.pi)*np.arccos(n_vec @ p_vec)

    print((task_vecs @ n_vec).shape)
    print(task_vecs.shape, n_vec.shape)
    B = 1 - (2/np.pi)*np.arccos(task_vecs @ n_vec)
    print('shapes', M.shape, B.shape)
    w = np.linalg.inv(M) @ (B)
    # print('no partition', w_no)
    # print((P_A @ P_A.T).shape)
    # w = np.linalg.inv(M + P_A @ P_A.T) @ (B + P_B*P_A)
    return w


def make_simple_tasks(true_learn_dim, n_partitions, offset_var=0.4):
    offs = sts.norm(0, offset_var).rvs((n_partitions))
    vecs = sts.norm(0, 1).rvs((n_partitions, true_learn_dim))
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    p_funcs = []
    for i in range(n_partitions):
        pf_i = ft.partial(_task_func, vec=vecs[i : i + 1], off=offs[i : i + 1])
        p_funcs.append(pf_i)
    return p_funcs, vecs, offs


def _discrete_abstraction(reps, targ, mask_tr, mask_te):
    m = skm.LinearSVC()
    m.fit(reps[mask_tr], targ[mask_tr])
    score = m.score(reps[mask_te], targ[mask_te])
    return score


def _continuous_abstraction(reps, targ, mask_tr, mask_te):
    m = sklm.Ridge()
    m.fit(reps[mask_tr], targ[mask_tr])
    score = m.score(reps[mask_te], targ[mask_te])
    return score


def prediction_range(fdg, n_task_range, n_reps=10, **kwargs):
    rep_scores = np.zeros((len(n_task_range), n_reps, 2))
    targ_scores = np.zeros_like(rep_scores)
    for i, nt in enumerate(n_task_range):
        for j in range(n_reps):
            out = predict_abstraction(fdg, nt, **kwargs)
            rep_scores[i, j] = out[0]
            targ_scores[i, j] = out[1]
    return rep_scores, targ_scores


def _apply_tasks(lvs, tasks=None):
    return np.concatenate(list(t(lvs) for t in tasks), axis=1)


def predict_abstraction(
    fdg,
    n_tasks,
    n_samps=10000,
    gen_dim=1,
    dec_dim=0,
    dec_samps=2000,
    **task_kwargs
):
    dim = fdg.input_dim
    lvs_tr, inp_reps_tr = fdg.sample_reps(n_samps)

    if n_tasks > 0:
        tasks, vecs, _ = make_simple_tasks(dim, n_tasks, **task_kwargs)
        targ_func = ft.partial(_apply_tasks, tasks=tasks)
        targ = targ_func(lvs_tr)
        trs, resid, rank, s = np.linalg.lstsq(inp_reps_tr, targ, rcond=None)
    else:
        trs = np.identity(inp_reps_tr.shape[1])
        vecs = np.zeros((1, dim))
        targ_func = lambda x: inp_reps_te @ x

    lvs_te, inp_reps_te = fdg.sample_reps(dec_samps)
    lin_rep_te = inp_reps_te @ trs
    targ_rep_te = 2*(targ_func(lvs_te) - .5)
    nov_task_te = 2*((lvs_te[:, dec_dim] > 0) - .5)

    mask_tr = lvs_te[:, gen_dim] > 0
    mask_te = np.logical_not(mask_tr)

    trs_te, _, _, _ = np.linalg.lstsq(targ_rep_te,
                                      nov_task_te,
                                      rcond=None)
    
    print(np.mean(np.sign(targ_rep_te @ trs_te) == nov_task_te))
    pred_weights = ols_statistics(vecs, partition_dim=gen_dim, nov_dim=dec_dim)
    print(np.mean(np.sign(targ_rep_te @ pred_weights) == nov_task_te))
    print('ols', pred_weights)
    print('trs', trs_te)

    out_rep = compute_abstraction(
        lin_rep_te, lvs_te[:, dec_dim], mask_tr, mask_te
    )

    out_targ = compute_abstraction(
        targ_rep_te, lvs_te[:, dec_dim], mask_tr, mask_te,
    )

    scores = (out_rep, out_targ)
    return scores


def compute_abstraction(rep, label, mask_tr, mask_te):
    discrete_score = _discrete_abstraction(
        rep, label > 0, mask_tr, mask_te,
    )
    cont_score = _continuous_abstraction(
        rep, label, mask_tr, mask_te,
    )
    return discrete_score, cont_score


def make_simple_model(
    inp_dim,
    hl_widths,
    rep_dim,
    out_dim,
    layer_type=tfkl.Dense,
    act_func=None,
    rep_act=None,
    out_act=tf.nn.sigmoid,
):
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


def make_data_dict(
    fdg,
    n_training_samples=10**4,
    n_val_samples=10**3,
    n_grid_sides=30,
    grid_range=2,
    off_grid=0,
):
    data_dict = {}

    lv_tr, irs_tr = fdg.sample_reps(n_training_samples)
    data_dict["training"] = (irs_tr, lv_tr)

    lv_val, irs_val = fdg.sample_reps(n_val_samples)
    data_dict["validation"] = (irs_val, lv_val)

    n_samps = n_grid_sides**2
    lv_grid, _ = fdg.sample_reps(n_samps)
    grid_pts = np.linspace(-grid_range, grid_range, n_grid_sides)
    lv_grid[:, :2] = np.array(list(it.product(grid_pts, repeat=2)))
    lv_grid[:, 2:] = off_grid
    irs_grid = fdg.get_representation(lv_grid)
    data_dict["grid"] = (irs_grid, lv_grid)
    return data_dict


class LinearDisentangler:
    def __init__(
        self,
        inp_dim,
        layers,
        rep_dim,
        underlying_dim,
        n_tasks,
        offset_var=0.4,
        expansion_model=None,
        **kwargs
    ):
        self.inp_dim = inp_dim
        self.rep_dim = rep_dim
        self.out_dim = n_tasks
        self.underlying_dim = underlying_dim

        out = make_simple_tasks(underlying_dim, n_tasks, offset_var=offset_var)
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

    def _compile(self, optimizer=None, loss=tf.losses.MeanSquaredError()):
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

    def fit(
        self, fdg, n_training_samples=10**4, n_validation_samples=10**3, **kwargs
    ):
        training_data = self._make_training_data(
            fdg,
            n_samps=n_training_samples,
        )
        validation_data = self._make_training_data(
            fdg,
            n_samps=n_validation_samples,
        )
        return self.fit_data(training_data, validation_data=validation_data, **kwargs)

    def fit_data(
        self, training_data, validation_data=None, batch_size=200, epochs=10, **kwargs
    ):
        if not self.compiled:
            self._compile()

        out = self.out_model.fit(
            x=training_data[0],
            y=training_data[1],
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
        return out
