import numpy as np
import scipy.stats as sts
import pickle
import os
import itertools as it
import tensorflow as tf
import scipy.linalg as spla
import scipy.special as ss

import sklearn.decomposition as skd
import sklearn.svm as skc
import sklearn.linear_model as sklm
import sklearn.pipeline as sklpipe
import sklearn.preprocessing as skp

import general.utility as u
import general.plotting as gpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import disentangled.aux as da
import disentangled.data_generation as dg
import disentangled.disentanglers as dd

def classifier_generalization(gen, vae, train_func=None, train_distrib=None,
                              test_distrib=None, test_func=None,
                              n_train_samples=5*10**3, n_test_samples=10**3,
                              classifier=skc.LinearSVC, kernel='linear',
                              n_iters=2, mean=True,
                              shuffle=False, use_orthogonal=True,
                              learn_lvs='ignore', balance_samples=False,
                              repl_mean=None, gp_task_ls=None,
                              **classifier_params):
    if gp_task_ls is not None:
        task_type = 'gp'
        kernel = 'rbf'
        classifier = skc.SVC
    else:
        task_type = 'linear'
    if train_func is None:
        lv_mask = np.ones(gen.input_dim, dtype=bool)
        if use_orthogonal and hasattr(train_distrib, 'partition'):
            orth_vec = train_distrib.partition
            orth_off = train_distrib.offset
            if learn_lvs == 'ignore':
                out = da.generate_partition_functions(gen.input_dim,
                                                      n_funcs=n_iters,
                                                      orth_vec=orth_vec,
                                                      orth_off=orth_off,
                                                      task_type=task_type,
                                                      length_scale=gp_task_ls)
            elif learn_lvs == 'trained':
                lv_mask = vae.learn_lvs
                out = da.generate_partition_functions(
                    sum(vae.learn_lvs), n_funcs=n_iters,
                    orth_vec=orth_vec[lv_mask], orth_off=orth_off,
                    task_type=task_type,
                    length_scale=gp_task_ls)
            elif learn_lvs == 'untrained':
                n_untrained = gen.input_dim - sum(vae.learn_lvs)
                lv_mask = np.logical_not(vae.learn_lvs)
                out = da.generate_partition_functions(
                    n_untrained, n_funcs=n_iters, orth_vec=orth_vec[lv_mask],
                    orth_off=orth_off,
                    task_type=task_type,
                    length_scale=gp_task_ls)
            else:
                raise IOError('{} is not an understood option for '
                              'learn_lvs'.format(learn_lvs))
        else:
            out = da.generate_partition_functions(gen.input_dim, n_funcs=n_iters,
                                                  task_type=task_type,
                                                  length_scale=gp_task_ls)
        train_func, _, _ = out
    # print(vae.learn_lvs)
    # print(lv_mask)
    if train_distrib is None:
        train_distrib = gen.source_distribution
    if test_distrib is None:
        test_distrib = train_distrib
    if test_func is None:
        test_func = train_func
    scores = np.zeros(n_iters)
    chances = np.zeros(n_iters)
    for i in range(n_iters):
        if balance_samples:
            candidates = train_distrib.rvs(n_train_samples*10)
            cats = np.squeeze(train_func[i](candidates[:, lv_mask]))
            cat1_samps = candidates[cats == 0]
            cat2_samps = candidates[cats == 1]
            max_iter = 1000
            curr_iter = 0
            while ((len(cat1_samps) < n_train_samples
                    or len(cat2_samps) < n_train_samples)
                   and curr_iter < max_iter):
                candidates = train_distrib.rvs(n_train_samples*10)
                cats = np.squeeze(train_func[i](candidates[:, lv_mask]))
                cat1_samps = np.concatenate((cat1_samps, candidates[cats == 0]))
                cat2_samps = np.concatenate((cat2_samps, candidates[cats == 1]))
                curr_iter += 1
            train_samples = np.concatenate((cat1_samps[:n_train_samples],
                                            cat2_samps[:n_train_samples]),
                                           axis=0)
        else:
            train_samples = train_distrib.rvs(n_train_samples)
        if repl_mean is not None:
            train_samples[:, repl_mean] = train_distrib.mean[repl_mean]
        train_labels = np.squeeze(train_func[i](train_samples[:, lv_mask]))
        inp_reps = gen.generator(train_samples)
        train_rep = vae.get_representation(inp_reps)
        # print('tr', train_rep)
        # print('tl', train_labels)
        if not np.any(np.isnan(train_rep)):
            c = classifier(max_iter=100000, **classifier_params)
            ops = [skp.StandardScaler()]
            if train_rep.shape[1] > 2000:
                p = skd.PCA(.95)
                ops.append(p)
            ops.append(c)
            pipe = sklpipe.make_pipeline(*ops)
            pipe.fit(train_rep, train_labels)
            test_samples = test_distrib.rvs(n_test_samples)
            if repl_mean is not None:
                test_samples[:, repl_mean] = test_distrib.mean[repl_mean]
            test_labels = test_func[i](test_samples[:, lv_mask])
            if shuffle:
                snp.random.shuffle(test_labels)

            test_rep = vae.get_representation(gen.generator(test_samples))
            scores[i] = pipe.score(test_rep, test_labels)
            chances[i] = .5
        else:
            scores[i] = np.nan
            chances[i] = np.nan
    if mean:
        scores = np.nanmean(scores)
        chances = np.nanmean(chances)
    return scores, chances

def lr_pts_dist(stim, reps, targ_dist, neighbor_rad, eps=.05, same_prob=True,
                **kwargs):
    if same_prob:
        stim_dim = stim.shape[1]
        train_cent = (targ_dist/2)*u.make_unit_vector(np.random.randn(stim_dim))
        test_cent = -train_cent
    else:
        train_ind = np.random.choice(stim.shape[0])
        train_cent = stim[train_ind]
        dists = u.euclidean_distance(train_cent, stim)
        test_candidates = np.abs(dists - targ_dist) < eps
        if np.sum(test_candidates) > 0:
            test_ind = np.random.choice(np.sum(test_candidates))
            test_cent = stim[test_candidates][test_ind]
        else:
            test_cent = np.nan
    out = lr_gen_neighbors(stim, reps, train_cent, test_cent,
                           neighbor_rad, **kwargs)
    return out

def lr_gen_neighbors(stim, reps, train_pt, test_pt, radius,
                     lr=sklm.LinearRegression, min_set_pts=50,
                     print_disjoint=False, **kwargs):
    lri = lr(**kwargs)
    train_dists = u.euclidean_distance(train_pt, stim)
    test_dists = u.euclidean_distance(test_pt, stim)
    train_test_dist = u.euclidean_distance(train_pt, test_pt)
    if print_disjoint and train_test_dist < radius:
        print('the training and test points are closer than the radius so '
              'the training and test sets will not be disjoint')
    train_mask = train_dists < radius
    test_mask = test_dists < radius
    if np.sum(train_mask) < min_set_pts or np.sum(test_mask) < min_set_pts:
        print('no pts', np.sum(train_mask), np.sum(test_mask))
        sc = np.nan
    else:
        stim_train = stim[train_mask]
        reps_train = reps[train_mask]
        lri.fit(reps_train, stim_train)
        stim_test = stim[test_mask]
        reps_test = reps[test_mask]
        sc = lri.score(reps_test, stim_test)
    return sc    

def train_multiple_bvae(dg_use, betas, layer_spec, n_reps=10, batch_size=32,
                        n_train_samps=10**6, epochs=5, hide_print=False,
                        input_dim=None):
    training_history = np.zeros((len(betas), n_reps), dtype=object)
    models = np.zeros_like(training_history, dtype=object)
    if input_dim is None:
        input_dim = dg.input_dim
    for i, beta in enumerate(betas):
        for j in range(n_reps):
            inp_set, train_set = dg_use.sample_reps(sample_size=n_train_samps)
            inp_eval_set, eval_set = dg_use.sample_reps()
            bvae = d.BetaVAE(dg.output_dim, layer_spec, input_dim, beta=beta)
            if hide_print:
                with u.HiddenPrints():
                    th = bvae.fit(train_set, eval_x=eval_set, epochs=epochs,
                                  batch_size=batch_size)
            else:
                th = bvae.fit(train_set, eval_x=eval_set, epochs=epochs,
                              batch_size=batch_size)
            training_history[i, j] = th
            models[i, j] = bvae
    return models, training_history

def train_multiple_models_dims(input_dims, *args, n_train_samps=10**5,
                               samps_list=False, parallel_within=False,
                               **kwargs):
    models = []
    ths = []
    for i, inp_d in enumerate(input_dims):
        if samps_list:
            nts = n_train_samps[i]
        m, th = train_multiple_models(*args, input_dim=inp_d,  n_train_samps=nts,
                                      **kwargs)
        models.append(m)
        ths.append(th)
    return np.array(models), np.array(ths)

def evaluate_multiple_models_dims(dg_use, models, *args, n_train_samples=10**3,
                                  **kwargs):
    ps, cs = [], []
    for m in models:
        if u.check_list(n_train_samples):
            p_m = []
            c_m = []
            for nts in n_train_samples:
                p, c = evaluate_multiple_models(dg_use, m, *args,
                                                n_train_samples=nts,
                                                balance_samples=True,
                                                **kwargs)
                p_m.append(p)
                c_m.append(c)
        else:
            p_m, c_m = evaluate_multiple_models(dg_use, m, *args, **kwargs)
        ps.append(p_m)
        cs.append(c_m)
    return np.array(ps), np.array(cs)

def train_multiple_models(dg_use, model_kinds, layer_spec, n_reps=10, batch_size=32,
                          n_train_samps=10**6, epochs=5, hide_print=False,
                          input_dim=None, use_mp=False, standard_loss=True,
                          val_set=True, save_and_discard=False, true_val=True,
                          save_templ='m_{}-{}.tfmod',
                          **kwargs):
    training_history = np.zeros((len(model_kinds), n_reps), dtype=object)
    models = np.zeros_like(training_history, dtype=object)
    if input_dim is None:
        input_dim = dg_use.input_dim
    for i, mk in enumerate(model_kinds):
        for j in range(n_reps):
            tf.keras.backend.clear_session()
            train_set = dg_use.sample_reps(sample_size=n_train_samps)
            if val_set:
                eval_set = dg_use.sample_reps(sample_size=n_train_samps)
            else:
                eval_set = None
            if true_val:
                true_eval_set = dg_use.sample_reps(sample_size=n_train_samps)
            else:
                true_eval_set = None
            m = mk(dg_use.output_dim, layer_spec, input_dim, **kwargs)
            th = m.fit_sets(train_set, eval_set=eval_set, epochs=epochs,
                            batch_size=batch_size,
                            standard_loss=standard_loss,
                            use_multiprocessing=use_mp,
                            verbose=not hide_print,
                            true_eval_set=true_eval_set)
            if save_and_discard:
                m.save(save_templ.format(i, j))
            else:
                training_history[i, j] = th
                models[i, j] = m
    return models, training_history


def evaluate_multiple_models(dg_use, models, train_func, test_distributions,
                             train_distributions=None, mean=True,
                             n_iters=2, **classifier_args):
    if mean:
        dim_add = (len(test_distributions),)
    else:
        dim_add = (len(test_distributions), n_iters)
    performance = np.zeros(models.shape + dim_add)
    chance = np.zeros_like(performance)
    if train_distributions is None:
        train_distributions = (None,)*len(test_distributions)
    for ind in u.make_array_ind_iterator(models.shape):
        bvae = models[ind]
        for k, td in enumerate(test_distributions):
            train_d_k = train_distributions[k]
            out = classifier_generalization(dg_use, bvae, train_func,
                                            test_distrib=td,
                                            train_distrib=train_d_k,
                                            mean=mean, n_iters=n_iters,
                                            **classifier_args)
            performance[ind + (k,)] = out[0]
            chance[ind + (k,)]= out[1]
    return performance, chance

def train_and_evaluate_models(dg_use, betas, layer_spec, train_func,
                              test_distributions, n_reps=10, hide_print=False,
                              n_train_samps=10**6, epochs=5, input_dim=None,
                              models=None, batch_size=32, **classifier_args):
    if models is None:
        models, th = train_multiple_bvae(dg_use, betas, layer_spec, n_reps=n_reps,
                                         n_train_samps=n_train_samps,
                                         batch_size=batch_size, epochs=epochs,
                                         hide_print=hide_print,
                                         input_dim=input_dim)
        out2 = (models, th)
    else:
        out2 = (models, None)
    out = evaluate_multiple_models(dg_use, models, train_func, test_distributions,
                                   **classifier_args)
    return out, out2 

def _model_pca(dg_use, model, n_dim_red=10**4, use_arc_dim=False,
               use_circ_dim=False, supply_range=1, set_inds=(0, 1),
               start_vals=(0,), **pca_args):
    if supply_range is None:
        supply_range = dg_use.source_distribution.cov[0, 0]
    if len(start_vals) == 0:
        start_vals = start_vals*dg_use.input_dim
    if use_arc_dim:
        distrib_pts = np.zeros((n_dim_red, dg_use.input_dim))
        distrib_pts[:] = start_vals
        x0_pts = np.linspace(-supply_range, supply_range, n_dim_red)
        x1_pts = np.linspace(-supply_range, supply_range, n_dim_red)
        distrib_pts[:, set_inds[0]] = x0_pts
        distrib_pts[:, set_inds[1]] = x1_pts
    elif use_circ_dim:
        r = np.sqrt(dg_use.source_distribution.cov[0, 0])
        distrib_pts = da.get_circle_pts(n_dim_red, dg_use.input_dim, r=r)
    else:
        distrib_pts = dg_use.source_distribution.rvs(n_dim_red)
    distrib_reps = dg_use.get_representation(distrib_pts)
    mod_distrib_reps = model.get_representation(distrib_reps)
    p = skd.PCA(**pca_args)
    p.fit(mod_distrib_reps)
    return p.transform

def _model_linreg(dg_use, model, n_dim_red=10**4, use_arc_dim=False,
                  use_circ_dim=False, supply_range=1, set_inds=(0, 1),
                  linmod=sklm.Ridge, **pca_args):
    x, y = dg_use.sample_reps(n_dim_red)
    r = model.get_representation(y)
    p = linmod()
    p.fit(r, x[:, set_inds])
    ob = spla.orth(p.coef_.T)
    exp_inter = np.expand_dims(p.intercept_, 1)
    f = lambda x: (np.dot(x, ob).T + exp_inter).T
    return f # p.predict                         

def compute_task_performance(fdg, m, flip_tasks=False, decon_tasks=False,
                             n_samps=10000):
    stim, inps = fdg.sample_reps(n_samps)
    outs = np.array(m.class_model(m.get_representation(inps)))
    tasks = m.p_funcs
    if flip_tasks:
        tasks = da.flip_contextual_tasks(tasks)
    elif decon_tasks:
        tasks = da.decontextualize_tasks(tasks)
    targs = np.stack(list(t(stim) for t in tasks), axis=1)
    resps = (outs > .5).astype(int)
    perf = 1 - np.nanmean((targs - resps)**2, axis=0)
    return perf
    
def plot_func_rfs(fdg, func=None, extent=2, n_pts=100,
                  axs_flat=None, mask=True, fwid=3,
                  random_order=False, show_ticks=False,
                  rasterized=True, **kwargs):
    rng = np.random.default_rng()
    vals_x = np.linspace(-extent, extent, n_pts)
    vals_y = np.linspace(-extent, extent, n_pts)
    vals = np.array(list(it.product(vals_x, vals_y)))
    zs = np.zeros((vals.shape[0], 1))

    n_zeros = fdg.input_dim - 2
    input_map = np.concatenate((vals,) + (zs,)*n_zeros, axis=1)
    rep = np.array(fdg.get_representation(input_map))
    if func is not None:
        rep = func(rep)
    rep_map = np.reshape(rep, (n_pts, n_pts, rep.shape[1]))
    if axs_flat is None:
        f, axs_flat = plt.subplots(1, rep.shape[1],
                                   figsize=(fwid*rep.shape[1], fwid),
                                   squeeze=False,
                                   sharex=True,
                                   sharey=True)
        axs_flat = axs_flat[0]
    
    if mask:
        on_mask = np.any(rep_map > 0, axis=(0, 1))
        rep_map = rep_map[..., on_mask]
    if rep.shape[1] < len(axs_flat):
        replace = True
        print('replacing')
    else:
        replace = False
    if random_order:
        inds = rng.choice(rep_map.shape[-1], len(axs_flat), replace=replace)
    else:
        inds = np.arange(len(axs_flat), dtype=int)
    for i, ax in enumerate(axs_flat):
        m = gpl.pcolormesh(vals_x, vals_y, rep_map[..., inds[i]], ax=ax,
                           rasterized=rasterized, **kwargs)
        if show_ticks:
            ax.set_xticks([-extent, 0, extent])
            ax.set_yticks([-extent, 0, extent])
        else:
            ax.set_xticks([])
            ax.set_yticks([])
            
    return m, axs_flat
    

def plot_dg_rfs(fdg, func=None, n_plots_rows=5, axs=None, fwid=1,
                **kwargs):
    rng = np.random.default_rng()
    if axs is None:
        f, axs = plt.subplots(n_plots_rows, n_plots_rows, sharex=True,
                              sharey=True, figsize=(fwid*n_plots_rows,
                                                    fwid*n_plots_rows))

    axs_flat = axs.flatten()
    out = plot_func_rfs(fdg, func=func, axs_flat=axs_flat, random_order=True,
                  **kwargs)
    return out

def compute_sparsity(model, n_samps=10000):
    _, reps = model.sample_reps(n_samps)
    fracs = np.mean(reps != 0, axis=1)
    return fracs

def empirical_model_manifold(ls_pts, rep_pts, rads=(0, .2, .4, .6),
                             rad_eps=.05, near_eps=.5, ax=None,
                             subdims=(0, 1), use_lr=False,
                             half_pts=False, half_ind=-1):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    dim_mask = np.isin(np.array(list(range(ls_pts.shape[1]))),
                       subdims)
    if use_lr:
        lr = sklm.LinearRegression()
        if half_pts:
            hp_mask = ls_pts[:, half_ind] < 0
            rep_pts_m = rep_pts[hp_mask]
            ls_pts_m = ls_pts[hp_mask]
        else:
            rep_pts_m = rep_pts
            ls_pts_m = ls_pts
        lr.fit(rep_pts_m, ls_pts_m[:, dim_mask])
        if half_pts:
            ls_pts = ls_pts[np.logical_not(hp_mask)]
            rep_lp = lr.predict(rep_pts[np.logical_not(hp_mask)])
        else:
            rep_lp = lr.predict(rep_pts)            
        pr = u.cosine_similarity(*lr.coef_)
    else:
        p = skd.PCA()
        var_mask = np.all(np.abs(ls_pts[:, np.logical_not(dim_mask)]) < near_eps,
                          axis=1)
        p.fit(rep_pts[var_mask])
        rep_lp = p.transform(rep_pts)
        evr = p.explained_variance_ratio_
        pr = np.sum(evr)**2/np.sum(evr**2)
    for i, r in enumerate(rads):
        mask = np.abs(u.euclidean_distance(ls_pts[:, dim_mask], 0) - r) < rad_eps
        rep_pts_plot = rep_lp[mask]
        ax.plot(rep_pts_plot[:, subdims[0]], rep_pts_plot[:, subdims[1]], 'o')
    return pr

def plot_source_manifold(*args, axs=None, fwid=3, dim_red=True,
                         source_scale_mag=.2, rep_scale_mag=10,
                         titles=True, plot_source_3d=False, plot_model_3d=False,
                         source_view_init=None, model_view_init=None, 
                         l_axlab_str='PC {} (au)', r_axlab_str='PC {} (au)',
                         **kwargs):
    if axs is None:
        fsize = (fwid*2, fwid)
        f, axs = plt.subplots(1, 2, figsize=fsize)
        out = f, axs
    else:
        out = axs
    if len(axs) > 1:
        plot_diagnostics(*args, plot_partitions=True, plot_source=True,
                         dim_red=False, ax=axs[0], scale_mag=source_scale_mag,
                         plot_3d=plot_source_3d, view_init=source_view_init,
                         **kwargs)
        ind = 1
    else:
        ind = 0
    plot_diagnostics(*args, dim_red=dim_red, ax=axs[ind], scale_mag=rep_scale_mag,
                     compute_pr=True, plot_3d=plot_model_3d,
                     view_init=model_view_init, **kwargs)
    if titles:
        if len(axs) > 1:
            axs[0].set_title('latent variables')
        axs[ind].set_title('representation')
    if len(axs) > 1:
        axs[0].set_xlabel(l_axlab_str.format(1))
        axs[0].set_ylabel(l_axlab_str.format(2))
    axs[ind].set_xlabel(r_axlab_str.format(1))
    axs[ind].set_ylabel(r_axlab_str.format(2))
    return out

def make_half_square(n_pts_per_side, lpt=0, rpt=1):
    pts = np.zeros((n_pts_per_side*2, 2))
    side_trav = np.linspace(lpt, rpt, n_pts_per_side)
    h_lpt = lpt + (rpt - lpt)*.5
    half_trav = np.linspace(h_lpt, rpt, int(n_pts_per_side*.5))
    npps = n_pts_per_side
    h_npps = int(npps*.5)
    pts[:h_npps, 0] = lpt
    pts[:h_npps, 1] = half_trav
    pts[h_npps:npps + h_npps, 0] = side_trav
    pts[h_npps:npps + h_npps, 1] = rpt
    pts[npps + h_npps:2*npps, 0] = rpt
    pts[npps + h_npps:2*npps, 1] = half_trav[::-1]
    corners = np.array([[lpt, h_lpt],
                        [lpt, rpt],
                        [rpt, rpt],
                        [rpt, h_lpt]])
    return pts, corners
    

def make_square(n_pts_per_side=100, lpt=0, rpt=1):
    pts = np.zeros((n_pts_per_side*4, 2))
    side_trav = np.linspace(lpt, rpt, n_pts_per_side)
    npps = n_pts_per_side
    pts[:npps, 0] = lpt
    pts[:npps, 1] = side_trav
    pts[npps:2*npps, 0] = side_trav
    pts[npps:2*npps, 1] = rpt
    pts[2*npps:3*npps, 0] = rpt
    pts[2*npps:3*npps, 1] = side_trav[::-1]
    pts[3*npps:4*npps, 0] = side_trav[::-1]
    pts[3*npps:4*npps, 1] = lpt
    corners = np.array([[lpt, lpt],
                        [lpt, rpt],
                        [rpt, rpt],
                        [rpt, lpt],
                        [lpt, lpt]])

    return pts, corners

def plot_task_reps(dg_use, model, axs=None, fwid=3, n_samps=1000,
                   plot_tasks=None, bins=20, colors=None, **kwargs):
    if plot_tasks is None:
        plot_tasks = np.arange(model.n_partitions)
    if axs is None:
        f, axs = plt.subplots(1, len(plot_tasks),
                              figsize=(len(plot_tasks)*fwid, fwid),
                              squeeze=False)
        axs = axs[0]
    out_mat, out_bias = model.class_model.weights[-2:]
    out_mat, out_bias = out_mat.numpy(), np.expand_dims(out_bias.numpy(), 1)
    stim, inp_reps = dg_use.sample_reps(n_samps)
    corr = model.generate_target(stim)
    u_corrs_all = np.unique(corr)
    if colors is None:
        colors = (None,)*len(u_corrs_all)
    lat_reps = model.get_representation(inp_reps).numpy()
    out_act = (np.dot(out_mat.T, lat_reps.T) + out_bias).T
    for i, task_ind in enumerate(plot_tasks):
        ax = axs[i]
        for j, c_val in enumerate(np.unique(corr)):
            mask = corr[:, task_ind] == c_val
            ax.hist(out_act[mask, task_ind], bins=bins,
                    color=colors[j], **kwargs)    

def train_dim_dec(stim, rep, use_inds=(0, 1), use_thr=0,
                  regr_model=sklm.Ridge, class_model=skc.SVC,
                  y_axis='regression'):
        
    if use_thr is not None:
        half_mask = stim[:, use_inds[1]] < use_thr
    else:
        half_mask = np.ones(stim.shape[0], dtype=bool)
    if y_axis == 'regression':
        m_half = regr_model()
        targ = stim[:, use_inds[0]]
        m_half_func = m_half.predict
    elif y_axis == 'classification':
        m_half = class_model(kernel='linear')
        targ = stim[:, use_inds[0]] > use_thr
        m_half_func = m_half.decision_function
    else:
        raise IOError('y_axis {} is not recognized'.format(y_axis))
            
    m_half.fit(rep[half_mask], targ[half_mask])

    m_full = regr_model()
    m_full.fit(rep, stim[:, use_inds[1]])
    
    return m_full.predict, m_half_func

def make_rep_grid(dg_use, model, grid_len=2, grid_pts=100, use_inds=(0, 1),
                  dims=None):
    dims = dg_use.input_dim
    side_pts = np.linspace(-grid_len, grid_len, grid_pts)
    grid_pts = np.array(list(it.product(side_pts,
                                        repeat=2)))
    stim = np.zeros((grid_pts.shape[0], dims))
    stim[:, use_inds] = grid_pts
    inp_rep = dg_use.get_representation(stim)
    lat_rep = model.get_representation(inp_rep)
    return stim, inp_rep, lat_rep

def make_task_mask(task, grid_len=2, grid_pts=100, use_inds=(0, 1),
                   dims=None):
    if dims is None:
        dims = task.keywords['plane'].shape[1]
    side_pts = np.linspace(-grid_len, grid_len, grid_pts)
    grid = np.array(list(it.product(side_pts, side_pts)))
    stim = np.zeros((grid.shape[0], dims))
    stim[:, use_inds] = grid
    task_out = task(stim)
    mask = np.reshape(task_out, (grid_pts, grid_pts))
    return mask

def plot_tasks(model, use_inds=(0, 1), extent=1, buff=.2,
               ax=None, fwid=3, color=(.3, .3, .3),
               linestyle='dashed', y_extent=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
    if y_extent is None:
        y_extent = extent
    tasks = u.make_unit_vector(model.p_vectors[:, use_inds],
                               squeeze=False)*np.sqrt(2*extent**2)
    for i, task in enumerate(tasks):
        ax.plot([-task[use_inds[0]], task[use_inds[0]]],
                [task[use_inds[1]], -task[use_inds[1]]],
                linestyle=linestyle, 
                color=color,
                **kwargs)
    ax.set_xlim([-extent - buff, extent + buff])
    ax.set_ylim([-y_extent - buff, y_extent + buff])

def plot_task_groups(dg_use, model, grid_len=2, grid_pts=100, use_inds=(0, 1),
                     ax=None, fwid=3, colormap='Spectral',
                     class_thr=.5, ms=.5, buff=.2):  
    cmap = plt.get_cmap(colormap)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
    stim, inp_rep, lat_rep = make_rep_grid(dg_use, model, use_inds=use_inds,
                                           grid_len=grid_len, grid_pts=grid_pts)
    class_out = model.class_model(lat_rep).numpy() < class_thr
    u_classes = np.unique(class_out, axis=0)
    n_classes = len(u_classes)

    train_stim, train_inp = dg_use.sample_reps(stim.shape[0])
    train_rep = model.get_representation(train_inp)
    m_x, m_y = train_dim_dec(train_stim, train_rep, use_inds=use_inds,
                             use_thr=None)

    x_coords = m_x.predict(lat_rep)
    y_coords = m_y.predict(lat_rep)
    for i, c in enumerate(u_classes):
        mask = np.all(class_out == c, axis=1)
        xs = x_coords[mask]
        ys = y_coords[mask]
        ax.plot(xs, ys, 'o', ms=ms, color=cmap(i/n_classes))

    if model.p_vectors is not None:
        plot_tasks(model, use_inds=use_inds, extent=grid_len, buff=buff, ax=ax)        
        
    gpl.make_xaxis_scale_bar(ax, 1, label='changed feature')
    gpl.make_yaxis_scale_bar(ax, 1, label='learned feature')
    gpl.clean_plot(ax, 0)

def plot_grids_tasks(fdg, model_list, axs=None, fwid=3, add_ident=True,
                     **kwargs):
    i_thr = -1
    if add_ident:
        model_list = (dd.IdentityModel(),) + tuple(model_list)
        i_thr = 0
    if axs is None:
        f, axs = plt.subplots(len(model_list), 3,
                              figsize=(fwid*3, fwid*len(model_list)))
    for i, m in enumerate(model_list):
        plot_class_grid(fdg, m, ax=axs[i, 0], **kwargs)
        plot_regr_grid(fdg, m, ax=axs[i, 1], **kwargs)
        if i > i_thr:
            plot_task_reps(fdg, m, axs=(axs[i, 2],), plot_tasks=(0,))
    
def plot_regr_grid(*args, **kwargs):
    return plot_grid(*args, y_axis='regression',  **kwargs)

def plot_class_grid(*args, ax=None, y_label='learned\nclassification', **kwargs):
    out =  plot_grid(*args, y_axis='classification', n_digis=2,
                     ax=ax, y_label=y_label, use_max_out=True, **kwargs)
    gpl.add_hlines(0, ax)
    return out
        
def plot_grid(dg_use, model, grid_len=2, grid_pts=100, use_inds=(0, 1),
              ax=None, fwid=3, n_digis=8, eps=.01, colormap='Spectral',
              ms=1, buff=.2, y_axis='regression',
              y_label='learned feature', x_label='contextual feature',
              use_max_out=False, col_eps=0):
    cmap = plt.get_cmap(colormap)
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))

    stim, inp_rep, lat_rep = make_rep_grid(dg_use, model, grid_len=grid_len,
                                           grid_pts=grid_pts,
                                           use_inds=use_inds)
        
    train_stim, train_inp = dg_use.sample_reps(stim.shape[0])
    train_rep = model.get_representation(train_inp)

    m_full, m_half = train_dim_dec(train_stim, train_rep, use_inds=use_inds,
                                   y_axis=y_axis)
    
    y_coords = m_half(lat_rep)
    x_coords = m_full(lat_rep)

    digi_bins = np.linspace(-grid_len, grid_len + eps, n_digis + 1)
    stim_bins = np.digitize(stim[:, use_inds[0]], digi_bins) - 1
    norm_bins = np.max(stim_bins)
    colors = cmap(np.linspace(col_eps, 1 - col_eps, norm_bins + 1))
    if use_max_out:
        y_grid_len = np.round(max(np.max(np.abs(y_coords)), 1), 0)
    else:
        y_grid_len = grid_len
    for i, sb in enumerate(np.unique(stim_bins)):
        coord_mask = sb == stim_bins
        ax.plot(x_coords[coord_mask], y_coords[coord_mask], 'o',
                ms=ms, color=colors[sb])

    if len(model.p_vectors) > 0:
        plot_tasks(model, use_inds=use_inds, extent=grid_len, buff=buff, ax=ax,
                   linewidth=1,
                   y_extent=y_grid_len)
    
    gpl.make_xaxis_scale_bar(ax, grid_len/2, label=x_label)
    gpl.make_yaxis_scale_bar(ax, y_grid_len/2, label=y_label)
    gpl.clean_plot(ax, 0)
    gpl.add_vlines(0, ax)

def _get_rep_seq(fdg, model, hold_val, hold_feats=(1,),
                 zero_feats=(), n_samps=10000):
    samps, inps = fdg.sample_reps(n_samps)
    samps[:, zero_feats] = 0
    samps[:, hold_feats] = hold_val
    inps = fdg.get_representation(samps)
    reps = model.get_representation(inps)
    targs = model.generate_target(samps)
    return samps, inps, reps, targs
    
    
def compute_pointwise_generalization(fdg, model, grid_pts=10, extent=2,
                                     dec_feat=0, hold_feat=1, other_zeros=True,
                                     dec_type='regression', use_targ=False,
                                     regr_model=sklm.Ridge, class_model=skc.SVC,
                                     n_train_test=10000):
    pts = np.linspace(-extent, extent, grid_pts)
    if other_zeros:
        use_feats = (dec_feat, hold_feat)
        zero_feats = set(np.arange(fdg.input_dim)).difference(use_feats)
        zero_feats = tuple(zero_feats)
    else:
        zero_feats = ()
    out_sc = np.zeros((grid_pts, grid_pts))
    out_pr = np.zeros((grid_pts, grid_pts, n_train_test))
    for i, j in it.product(range(grid_pts), repeat=2):
        pt1, pt2 = pts[i], pts[j]
        out_tr = _get_rep_seq(fdg, model, pt1, zero_feats=zero_feats,
                              hold_feats=(hold_feat,), n_samps=n_train_test)
        samps_tr, inps_tr, reps_tr, targs_tr = out_tr

        out_te = _get_rep_seq(fdg, model, pt2, zero_feats=zero_feats,
                              hold_feats=(hold_feat,), n_samps=n_train_test)
        samps_te, inps_te, reps_te, targs_te = out_te
        if use_targ:
            rep_data_tr = targs_tr
            rep_data_te = targs_te
        else:
            rep_data_tr = reps_tr
            rep_data_te = reps_te
        if dec_type == 'regression':
            m = regr_model()
            m.fit(rep_data_tr, samps_tr[:, dec_feat])
            out_sc[i, j] = m.score(rep_data_te, samps_te[:, dec_feat])
        elif dec_type == 'classification':
            m = class_model(kernel='linear')
            m.fit(rep_data_tr, samps_tr[:, dec_feat] < 0)
            out_sc[i, j] = m.score(rep_data_te, samps_te[:, dec_feat] < 0)
        out_pr[i, j] = m.predict(rep_data_te)
    return pts, out_sc, out_pr

def plot_diagnostics(dg_use, model, rs, n_arcs, ax=None, n=1000, dim_red=True,
                     n_dim_red=10**4, pt_size=2, line_style='solid',
                     markers=True, line_alpha=1, use_arc_dim=False,
                     use_circ_dim=False, arc_col=(.8, .8, .8), scale_mag=.5,
                     fwid=2.5, set_inds=(0, 1), plot_partitions=False,
                     plot_source=False, square=True, start_vals=(0,),
                     supply_range=1, plot_3d=False, dim_red_func=None,
                     compute_pr=False, ret_dim_red=False, buff=0,
                     model_trs=_model_pca, view_init=None,
                     linewidth=1, **pca_args):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
        
    pts = np.zeros((n, dg_use.input_dim))
    if len(start_vals) == 1:
        start_vals = start_vals*dg_use.input_dim
    pts[:] = start_vals
    if square:
        quarter = int(np.floor(n/4))
        side_pts = np.linspace(-1, 1, quarter)
        
        pts[:quarter, set_inds[0]] = side_pts
        pts[:quarter, set_inds[1]] = 1
        
        pts[quarter:2*quarter, set_inds[0]] = 1
        pts[quarter:2*quarter, set_inds[1]] = -side_pts
        
        pts[2*quarter:3*quarter, set_inds[0]] = -side_pts
        pts[2*quarter:3*quarter, set_inds[1]] = -1
        
        pts[3*quarter:4*quarter, set_inds[0]] = -1
        pts[3*quarter:4*quarter, set_inds[1]] = side_pts
        pts = pts[:4*quarter]
    else:
        angs = np.linspace(0, 2*np.pi, n)
        pts[:, set_inds[0]] = np.cos(angs)
        pts[:, set_inds[1]] = np.sin(angs)
    
    if dim_red:
        if dim_red_func is None:
            ptrs = model_trs(dg_use, model, n_dim_red=n_dim_red,
                             use_arc_dim=use_arc_dim, use_circ_dim=use_circ_dim,
                             set_inds=set_inds, start_vals=start_vals,
                             supply_range=supply_range, **pca_args)
            if compute_pr:
                # pd = p.explained_variance_ratio_
                # pr = np.sum(pd)**2/np.sum(pd**2)
                pr = None
        else:
            ptrs = dim_red_func
        
    if n_arcs > 0:
        skips = int(np.round(n/n_arcs))
        sub_pts = pts[::skips]
        plot_pts = np.zeros_like(sub_pts)
        plot_pts[:, set_inds[0]] = rs[-1]*sub_pts[:, set_inds[0]]
        plot_pts[:, set_inds[1]] = rs[-1]*sub_pts[:, set_inds[1]]
        y = np.ones((n, dg_use.input_dim))
        y[:, set_inds[0]] = np.linspace(0, 1, n)
        y[:, set_inds[1]] = np.linspace(0, 1, n)
        for sp in plot_pts:
            sp = np.expand_dims(sp, 0)
            s_reps = dg_use.generator(sp*y)
            if plot_source:
                mod_reps = sp*y
            else:
                mod_reps = model.get_representation(s_reps)
            if dim_red:
                mod_reps = ptrs(mod_reps)
            if plot_3d:
                to_plot = (mod_reps[:, 0], mod_reps[:, 1], mod_reps[:, 2])
            else:
                to_plot = (mod_reps[:, 0], mod_reps[:, 1])
            l = ax.plot(*to_plot, linestyle=line_style,
                        alpha=line_alpha, color=arc_col,
                        linewidth=linewidth)
            if markers:
                ax.plot(*to_plot, 'o', markersize=pt_size,
                        color=l[0].get_color())

    for r in rs:
        r_arr = np.ones_like(pts)
        r_arr[:, set_inds[0]] = r
        r_arr[:, set_inds[1]] = r
        s_reps = dg_use.generator(r_arr*pts)

        if plot_source:
            mod_reps = r*pts
        else:
            mod_reps = model.get_representation(s_reps)
        if dim_red:
            mod_reps = ptrs(mod_reps)
        if plot_3d:
            to_plot = (mod_reps[:, 0], mod_reps[:, 1], mod_reps[:, 2])
        else:
            to_plot = (mod_reps[:, 0], mod_reps[:, 1])
        l = ax.plot(*to_plot,
                    linestyle=line_style, alpha=line_alpha,
                    linewidth=linewidth)
        if markers:
            ax.plot(*to_plot, 'o', markersize=pt_size,
                    color=l[0].get_color())

    if plot_partitions:
        vs = model.p_vectors
        os = model.p_offsets
        for i, v in enumerate(vs):
            v_unit = u.make_unit_vector(v)
            v_o = os[i]*v_unit
            orth_v = u.generate_orthonormal_vectors(v_unit, 1)/(1*rs[-1])
            xs = np.array([-orth_v[0], orth_v[0]]) 
            ys = np.array([-orth_v[1], orth_v[1]])
            ax.plot(xs + v_o[0], ys + v_o[1], color='r',
                    linewidth=linewidth)

    gpl.clean_plot(ax, 0)
    if plot_3d:
        cent = (np.min(to_plot[0]) - buff, np.min(to_plot[1]) - buff,
                np.min(to_plot[2]) - buff)
        var = np.max([np.std(to_plot[0]), np.std(to_plot[1]),
                     np.std(to_plot[2])])
        gpl.make_3d_bars(ax, center=cent, bar_len=var)
        gpl.clean_3d_plot(ax)
        if view_init is not None:
            ax.view_init(*view_init)
    else:
        ax.set_aspect('equal')
        gpl.make_xaxis_scale_bar(ax, scale_mag)
        gpl.make_yaxis_scale_bar(ax, scale_mag)
    if compute_pr:
        out = ax, pr
    else:
        out = ax
    if ret_dim_red:
        out = out, ptrs
    
    return out

def make_classifier_dimred(pcs, pvs, dg_use, model, n_train=500, 
                           n_reps=1, **params):
    classers = []
    norms = []
    test_results = np.zeros((len(pcs), n_reps))
    for i, pc in enumerate(pcs):
        for j in range(n_reps):
            pv = pvs[i]
            out = make_classifier(pc, pv, dg_use, model, n_train=n_train,
                                  **params)
            classer_i, tin, tout = out
            test_results[i, j] = tout
        classers.append(classer_i)
        norms.append(np.sqrt(np.sum(classer_i.coef_**2)))
        
    def transform(x):
        out = np.zeros((x.shape[0], len(classers)))
        for i, cl in enumerate(classers):
            out[:, i] = cl.decision_function(x)/norms[i]
        return out

    return transform, test_results

def _get_classifier_reps(n_samps, td, pc, dg_use, model):
    samps = td.rvs(n_samps)
    cats = np.dot(samps, pc) > 0
    imgs = dg_use.get_representation(samps)
    reps = model.get_representation(imgs)
    return samps, cats, imgs, reps

def make_all_classifiers(dg_use, model, n_reps=1, **params):
    n_dims = dg_use.input_dim
    class_partitions = np.identity(n_dims)
    train_partitions = np.identity(n_dims)
    dec_matrix = np.zeros((n_dims, n_dims, n_reps))
    for i, cp in enumerate(class_partitions):
        for j, tp in enumerate(train_partitions):
            for k in range(n_reps):
                if i == j:
                    out = make_classifier(cp, None, dg_use, model, test=True,
                                          **params)
                else:
                    out = make_classifier(cp, tp, dg_use, model, test=True,
                                          **params)
                dec_matrix[i, j, k] = out[1]
    return dec_matrix 

_default_labels = ('rotation', 'pitch', 'x-pos', 'y-pos')
def plot_feature_ccgp(dec_mat, ax_labels=_default_labels,
                      axs=None, fwid=2, **kwargs):
    if axs is None:
        f, axs = plt.subplots(1, 3, figsize=(3*fwid, fwid))
    no_abstract = np.identity(dec_mat.shape[0])
    all_abstract = np.ones(dec_mat.shape[:2])
    actual = np.mean(dec_mat, axis=2)
    axvals = np.arange(dec_mat.shape[0])
    gpl.pcolormesh(axvals, axvals[::-1], no_abstract, vmin=.5, vmax=1, ax=axs[0],
                   **kwargs)
    gpl.pcolormesh(axvals, axvals[::-1], all_abstract,vmin=.5, vmax=1,  ax=axs[1],
                   **kwargs)
    img = gpl.pcolormesh(axvals, axvals[::-1], actual, vmin=.5, vmax=1,
                         ax=axs[2], **kwargs)
    axs[0].set_ylabel('decoded')
    axs[1].set_xlabel('partitioned')
    axs[0].set_title('not abstract')
    axs[1].set_title('fully abstract')
    axs[2].set_title('actual')
    for ax in axs:
        ax.set_ylabel('decoded')
        ax.set_xlabel('partitioned')
        ax.set_xticks(axvals)
        ax.set_yticks(axvals)
        ax.set_xticklabels(ax_labels, rotation=90)
        ax.set_yticklabels(ax_labels[::-1])
        ax.set_aspect('equal')
    cb = f.colorbar(img, ax=axs, orientation='vertical')
    cb.set_ticks([.5, 1])
    cb.set_label('decoding performance')
    return f, axs    

def make_classifier(pc, pv, dg_use, model, n_train=500, test=False,
                    classifier_model=skc.LinearSVC, max_iter=4000,
                    norm=True, **params):
    if pv is not None:
        td = dg_use.source_distribution.make_partition(partition_vec=pv)
        flip = True
    else:
        td = dg_use.source_distribution
        n_train = 2*n_train
        flip = False

    out = _get_classifier_reps(n_train, td, pc, dg_use, model)
    train_samps, train_cats, train_imgs, train_reps = out

    classer = classifier_model(max_iter=max_iter, **params)
    classer.fit(train_reps, train_cats)

    if test:
        if flip:
            test = td.flip()
        else:
            test = td
        out = _get_classifier_reps(n_train, test, pc, dg_use, model)
        test_samps, test_cats, test_imgs, test_reps = out
        test_out = classer.score(test_reps, test_cats)
    else:
        test_out = None
    return classer, test_out

def plot_partitions(pf_planes, ax, dim_red=None, scale=1):
    for pfp in pf_planes:
        if dim_red is not None:
            pfp = dim_red(pfp)
        pfp = np.stack((pfp, np.zeros_like(pfp)), axis=1)*scale
        l = ax.plot(*pfp)
        neg_pfp = -pfp
        ax.plot(*neg_pfp, color=l[0].get_color())
    return ax

def get_model_dimensionality(dg_use, models, cutoff=.95, **pca_args):
    evr, _ = dg_use.representation_dimensionality(n_components=cutoff)
    inp_dim = len(evr)
    model_dims = np.zeros_like(models, dtype=int)
    for i, m_i in enumerate(models):
        for j, m_ij in enumerate(m_i):
            for k, m_ijk in enumerate(m_ij):
                p = _model_pca(dg_use, m_ijk, n_components=cutoff, **pca_args)
                model_dims[i, j, k] = 1 # p.n_components_
    return model_dims, inp_dim


model_kinds_default = (dd.SupervisedDisentangler, dd.StandardAE)
dg_kind_default = dg.FunctionalDataGenerator

def test_generalization(dg_use=None, models_ths=None, train_models_blind=False,
                        p_c=None, dg_kind=dg_kind_default,
                        hide_print=True, est_inp_dim=None,
                        use_mp=False, n_reps=5, n_train_diffs=6,
                        model_kinds=model_kinds_default, layer_spec=None):

    # train data generator
    inp_dim = 2
    out_dim = 30

    noise = .1
    reg_weight = (0, .1)
    layers =  (20, 50, 50, 50, 30)
    epochs = 25

    source_var = 1

    if dg_use is None:
        dg_use = dg_kind(inp_dim, layers, out_dim, 
                     l2_weight=reg_weight, noise=noise)

        source_distr = sts.multivariate_normal(np.zeros(inp_dim), source_var)
        dg_use.fit(source_distribution=source_distr, epochs=epochs,
               use_multiprocessing=use_mp)
    else:
        source_distr = dg_use.source_distribution

    # test dimensionality
    rv, vecs = dg_use.representation_dimensionality(source_distribution=source_distr)
    f, ax = plt.subplots(1, 1)
    ax.plot(rv)

    if train_models_blind:
        train_d2 = da.HalfMultidimensionalNormal(np.zeros(inp_dim), 1)
        test_d2 = train_d2.flip()
        dg.source_distribution = train_d2

    # train models
    if est_inp_dim is None:
        est_inp_dim = inp_dim
    if layer_spec is None:
        layer_spec = ((40,), (40,), (25,), (10,))
    batch_size = 1000
    epochs = 60
    train_samples = np.logspace(3, 6.5, n_train_diffs, dtype=int)
    input_dims = (est_inp_dim,)*n_train_diffs
    samps_list = True
    use_x = train_samples
    log_x = True
    
    if models_ths is None:
        models, th = train_multiple_models_dims(input_dims, dg_use, model_kinds,
                                                layer_spec, n_reps=n_reps, 
                                                batch_size=batch_size,
                                                epochs=epochs,
                                                samps_list=samps_list,
                                                hide_print=hide_print,
                                                n_train_samps=train_samples,
                                                use_mp=use_mp)

    else:
        models, th = models_ths
        
    f, ax = plt.subplots(1, 1)
    legend_patches = []
    line_styles = ('-', '--', ':', '-.')
    for i, h_i in enumerate(th):
        color = None
        for j, h_ij in enumerate(h_i):
            line_style = line_styles[j % len(line_styles)]
            for h_ijk in h_ij:
                vl = h_ijk.history['val_loss']
                l = ax.plot(range(len(vl)), vl, color=color,
                            linestyle=line_style)
                color = l[0].get_color()
        lpatch = mpatches.Patch(color=color,
                                label='N = {}'.format(train_samples[i]))
        legend_patches.append(lpatch)
    ax.legend(handles=legend_patches)
        
    dims, dg_dim = get_model_dimensionality(dg_use, models)
    
    f, ax = plt.subplots(1, 1)
    
    ax.hlines(dg_dim, use_x[0], use_x[-1])
    for i in range(dims.shape[1]):
        gpl.plot_trace_werr(use_x, dims[:, i].T, ax=ax, log_x=log_x)


    tf = None
    td_same = None
    input_dim = None

    try:
        train_d2 = dg_use.source_distribution.make_partition()
    except AttributeError:
        train_d2 = da.HalfMultidimensionalNormal.partition(
            dg_use.source_distribution)
    test_d2 = train_d2.flip()

    train_ds = (None, train_d2)
    tds = (td_same, test_d2)
    n_iters = 2

    if p_c is None:
        p, c = evaluate_multiple_models_dims(dg_use, models, tf, tds,
                                             train_distributions=train_ds,
                                             n_iters=n_iters)
    else:
        p, c = p_c

    plot_wid = 3
    n_models = p.shape[1]
    ref_model = 0
    tds_labels = ('source', 'half-distribution generalization')

    f = plt.figure(figsize=(plot_wid*(n_models + 1), plot_wid))

    ax_comp = f.add_subplot(1, n_models + 1, n_models + 1)
    ax_j = None

    for j in range(n_models):
        ax_j = f.add_subplot(1, n_models + 1, j + 1, sharey=ax_j)
        line_style = line_styles[j % len(line_styles)]
        for i in range(len(tds)):
            p_diff = p[:, j] - p[:, ref_model]
            if j == 0:
                label = tds_labels[i]
            else:
                label = ''
            l = gpl.plot_trace_werr(use_x, p[:, j, :, i].T, ax=ax_j, 
                                    points=True, log_x=log_x,
                                    linestyle=line_style)
            col = l[0].get_color()
            l = gpl.plot_trace_werr(use_x, p_diff[..., i].T, ax=ax_comp,
                                    label=label, points=True,
                                    log_x=log_x, color=col,
                                    linestyle=line_style)

    rs = (.1, .2, .5)
    n_arcs = 1

    n_ds, n_mks, n_reps = models.shape
    rep_ind = 0

    dim_red = True

    psize = 4
    fsize = (n_mks*psize, n_ds*psize)
    f, axs = plt.subplots(n_ds, n_mks, figsize=fsize, squeeze=False)
    
    for i in range(n_ds):
        for j in range(n_mks):
            mod = models[i, j, rep_ind]
            plot_diagnostics(dg_use, mod, rs, n_arcs, ax=axs[i, j],
                             dim_red=dim_red)

    return dg_use, (models, th), (p, c)

def plot_representation_dimensionality(dg_use, source_distr=None, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    rv, vecs = dg_use.representation_dimensionality(source_distribution=source_distr)
    ax.plot(rv)
    return ax

def plot_model_dimensionality(dg_use, models, use_x, ax=None, log_x=True):
    if ax is None:
        f, ax = plt.subplots(1, 1)

    dims, dg_dim = get_model_dimensionality(dg_use, models)
    ax.hlines(dg_dim, use_x[0], use_x[-1])
    for i in range(dims.shape[1]):
        gpl.plot_trace_werr(use_x, dims[:, i].T, ax=ax, log_x=log_x)
    return ax

def plot_training_progress(th, labels, ax=None,
                           line_styles=('-', '--', ':', '-.')):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    legend_patches = []
    for i, h_i in enumerate(th):
        color = None
        for j, h_ij in enumerate(h_i):
            line_style = line_styles[j % len(line_styles)]
            for h_ijk in h_ij:
                vl = h_ijk.history['val_loss']
                l = ax.plot(range(len(vl)), vl, color=color,
                            linestyle=line_style)
                color = l[0].get_color()
        lpatch = mpatches.Patch(color=color,
                                label='N = {}'.format(labels[i]))
        legend_patches.append(lpatch)
    ax.legend(handles=legend_patches)
    return ax

def plot_generalization_performance(use_x, p, plot_wid=3, ref_model=0,
                                    tds_labels=None, log_x=True,
                                    indiv_pts=True,
                                    line_styles=('-', '--', ':', '-.')):
    if tds_labels is None:
        tds_labels = ('source', 'half-distribution generalization')

    n_models = p.shape[1]
    n_tds = p.shape[-1]
    n_reps = p.shape[-2]
    f = plt.figure(figsize=(plot_wid*(n_models + 1), plot_wid))
    ax_comp = f.add_subplot(1, n_models + 1, n_models + 1)
    ax_j = None

    for j in range(n_models):
        ax_j = f.add_subplot(1, n_models + 1, j + 1, sharey=ax_j)
        line_style = line_styles[j % len(line_styles)]
        for i in range(n_tds):
            p_diff = p[:, j] - p[:, ref_model]
            if j == 0:
                label = tds_labels[i]
            else:
                label = ''
            l = gpl.plot_trace_werr(use_x, p[:, j, :, i].T, ax=ax_j, 
                                    points=True, log_x=log_x,
                                    linestyle=line_style)
            col = l[0].get_color()
            if indiv_pts:
                for k in range(n_reps):
                    ax_j.plot(use_x, p[:, j, k, i], 'o', color=col)
            l = gpl.plot_trace_werr(use_x, p_diff[..., i].T, ax=ax_comp,
                                    label=label, points=True,
                                    log_x=log_x, color=col,
                                    linestyle=line_style)
    return f

def plot_model_manifolds(dg_use, models, rs=(.1, .2, .5), n_arcs=1, rep_ind=0,
                         dim_red=True, psize=4):
    n_ds, n_mks = models.shape[:2]
    f, axs = plt.subplots(n_ds, n_mks, squeeze=False,
                          figsize=(n_mks*psize, n_ds*psize))
    for i in range(n_ds):
        for j in range(n_mks):
            mod = models[i, j, rep_ind]
            plot_diagnostics(dg_use, mod, rs, n_arcs, ax=axs[i, j],
                                dim_red=dim_red)
    return f

def plot_recon_gen_corr(scores, gens, color_ax=None, ax=None):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    gens = np.mean(gens, axis=3)
    if color_ax is None:
        sc_flat = np.reshape(scores, (1, -1))
        gen_flat = np.reshape(gens, (1, -1))
        ax_sz = 1
    else:
        scores = np.swapaxes(scores, 0, color_ax)
        gens = np.swapaxes(gens, 0, color_ax)
        ax_sz = scores.shape[color_ax]
        sc_flat = np.reshape(scores, (ax_sz, -1))
        gen_flat = np.reshape(gens, (ax_sz, -1))
    for i in range(ax_sz):
        ax.plot(sc_flat[i], gen_flat[i], 'o')
    return ax

def plot_recon_accuracies_ntrain(scores, xs=None, axs=None, fwid=2,
                                 plot_labels='train egs = {}', n_plots=None,
                                 ylabel='', ylim=None, num_dims=None,
                                 xlab='partitions', collapse_plots=False,
                                 set_title=True, intermediate=False,
                                 plot_intermediate=True, **kwargs):
    print(scores.shape)
    if len(scores.shape) == 4 and not intermediate:
        scores = np.mean(scores, axis=3)
        n_ds, n_mks, n_reps = scores.shape
        n_inter = 0
    elif len(scores.shape) == 4 and intermediate:
        if plot_intermediate:
            n_ds, n_mks, n_reps, n_inter = scores.shape
        else:
            n_ds, n_mks, n_reps, _ = scores.shape
            n_inter = 0
            scores = scores[..., -1]
    elif len(scores.shape) == 5:
        scores = np.mean(scores, axis=-1)
        if plot_intermediate:
            n_ds, n_mks, n_reps, n_inter = scores.shape
        else:
            n_ds, n_mks, n_reps, _ = scores.shape
            n_inter = 0
            scores = scores[..., -1]            
    else:
        n_ds, n_mks, n_reps = scores.shape
        n_inter = 0
    if axs is None:
        f, axs = plt.subplots(n_ds, 1, sharey=True,
                              figsize=(fwid, n_ds*fwid))
    for i, sc in enumerate(scores):
        # title = plot_labels.format(n_plots[i])
        title = ''
        if collapse_plots:
            plot_ind = 0
            legend = title
            kwargs['label'] = legend
        else:
            plot_ind = i
        if not set_title and collapse_plots:
            kwargs['label'] = ''
        if n_inter > 0:
            labels = kwargs.get('inter_labels')
            if labels is None:
                labels = (None,)*(n_inter - 1)
            colors = kwargs.get('inter_colors')
            if colors is None:
                colors = (None,)*(n_inter - 1)
            for j in range(1, n_inter):
                kwargs['label'] = labels[j - 1]
                kwargs['color'] = colors[j - 1]
                plot_recon_accuracy_partition(sc[..., j], ax=axs[plot_ind],
                                              mks=xs, 
                                              **kwargs)
        else:
            plot_recon_accuracy_partition(sc, ax=axs[plot_ind], mks=xs,
                                          **kwargs)
        axs[plot_ind].set_ylabel(ylabel)
        if (n_plots is not None and len(plot_labels) > 0 and set_title
            and not collapse_plots):
            axs[plot_ind].set_title(title)
        if ylim is not None:
            axs[plot_ind].set_ylim(ylim)
        if num_dims is not None:
            gpl.add_vlines(num_dims, axs[plot_ind])
    axs[plot_ind].set_xlabel(xlab)
    return axs        

def input_dim_tasks(inp_dims, n_tasks, input_distr='normal',
                    model=sklm.Ridge, dec_dim=0, hold_dim=1,
                    thr=0, n_samps=100000, n_reps=10, norm_dist=False,
                    offset_std=0, **kwargs):
    offset_distr = sts.norm(0, offset_std)
    sc_arr = np.zeros((len(inp_dims), len(n_tasks), n_reps))
    pred_arr = np.zeros_like(sc_arr, dtype=object)
    samp_arr = np.zeros_like(sc_arr, dtype=object)
    pred_in_arr = np.zeros_like(sc_arr, dtype=object)
    samp_in_arr = np.zeros_like(sc_arr, dtype=object)
    for i, inp_d in enumerate(inp_dims):
        if input_distr == 'normal':
            sd = sts.multivariate_normal((0,)*inp_d, 1)
        elif input_distr == 'uniform':
            sd = da.MultivariateUniform(inp_d, (-1, 1))
        for j, nt in enumerate(n_tasks):
            for k in range(n_reps):
                samps = sd.rvs(n_samps)
                if norm_dist:
                    samps = samps/np.sqrt(np.sum(samps**2, axis=1, keepdims=True))
                tasks, _, _ = dd.make_tasks(inp_d, nt, offset_distr=offset_distr,
                                            **kwargs)
                targ = np.stack(list(task(samps) for task in tasks),
                                axis=1)
                m = model()
                mask = samps[:, hold_dim] < thr
                m.fit(targ[mask], samps[mask, dec_dim])
                sc = m.score(targ[~mask], samps[~mask, dec_dim])
                pred = m.predict(targ[~mask])
                pred_in = m.predict(targ[mask])
                pred_arr[i, j, k] = pred
                samp_arr[i, j, k] = samps[~mask, dec_dim]
                pred_in_arr[i, j, k] = pred_in
                samp_in_arr[i, j, k] = samps[mask, dec_dim]
                sc_arr[i, j, k] = sc
    return sc_arr, samp_arr, pred_arr, samp_in_arr, pred_in_arr

def expand_intermediate_models(mod_arr):
    for i, ind in enumerate(u.make_array_ind_iterator(mod_arr.shape)):
        model = mod_arr[ind]
        n_layers = len(model.rep_model.layers)
        if i == 0:
            mod_out = np.zeros(mod_arr.shape + (n_layers,),
                               dtype=object)
        for j in range(n_layers):
            mod_out[ind + (j,)] = dd.IntermediateLayers(
                model.rep_model, learn_lvs=model.learn_lvs, use_i=j)
    return mod_out

def plot_recon_accuracy_partition(scores, mks=None, ax=None, indiv_pts=True,
                                  log_x=False, errorbar=False, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    n_mks, n_reps = scores.shape
    if mks is None:
        mks = np.arange(n_mks)
    if errorbar:
        l = gpl.plot_trace_werr(mks, scores.T, ax=ax, log_x=log_x, **kwargs)
    else:
        l = ax.plot(mks, np.nanmedian(scores.T, axis=0),
                    label=kwargs['label'], color=kwargs.get('color'),
                    linestyle=kwargs.get('linestyle'))
        gpl.clean_plot(ax, 0)
        ax.legend(frameon=False)
    if indiv_pts:
        col = l[0].get_color()
        for k in range(n_reps):
            ax.plot(mks, scores[:, k], 'o', color=col)
    return ax
            
def plot_recon_accuracy(scores, use_x=None, ax=None, log_x=False,
                        indiv_pts=True):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if len(scores.shape) > 3:
        scores = np.mean(scores, axis=3)
    n_ds, n_mks, n_reps = scores.shape
    if use_x is None:
        use_x = np.arange(n_ds)
    for j in range(n_mks):
        l = gpl.plot_trace_werr(use_x, scores[:, j].T, ax=ax, log_x=log_x)
        if indiv_pts:
            col = l[0].get_color()
            for k in range(n_reps):
                ax.plot(use_x, scores[:, j, k], 'o', color=col)
    return ax

def find_linear_mappings(dg_use, model_arr, n_train_samps=10**4, half_ns=100,
                         half=True, **kwargs):
    inds = it.product(*(range(x) for x in model_arr.shape))
    scores_shape = model_arr.shape
    if u.check_list(n_train_samps):
        scores_shape = scores_shape + (len(n_train_samps),)
    if half:
        scores_shape = scores_shape + (half_ns,)
    scores = np.zeros(scores_shape, dtype=float)
    sims = np.zeros_like(scores, dtype=object)
    lintrans = np.zeros(model_arr.shape + (2,), dtype=object)
    for ind in inds:
        if u.check_list(n_train_samps):
            lr = []
            sc = []
            sim = []
            for nts in n_train_samps:
                lr_n, sc_n, sim_n = find_linear_mapping(dg_use, model_arr[ind],
                                                        n_train_samps=nts,
                                                        **kwargs)
                lr.append(lr_n)
                sc.append(sc_n)
                sim.append(sim_n)
            lr = np.stack(lr, axis=0)
            sc = np.stack(sc, axis=0)
            sim = np.stack(sim, axis=0)
        else:
            lr, sc, sim = find_linear_mapping(dg_use, model_arr[ind],
                                              n_train_samps=n_train_samps,
                                              **kwargs)
        scores[ind] = sc
        lintrans[ind] = None # lr
        sims[ind] = None 
    return lintrans, scores, sims

def find_linear_mapping(*args, half=True, half_ns=100, comb_func=np.median,
                        **kwargs):
    if half:
        score = np.zeros(half_ns)
        sims = np.zeros_like(score, dtype=object)
        for i in range(half_ns):
            lr, sc, sim = find_linear_mapping_single(*args, half=half, **kwargs)
            score[i] = sc
            sims[i] = sim
    else:
        lr, score, sim = find_linear_mapping_single(*args, half=half, **kwargs)
    return lr, score, sims

def plot_autodis_performance(latents, perf, axs=None):
    if axs is None:
        f, axs = plt.subplots(1, 2)
    ax_c, ax_r = axs
    for layers, (clas, regr) in perf.items():
        l = gpl.plot_trace_werr(latents, clas[..., 1].T, ax=ax_c,
                                label=str(layers))
        col = l[0].get_color()
        gpl.plot_trace_werr(latents, regr[..., 1].T, ax=ax_r,
                            color=col)
    ax_c.set_ylim([.5, 1])
    ax_r.set_ylim([0, 1])

def find_linear_mapping_single(dg_use, model, n_train_samps=10**4,
                               n_test_samps=10**4, half=True,
                               get_parallelism=True, train_stim_set=None,
                               train_labels=None, test_stim_set=None,
                               test_labels=None, feat_mask=None,
                               lr_type=sklm.Ridge,
                               correct=False, repl_mean=None,
                               partition_vec=None, learn_lvs='ignore',
                               eval_dg=None, flip_cat=False,
                               norm_samples=False, **kwargs):
    if learn_lvs == 'trained':
        feat_mask = model.learn_lvs
    elif learn_lvs == 'untrained':
        feat_mask = np.logical_not(model.learn_lvs)
    elif learn_lvs == 'ignore':
        pass
    else:
        raise IOError('{} is not an understood option for '
                      'learn_lvs'.format(learn_lvs))
    if train_stim_set is not None and test_stim_set is not None:
        enc_pts = model.get_representation(train_stim_set)
        test_enc_pts = model.get_representation(test_stim_set)
        stim = train_labels
        test_stim = test_labels
    else:
        if half:
            try:
                src = dg_use.source_distribution.make_partition(
                    partition_vec=partition_vec)
            except AttributeError:
                src = da.HalfMultidimensionalNormal.partition(
                    dg_use.source_distribution)
            stim = src.rvs(n_train_samps)
            if repl_mean is not None:
                stim[:, repl_mean] = src.mean[repl_mean]
        else:
            stim = dg_use.source_distribution.rvs(n_train_samps)
            if repl_mean is not None:
                stim[:, repl_mean] = dg_use.source_distribution.mean[repl_mean]

        enc_pts = model.get_representation(dg_use.generator(stim))
        if half:
            flipped = src.flip()
            if flip_cat:
                flipped = flipped.flip_cat_partition()
            test_stim = flipped.rvs(n_test_samps)
        else:
            test_stim = dg_use.source_distribution.rvs(n_test_samps)
        test_enc_pts = model.get_representation(dg_use.generator(test_stim))
    if feat_mask is None:
        feat_mask = np.ones(stim.shape[1], dtype=bool)
    lr = lr_type(**kwargs)
    if norm_samples:
        stim = stim/np.sqrt(np.sum(stim**2, axis=1, keepdims=True))
        test_stim = test_stim/np.sqrt(np.sum(test_stim**2, axis=1,
                                             keepdims=True))
    if not np.any(np.isnan(enc_pts)):
        lr.fit(enc_pts, stim[:, feat_mask])
        score = lr.score(test_enc_pts, test_stim[:, feat_mask])
        params = lr.get_params()
        if get_parallelism:
            lr2 = lr_type(**kwargs)
            lr2.fit(test_enc_pts, test_stim[:, feat_mask])
            sim = u.cosine_similarity(lr.coef_, lr2.coef_)
        else:
            lr2 = None
            sim = None
        out = (lr, lr2), score, sim
        if correct:
            pred_stim = lr.predict(test_enc_pts)
            out = out + ((test_stim[:, feat_mask], pred_stim),)
    else:
        out = (None, None), np.nan, np.nan
    return out

def compute_contextual_extrapolation(fdg, models, **kwargs):
    perf_all = np.zeros_like(models)
    flip_perf_all = np.zeros_like(models)
    for ind in u.make_array_ind_iterator(models.shape):
        m = models[ind]
        perf = compute_task_performance(fdg, m, **kwargs)
        print(perf)
        flip_perf = compute_task_performance(fdg, m, flip_tasks=True,
                                             **kwargs)
        print(flip_perf)
        perf_all[ind] = np.mean(perf)
        flip_perf_all[ind] = np.mean(flip_perf)
    return perf_all, flip_perf_all

def test_generalization_new(dg_use=None, models_ths=None, lts_scores=None,
                            train_models_blind=False, inp_dim=2,
                            p_c=None, dg_kind=dg_kind_default,
                            dg_args=None, dg_kwargs=None, dg_source_var=1,
                            dg_train_epochs=0, models_args=None,
                            models_kwargs=None, models_log_x=True,
                            use_samples_x=True, models_n_diffs=6,
                            models_n_bounds=(2, 6.5),
                            hide_print=True, est_inp_dim=None,
                            eval_n_iters=10, use_mp=False,
                            train_test_distrs=None, n_reps=5,
                            model_kinds=model_kinds_default,
                            layer_spec=None, model_n_epochs=60,
                            plot=True, gpu_samples=False, dg_dim=500,
                            generate_data=True, n_save_samps=10**4,
                            model_batch_size=30, p_mean=True,
                            distr_type='normal', dg_layers=None,
                            compute_trained_lvs=False,
                            compute_untrained=True,
                            categ_var=None,
                            extrapolate_test=False, 
                            evaluate_intermediate=False,
                            samples_seq=None,
                            use_test_gp_tasks=None,
                            contextual_extrapolation=False):
    # train data generator
    if dg_args is None:
        out_dim = dg_dim
        if dg_train_epochs == 0:
            layers = (100, 200, 300, 100)
        else:
            layers =  (100, 200)
        if dg_layers is not None:
            layers = dg_layers
        dg_args = (inp_dim, layers, out_dim)
    if dg_kwargs is None:
        noise = .2
        reg_weight = (0, .3)
        dg_kwargs = {'noise':noise, 'l2_weight':reg_weight}
    
    if dg_use is None:
        dg_use = dg_kind(*dg_args, **dg_kwargs)

        if distr_type == 'normal':
            source_distr = sts.multivariate_normal(np.zeros(inp_dim),
                                                   dg_source_var)
        elif distr_type == 'uniform':
            bounds = (-dg_source_var, dg_source_var)
            source_distr = da.MultivariateUniform(inp_dim,
                                                  bounds)
        else:
            raise IOError('distribution type indicated ({}) is not '
                          'recognized'.format(distr_type))
            
        dg_use.fit(source_distribution=source_distr, epochs=dg_train_epochs,
                   use_multiprocessing=use_mp)
    else:
        source_distr = dg_use.source_distribution

    # test dimensionality
    pdims = dg_use.representation_dimensionality()[0]
    print('participation ratio', np.sum(pdims)**2/np.sum(pdims**2))

    flip_cat = False
    lr_flip = False
    if train_models_blind:
        train_d2 = da.HalfMultidimensionalNormal(np.zeros(inp_dim), dg_source_var)
        test_d2 = train_d2.flip()
        dg_use.source_distribution = train_d2
    elif categ_var is not None:
        train_cat = dg_use.source_distribution.make_cat_partition(
            part_frac=categ_var)
        dg_use.source_distribution = train_cat

        if extrapolate_test:
            train_d2 = train_cat.make_partition()
            test_d2 = train_d2.flip_cat_partition().flip()
            flip_cat = True
        else:
            lr_flip = True
            train_d2 = train_cat.flip_cat_partition().make_partition()
            test_d2 = train_d2.flip() 
        train_test_distrs = ((None, train_d2),
                             (None, test_d2))
        
    # train models
    if models_args is None:
        if est_inp_dim is None:
            est_inp_dim = inp_dim
        input_dims = (est_inp_dim,)*models_n_diffs
        if layer_spec is None:
            layer_spec = ((50,), (50,), (50,))
            # layer_spec = ()
        models_args = (input_dims, dg_use, model_kinds, layer_spec)
    if models_kwargs is None:
        batch_size = model_batch_size
        epochs = model_n_epochs
        train_samples = np.logspace(models_n_bounds[0], models_n_bounds[1],
                                    models_n_diffs, dtype=int)
        samps_list = True
        models_kwargs = {'batch_size':batch_size, 'epochs':epochs,
                         'samps_list':samps_list, 'n_train_samps':train_samples,
                         'use_mp':use_mp, 'hide_print':hide_print,
                         'n_reps':n_reps}
    if use_samples_x:
        use_x = models_kwargs['n_train_samps']
    else:
        use_x = models_args[0]

    print('args', models_args)
    print('kwargs', models_kwargs)
    if models_ths is None:
        models, th = train_multiple_models_dims(*models_args, **models_kwargs)
    else:
        models, th = models_ths

    if evaluate_intermediate:
        models = expand_intermediate_models(models)

    print('training done')
    if th is not None and plot:
        plot_training_progress(th, use_x)
    if plot:
        plot_model_dimensionality(dg_use, models, use_x, log_x=models_log_x)

    other = {}
    if contextual_extrapolation:
        out = compute_contextual_extrapolation(dg_use, models)
        other['contextual_extrapolation'] = out
        
    if train_test_distrs is None:
        try:
            train_d2 = dg_use.source_distribution.make_partition()
        except AttributeError:
            if distr_type == 'normal':
                train_d2 = da.HalfMultidimensionalNormal.partition(
                    dg_use.source_distribution)
            elif distr_type == 'uniform':
                train_d2 = da.HalfMultidimensionalUniform.partition(
                    dg_use.source_distribution)
            else:
                raise IOError('distribution type indicated ({}) is not '
                              'recognized'.format(distr_type))
                
        train_ds = (None, train_d2)
        test_ds = (None, train_d2.flip())
    else:
        train_ds, test_ds = train_test_distrs

    if gpu_samples:
        n_train_samples = 2*10**3
        n_test_samples = 10**3
        n_save_samps = int(n_save_samps/10)
    elif samples_seq is not None:
        n_train_samples = np.logspace(*samples_seq[0:2], int(samples_seq[2]),
                                      dtype=int)
        n_test_samples = 10**3
    else:
        n_train_samples = 2*10**3
        n_test_samples = 10**3
    n_train_samples_c = n_train_samples
    n_train_samples_r = n_train_samples_c*2
        
    print('distr set')
    if p_c is None:
        if compute_trained_lvs:
            pt, ct = evaluate_multiple_models_dims(
                dg_use, models, None, test_ds, train_distributions=train_ds,
                n_iters=eval_n_iters, n_train_samples=n_train_samples_c,
                n_test_samples=n_test_samples, mean=p_mean, learn_lvs='trained',
                gp_task_ls=use_test_gp_tasks)
            if compute_untrained:
                pu, cu = evaluate_multiple_models_dims(
                    dg_use, models, None, test_ds, train_distributions=train_ds,
                    n_iters=eval_n_iters, n_train_samples=n_train_samples_c,
                    n_test_samples=n_test_samples, mean=p_mean,
                    learn_lvs='untrained',
                    gp_task_ls=use_test_gp_tasks)
                p = np.stack((pt, pu), axis=0)
                c = np.stack((ct, cu), axis=0)
            else:
                p = pt
                c = ct
        else:
            p, c = evaluate_multiple_models_dims(
                dg_use, models, None, test_ds, train_distributions=train_ds,
                n_iters=eval_n_iters, n_train_samples=n_train_samples_c,
                n_test_samples=n_test_samples, mean=p_mean,
                gp_task_ls=use_test_gp_tasks)
    else:
        p, c = p_c

    print('p')
    print(p)
    if plot:
        plot_generalization_performance(use_x, p, log_x=models_log_x)
        plot_model_manifolds(dg_use, models)

    if lts_scores is None:
        if lr_flip:
            flip_sd = dg_use.source_distribution.flip_cat_partition()
            dg_use.source_distribution = flip_sd
        if compute_trained_lvs:
            lts_t = find_linear_mappings(
                dg_use, models, half=True, n_train_samps=n_train_samples_r,
                n_test_samps=n_test_samples,
                learn_lvs='trained', flip_cat=flip_cat)
            if compute_untrained:
                lts_u = find_linear_mappings(
                    dg_use, models, half=True, n_train_samps=n_train_samples_r,
                    n_test_samps=n_test_samples,
                    learn_lvs='untrained', flip_cat=flip_cat)
                lts_scores = list(np.stack((lts_ti, lts_u[i]), axis=0)
                                  for i, lts_ti in enumerate(lts_t))
            else:
                lts_scores = lts_t
        else:
            lts_scores = find_linear_mappings(
                dg_use, models, half=True, n_train_samps=n_train_samples_r,
                n_test_samps=n_test_samples,
                flip_cat=flip_cat)
    print(np.mean(lts_scores[1], axis=-1))
    if plot:
        plot_recon_accuracy(lts_scores[1], use_x=use_x, log_x=models_log_x)

    if generate_data:
        latent_samps, samps = dg_use.sample_reps(sample_size=n_save_samps)
        reps = np.zeros(models.shape
                        + (n_save_samps, models[0, 0, 0].encoded_size))
        ind_combs = list(list(range(d)) for d in models.shape)
        for ic in it.product(*ind_combs):
            reps[ic] = models[ic].get_representation(samps)
            print('mse', models[ic].get_reconstruction_mse(samps))
        gd = latent_samps, samps, reps
    else:
        gd = None

    return dg_use, (models, th), (p, c), lts_scores, gd, other

def model_eval(dg_use, models, eval_n_iters=10, n_train_samples=500,
               n_test_samples=500, mean=False):
    try:
        train_d2 = dg_use.source_distribution.make_partition()
    except AttributeError:
        train_d2 = da.HalfMultidimensionalNormal.partition(
            dg_use.source_distribution)
    train_ds = (None, train_d2)
    test_ds = (None, train_d2.flip())
    try:
        len(models)
    except TypeError:
        m = np.zeros((1, 1, 1), dtype=object)
        m[0, 0, 0] = models
        models = m
    p, c = evaluate_multiple_models_dims(dg_use, models, None, test_ds,
                                         train_distributions=train_ds,
                                         n_iters=eval_n_iters,
                                         n_train_samples=n_train_samples,
                                         n_test_samples=n_test_samples,
                                         mean=mean)
    lts_scores = find_linear_mappings(dg_use, models, half=True,
                                      n_samps=n_test_samples)
    return p, c, lts_scores

def plot_sidebyside(m1, m2, m_off=.1, axs=None):
    if axs is None:
        f, axs = plt.subplots(1, 2)
    c1, _, l1 = m1
    c2, _, l2 = m2
    gpl.violinplot(np.squeeze(c1), [0, 1], ax=axs[0])
    gpl.violinplot(np.squeeze(c2), [m_off, 1 + m_off], ax=axs[0])

def compute_distgen(l_samps, r_samps, dists, rads, n_reps=10, **kwargs):
    perf = np.zeros((len(dists), len(rads), n_reps))
    for (d_i, r_i, n_i) in u.make_array_ind_iterator(perf.shape):
        p = lr_pts_dist(l_samps, r_samps, dists[d_i], rads[r_i], **kwargs)
        perf[d_i, r_i, n_i] = p
    return perf

def compute_distgen_fromrun(dists, rads, run_ind, f_pattern, n_reps=10,
                            folder='disentangled/simulation_data/partition/',
                            **kwargs):
    data, info = da.load_full_run(folder, run_ind, file_template=f_pattern,
                                  analysis_only=True, skip_gd=False)
    n_parts, _, _, _, _, _, _, _, samps = data
    arg_dict = info['args'][0]
    n_trains = np.logspace(arg_dict['n_train_bounds'][0],
                           arg_dict['n_train_bounds'][1],
                           arg_dict['n_train_diffs'])
    latent_samps, dg_samps, rep_samps = samps
    if latent_samps is None:
        print('this run has no stored samples, exiting')
        perf = np.nan
    else:
        perf_all = np.zeros(rep_samps.shape[:3] + (len(dists), len(rads),
                                                   n_reps))
        for ic in u.make_array_ind_iterator(perf_all.shape[:3]):
            perf = compute_distgen(latent_samps[:, ic[1]], rep_samps[ic],
                                   dists, rads, n_reps=n_reps, **kwargs)
            perf_all[ic] = perf
    inds = (n_trains, n_parts, np.arange(perf_all.shape[2]))
    return inds, perf_all

def analyze_factor_alignment(run_ind, f_pattern, 
                             folder='disentangled/simulation_data/partition/',
                             reg=.5):
    data, info = da.load_full_run(folder, run_ind, file_template=f_pattern,
                                  analysis_only=True, skip_gd=False)
    n_parts, _, _, _, _, _, _, _, samps = data
    latent_samps, stim_samps, rep_samps = samps
    perfs = np.zeros(rep_samps.shape[:3] + (latent_samps.shape[-1],))
    angs = np.zeros_like(perfs)
    dims = np.zeros_like(perfs)
    comp_mat = np.identity(rep_samps.shape[-1])
    for ind in u.make_array_ind_iterator(rep_samps.shape[:3]):
        latents = latent_samps[:, ind[1]]
        stims = stim_samps[:, ind[1]]
        reps = rep_samps[ind]
        for i in range(latent_samps.shape[-1]):
            lr = sklm.LinearRegression()
            lr.fit(reps, latents[:, i])
            ind_i = ind + (i,)
            perfs[ind_i] = lr.score(reps, reps[:, i])
            if np.all(lr.coef_ == 0):
                sim = np.nan
            else:
                lrc = u.make_unit_vector(lr.coef_)
                sim = np.max(np.abs(u.cosine_similarity(comp_mat, lrc)))
            angs[ind_i] = sim
            dims[ind_i] = u.participation_ratio(reps)
    return n_parts, perfs, angs, dims
    

def plot_distgen_index(perfs, plot_labels, x_labels, p_ind, axs=None, fwid=3,
                       ref_ind=0, ref_ax=0):
    if axs is None:
        figsize = (fwid, fwid*len(plot_labels))
        f, axs = plt.subplots(len(plot_labels), 1, figsize=figsize)
    for i, pl in enumerate(plot_labels):
        ax = axs[i]
        pi = 1 - perfs[i, :, :, p_ind[0], p_ind[1]]
        p_ref = 1 - perfs[i, ref_ind, :, p_ind[0], p_ind[1]]
        p_index = (pi - p_ref)/(pi + p_ref)
        l = gpl.plot_trace_werr(x_labels, np.nanmean(p_index, axis=-1).T,
                                ax=ax)
        col = l[0].get_color()
        for j in range(pi.shape[1]):
            l = gpl.plot_trace_werr(x_labels, np.squeeze(p_index[:, j].T),
                                    ax=ax, color=col, line_alpha=.1)
        ax.set_ylim([-1, 1])
        gpl.add_hlines(0, ax)

def plot_distgen(perfs, plot_labels, x_labels, p_ind, axs=None, fwid=3,
                 log_y=True):
    if axs is None:
        figsize = (fwid, fwid*len(plot_labels))
        f, axs = plt.subplots(len(plot_labels), 1, figsize=figsize)
    for i, pl in enumerate(plot_labels):
        ax = axs[i]
        pi = 1 - perfs[i, :, :, p_ind[0], p_ind[1]]
        l = gpl.plot_trace_werr(x_labels, np.nanmean(pi, axis=-1).T,
                                ax=ax, log_y=log_y)
        col = l[0].get_color()
        for j in range(pi.shape[1]):
            l = gpl.plot_trace_werr(x_labels, np.squeeze(pi[:, j].T), ax=ax,
                                    color=col, line_alpha=.1)
        if not log_y:
            ax.set_ylim([-1, 1])    

def quantify_sparseness(*args, **kwargs):
    return u.quantify_sparseness(*args, **kwargs)

def centered_hyperplane_regions(dim, planes):
    ms = np.arange(dim)
    return 2*np.sum(ss.binom(planes - 1, ms))

def compute_label_means(resps, labels):
    u_labels = np.unique(labels)
    out = np.zeros((len(u_labels), resps.shape[1]))
    for i, l in enumerate(u_labels):
        l_mask = labels == l
        out[i] = np.mean(resps[l_mask], axis=0)
    return out

def predict_decoding(dim, n_tasks, n_samps=10000):
    # only works for 2D now, but works all right
    # need to test for generalization (has different terms)
    samps = np.abs(2*(sts.beta((dim - 1)/2, (dim - 1)/2).rvs(n_samps)
                      - .5))
    c = 2/np.pi
    t = np.arccos(samps)
    m_t = np.mean(t)
    v_t = np.mean(t**2)
    std_t = np.std(t)/np.sqrt(n_tasks)
    
    t_agree = (m_t - c*v_t)/(1 - c*m_t)
    t_dis = c*v_t/(c*m_t)
    p_agree = 1 - c*m_t
    r = p_agree*(1 - c*t_agree) - (1 - p_agree)*(1 - c*t_dis)

    # not sure why not mult by tasks here but it works well not to
    # (maybe something to do with correlation between additional tasks
    # in the low-d space
    # probably need to account for it in more detail
    # mult tasks also probably have diff normalization than doing now
    err = sts.norm(0, 1).cdf(-r/std_t)

    
    r_gen = 0
    gen_err = sts.norm(0, 1).cdf(-r_gen/std_t)
    return 1 - err
    
def sample_planes(dim, n_samps=1000, lv_samps=10000):
    lv_samps = sts.norm(0, 1).rvs((lv_samps, dim))
    tfs, samp, _ = dd.make_tasks(dim, n_samps)
    
    task_out_samps = np.stack(list(tf_i(lv_samps) for tf_i in tfs), axis=1)
    task_out_samps[task_out_samps < .5] = -1
    (nov_func,), nov_vec, _ = dd.make_tasks(dim, 1)
    (gen_func,), gen_vec, _ = da.generate_partition_functions(dim, n_funcs=1,
                                                              orth_vec=nov_vec)
    flip_mask = np.sum(samp*nov_vec, axis=1) < 0
    mult_mask = np.where(flip_mask, -1, 1)
    task_out_samps = task_out_samps*mult_mask
    samp = samp*np.expand_dims(mult_mask, 1)

    tr_mask = gen_func(lv_samps) == 0
    te_mask = gen_func(lv_samps) == 1
    
    mask_c1 = nov_func(lv_samps) == 0
    mask_c2 = nov_func(lv_samps) == 1

    task_tr_c1 = task_out_samps[np.logical_and(tr_mask, mask_c1)]
    task_tr_c2 = task_out_samps[np.logical_and(tr_mask, mask_c2)]

    task_te_c1 = task_out_samps[np.logical_and(te_mask, mask_c1)]
    task_te_c2 = task_out_samps[np.logical_and(te_mask, mask_c2)]

    
    m1 = sklm.Ridge()
    # m = skc.LinearSVC()
    task_tr = np.concatenate((task_tr_c1, task_tr_c2), axis=0)
    task_labels = np.concatenate((np.ones(len(task_tr_c1)),
                                  np.ones(len(task_tr_c2))*-1),
                                 axis=0)
    m1.fit(task_tr, task_labels)
    emp1_err = np.mean(m1.predict(task_tr_c1) > 0)
    emp2_err = np.mean(m1.predict(task_tr_c2) < 0)
    emp_err = np.mean([emp1_err, emp2_err])
    # print(emp_err, np.mean(m.predict(task_tr_c2) < 0))
    gen_err = np.mean(m1.predict(task_te_c1) > 0)

    m2 = sklm.Ridge()
    regr_targ = np.sum(lv_samps*nov_vec, axis=1)
    m2.fit(task_out_samps[tr_mask], regr_targ[tr_mask])
    print(m2.score(task_out_samps[tr_mask], regr_targ[tr_mask]),
          m2.score(task_out_samps[te_mask], regr_targ[te_mask]))
    
    angs = list(np.sum(samp*v, axis=1)
                for v in (nov_vec, gen_vec))
    return (samp, angs, (emp_err, gen_err), (task_tr_c1, task_tr_c2),
            (task_te_c1, task_te_c2), (nov_func, gen_func))

def correlation_length(dim, planes, n_samps=10000, thr=0,
                       cond_dim=1, dec_dim=0, n_bins=10, extent=2,
                       grid_pts=100,
                       eps=1e-10):
    samps = sts.norm(0, 1).rvs((grid_pts**2, dim))
    grid = np.array(list(it.product(np.linspace(-extent, extent, grid_pts),
                                    repeat=2)))
    samps[:, (dec_dim, cond_dim)] = grid
    funcs, _, _ = dd.make_tasks(dim, planes)
    resps = np.stack(list(f(samps) for f in funcs), axis=1)

    tr_mask =  samps[:, cond_dim] < thr
    te_mask = np.logical_not(tr_mask)

    tr_samps = samps[tr_mask]
    te_samps = samps[te_mask]

    tr_resps = resps[tr_mask]
    te_resps = resps[te_mask]

    digi_bins = np.linspace(-extent - eps, extent + eps, n_bins + 1)
    labels = np.digitize(samps[:, dec_dim], digi_bins)

    ref_means = compute_label_means(tr_resps, labels[tr_mask])
    
    dist_digi_bins = np.linspace(thr, extent + eps, n_bins + 1)
    dists = dist_digi_bins[:-1] + np.diff(dist_digi_bins)[0]/2 
    dist_labels = np.digitize(te_samps[:, cond_dim], dist_digi_bins)
    u_dist_labels = np.unique(dist_labels)
    
    te_labels = labels[te_mask]

    out = np.zeros((n_bins, n_bins, planes))
    corr = np.zeros((n_bins, planes))
    for i, dl in enumerate(u_dist_labels):
        dist_label_mask = dist_labels == dl
        lresps = te_resps[dist_label_mask]
        
        out[i] = compute_label_means(lresps, te_labels[dist_label_mask])
        corr[i] = list(sts.pearsonr(ref_means[:, j], out[i, :, j])[0]
                       for j in range(planes))
    corr[np.isnan(corr)] = 0
    return ref_means, dists, out, corr

def centered_hyperplane_regions_empirical(dim, planes, n_samps=100000):
    samps = sts.norm(0, 1).rvs((n_samps, dim))
    funcs, _, _ = dd.make_tasks(dim, planes)
    resps = np.stack(list(f(samps) for f in funcs), axis=1)
    u_resps = np.unique(resps, axis=0)
    return u_resps.shape[0]

def plot_recon_gen_summary(run_ind, f_pattern, fwid=3, log_x=True,
                           dg_type=dg.FunctionalDataGenerator,
                           model_type=dd.FlexibleDisentanglerAE, axs=None,
                           folder='disentangled/simulation_data/partition/',
                           ret_info=False, collapse_plots=False,  pv_mask=None,
                           xlab='tasks', ret_fig=False, legend='',
                           print_args=True, set_title=True, color=None,
                           plot_hline=True, distr_parts=None, linestyle='solid',
                           double_ind=None, set_lims=True,
                           intermediate=False, list_run_ind=False,
                           multi_train=False, label_field=None,
                           inter_labels=None, inter_colors=None,
                           plot_intermediate=True,
                           rl_model=False,
                           **kwargs):
    if double_ind is not None:
        merge_axis = 2
    else:
        merge_axis = 1
    if list_run_ind:
        all_p, all_sc = [], []
        for i, ri in enumerate(run_ind): 
            data, info = da.load_full_run(folder, ri, merge_axis=merge_axis,
                                          dg_type=dg_type, model_type=model_type,
                                          file_template=f_pattern, analysis_only=True,
                                          multi_train=multi_train, **kwargs)
            n_parts, _, _, _, p, c, _, sc, _ = data[:9]
            if rl_model:
                p = p[0]
                sc = sc[0]
            all_p.append(p)
            all_sc.append(sc)
        p = np.concatenate(all_p, axis=2)
        sc = np.concatenate(all_sc, axis=2)
    else:
        data, info = da.load_full_run(folder, run_ind, merge_axis=merge_axis,
                                      dg_type=dg_type, model_type=model_type,
                                      file_template=f_pattern, analysis_only=True,
                                      multi_train=multi_train, **kwargs) 
        n_parts, _, _, _, p, c, _, sc, _ = data[:9]
        if len(legend) == 0 and label_field is not None:
            legend = '{:1.0e}'.format(info['args'][0][label_field])

    if multi_train:
        sc = np.moveaxis(sc, -2, 1)
    if 'beta_mult' in info['args'][0].keys():
        n_parts = np.array(n_parts)*info['args'][0]['beta_mult']
    if ('l2pr_weights_mult' in info['args'][0].keys()
        and info['args'][0]['l2pr_weights'] is not None):
        n_parts = np.array(n_parts)*info['args'][0]['l2pr_weights_mult']*100
    if print_args:
        print(info['args'][0])
    p = p[..., 1]
    if double_ind is not None and not rl_model:
        p = p[double_ind]
        sc = sc[double_ind]
    if distr_parts is not None:
        n_parts = np.array(list(info_i[distr_parts] for info_i in info['args']))
        sort_inds = np.argsort(n_parts)
        n_parts = n_parts[sort_inds]
        p = p[:, sort_inds]
        sc = sc[:, sort_inds]
    if rl_model:
        panel_vals = info.get('initial_collects')
        if panel_vals is None:
            panel_vals = (1000, 5000, 10000)
        panel_vals = np.array(panel_vals)
    else:
        panel_vals = np.logspace(*info['training_eg_args'], dtype=int)
    if pv_mask is not None:
        panel_vals = panel_vals[pv_mask]
        p = p[pv_mask]
        sc = sc[pv_mask]
    if set_lims:
        ylims = ((.5, 1), (0, 1))
    else:
        ylims = None
    thresh = (.9, .8)
    out = plot_recon_gen_summary_data((p, sc), n_parts, ylims=ylims,
                                      labels=('classifier\ngeneralization',
                                              'regression\ngeneralization'),
                                      info=info, log_x=log_x,
                                      panel_vals=panel_vals, xlab=xlab,
                                      axs=axs, collapse_plots=collapse_plots,
                                      ret_fig=ret_fig, label=legend,
                                      fwid=fwid, set_title=set_title,
                                      color=color, plot_hline=plot_hline,
                                      linestyle=linestyle, thresh=thresh,
                                      intermediate=intermediate,
                                      inter_labels=inter_labels,
                                      inter_colors=inter_colors,
                                      plot_intermediate=plot_intermediate)
    if ret_info:
        out_all = (out, info)
    else:
        out_all = out
    return out_all

def plot_recon_gen_summary_data(quants_plot, x_vals, panel_vals=None,
                                ylims=None, labels=None, x_ax=1, panel_ax=0,
                                fwid=3, info=None, log_x=True, label='',
                                panel_labels='train egs = {}',
                                xlab='partitions', axs=None, ct=np.nanmedian,
                                collapse_plots=False, ret_fig=False,
                                set_title=True, color=None, thresh=(.9,),
                                plot_hline=True, **kwargs):
    if len(thresh) < len(quants_plot):
        thresh = (thresh[0],)*len(quants_plot)
    n_plots = len(quants_plot)
    n_panels = quants_plot[0].shape[panel_ax]
    if ylims is None:
        ylims = (None,)*n_plots
    if labels is None:
        labels = ('',)*n_plots
    if panel_vals is None:
        panel_vals = np.logspace(2, 6.5, quants_plot[0].shape[panel_ax],
                                 dtype=int)

    fsize = (n_plots*fwid, fwid*n_panels)
    if axs is None:
        if collapse_plots:
            f, axs = plt.subplots(1, n_plots, figsize=(n_plots*fwid, fwid),
                                  squeeze=False)
        else:
            f, axs = plt.subplots(n_panels, n_plots, figsize=fsize,
                                  squeeze=False)

    for i, qp in enumerate(quants_plot):
        if info is not None and plot_hline:
            if 'args' in info.keys() and  'betas' in info['args'][0].keys():
                nd = 1
            else:
                nd = info.get('input_dimensions', None)
        else:
            nd = None
        if x_ax == 0 and panel_ax == 1:
            qp = np.swapaxes(qp, 0, 1)
        else:
            qp = np.swapaxes(qp, x_ax, 1)
            qp = np.swapaxes(qp, panel_ax, 0)
        if i < (len(quants_plot) - 1):
            use_label = ''
        else:
            use_label = label
        axs_i = plot_recon_accuracies_ntrain(qp, central_tendency=ct,
                                             axs=axs[:, i], xs=x_vals,
                                             log_x=log_x, n_plots=panel_vals,
                                             ylabel=labels[i], ylim=ylims[i],
                                             num_dims=nd, xlab=xlab,
                                             label=use_label, color=color,
                                             collapse_plots=collapse_plots,
                                             plot_labels=panel_labels,
                                             set_title=set_title, **kwargs)
        print_thresh_exceed(x_vals, qp, thresh[i], labels[i], super_label=label)
    if ret_fig:
        out = f, axs
    else:
        out = axs
    return out

def print_thresh_exceed(xs, ys, thresh, label, super_label='',
                        s='{} exceeded {} at {} x-axis',
                        prefix='{}: ', y_ind=0):
    if ys.shape[0] > 1:
        print('multiple ys, taking {} ind'.format(y_ind))
    inds = np.where(np.nanmean(ys[y_ind], axis=1) > thresh)[0]
    if len(inds) > 0:
        pt = xs[inds[0]]
        sp = s.format(label, thresh, pt)
    else:
        sp = '{} did not exceed threshold'.format(label)
    if len(super_label) > 0:
        spre = prefix.format(super_label)
        sp = spre + sp
    sp = sp.replace('\n', ' ')
    print(sp)
    return sp                 

def _create_samps(vals, dim, others):
    out = np.zeros((len(vals), len(others) + 1))
    for i, v in enumerate(vals):
        out[i, :dim] = others[:dim]
        out[i, dim] = v
        out[i, dim+1:] = others[dim:]
    return out

def _gen_perturbed(autoenc, center_img, vec, full_perturb, n_perts):    
    center_rep = autoenc.get_representation(center_img)
    perts = np.linspace(-full_perturb, full_perturb, n_perts)
    dev = np.expand_dims(perts, 0)*np.expand_dims(vec, 1)
    pert_reps = (np.expand_dims(center_rep[0], 1)
                 + dev)
    # print(lr_m.predict(pert_reps.T))
    pert_recons = autoenc.get_reconstruction(pert_reps.T)
    return pert_recons

def plot_traversal_plot(gen, autoenc, trav_dim=1, learn_dim=0, full_perturb=1,
                        n_pts=1000, lr=sklm.Ridge, n_perts=5,
                        leave_const=(2,), test_shape=0,
                        **kwargs):
    ld = gen.img_params[learn_dim]
    ld_u = np.unique(gen.data_table[ld])
    # np.random.shuffle(ld_u)
    test_ld = ld_u[test_shape]
    inds = list(np.arange(len(ld_u), dtype=int))
    test_ind = inds.pop(test_shape)
    test_ld = ld_u[test_ind]
    train_ld = ld_u[np.array(inds)]
    train_ld = ld_u[1:]
    train_xs, train_imgs = gen.sample_reps(sample_size=n_pts)
    center_all = gen.get_center()
    if leave_const is not None:
        train_xs[:, leave_const] = center_all[leave_const]
        train_imgs = gen.get_representation(train_xs)
    mask = np.isin(train_xs[:, learn_dim], train_ld)
    train_xs, train_imgs = train_xs[mask], train_imgs[mask]
    train_reps = autoenc.get_representation(train_imgs)
    train_recons = autoenc.get_reconstruction(train_reps)
    lr_m = lr()
    lr_m.fit(train_reps, train_xs[:, trav_dim])
    lr_norm = u.make_unit_vector(lr_m.coef_)
    center_test = gen.get_center()
    center_test[learn_dim] = test_ld
    center_test_img = gen.get_representation(center_test)
    center_in = gen.get_center()
    center_in[learn_dim] = train_ld[0]
    center_in_img = gen.get_representation(center_in)

    pert_recons = _gen_perturbed(autoenc, center_test_img, lr_norm,
                                 full_perturb, n_perts)
    pert_recons_in = _gen_perturbed(autoenc, center_in_img, lr_norm,
                                    full_perturb, n_perts)
    
    return pert_recons, train_imgs, train_reps, pert_recons_in, lr_m
    

# def plot_traversal_plot(gen, autoenc, trav_dim=0, axs=None, n_pts=5,
#                         other_vals=None, eps_d=.1, reps=20,
#                         n_dense_pts=10, full_perturb=1, learn_dim=None):
#     if other_vals is None:
#         other_vals = list(gen.get_center())
#         other_vals.pop(trav_dim)
#     dense_samp = np.linspace(.5 - eps_d, .5 + eps_d, n_dense_pts)
#     dense_pts = list(gen.ppf(ds, trav_dim) for ds in dense_samp)
#     dense_xs = _create_samps(dense_pts, trav_dim, other_vals)
#     print(dense_xs)
#     dense_pts_all = dense_pts*reps
#     dense_latents_all = []
#     for i in range(reps):
#         dense_imgs = gen.get_representation(dense_xs, same_img=True)
#         dense_latents = autoenc.get_representation(dense_imgs)
#         dense_latents_all.append(dense_latents)
#     dense_latents_all = np.concatenate(dense_latents_all, axis=0)
#     dense_recons = autoenc.get_reconstruction(dense_latents)
#     lr = sklm.Ridge()
#     lr.fit(dense_latents_all, dense_pts_all)
#     lr_val = lr.score(dense_latents, dense_pts)
#     perts = np.linspace(-full_perturb, full_perturb, n_pts)
#     if learn_dim is not None:
#         vals = np.unique(gen.data_table[gen.img_params[learn_dim]])
#         c_val = gen.ppf(.5, learn_dim)
#         vals = vals[vals != c_val]
#         interp_val = np.random.choice(vals, 1)
        
#         ind = int(n_dense_pts/2)
#         center_x = dense_xs[ind]
#         center_x[learn_dim] = interp_val
#         center_img = gen.get_representation(center_x)
#         center_rep = autoenc.get_representation(center_img)[0]
#     else:
#         center_rep = dense_latents[int(n_dense_pts/2)]
#     lr_norm = u.make_unit_vector(lr.coef_)
#     dev = np.expand_dims(perts, 0)*np.expand_dims(lr_norm, 1)
#     pert_reps = (np.expand_dims(center_rep, 1)
#                  + dev)
#     print(lr.predict(pert_reps.T))
#     pert_recons = autoenc.get_reconstruction(pert_reps.T)
#     return pert_recons, dense_imgs, dense_latents, dense_recons, lr

def plot_img_series(imgs, fwid=4, axs=None, lines=False, title='',
                    **kwargs):
    fwid = 4
    fsize = (fwid*imgs.shape[0], fwid)
    if axs is None: 
        f, axs = plt.subplots(1, imgs.shape[0], figsize=fsize)
        f.suptitle(title)
    else:
        axs[0].set_title(title)
    for i in range(imgs.shape[0]):
        axs[i].imshow(imgs[i], **kwargs)
        if lines:
            gpl.add_hlines(imgs.shape[1]/2, axs[i])
            gpl.add_vlines(imgs.shape[2]/2, axs[i])
        axs[i].set_xticks([])
        axs[i].set_yticks([])
            
