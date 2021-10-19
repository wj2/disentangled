import numpy as np
import scipy.stats as sts
import pickle
import os
import itertools as it
import tensorflow as tf
import scipy.linalg as spla

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
                              learn_lvs='ignore',
                              repl_mean=None, **classifier_params):
    if train_func is None:
        lv_mask = np.ones(gen.input_dim, dtype=bool)
        if use_orthogonal and hasattr(train_distrib, 'partition'):
            orth_vec = train_distrib.partition
            orth_off = train_distrib.offset
            if learn_lvs == 'ignore':
                out = da.generate_partition_functions(gen.input_dim,
                                                      n_funcs=n_iters,
                                                      orth_vec=orth_vec,
                                                      orth_off=orth_off)
            elif learn_lvs == 'trained':
                lv_mask = vae.learn_lvs
                out = da.generate_partition_functions(
                    sum(vae.learn_lvs), n_funcs=n_iters,
                    orth_vec=orth_vec[lv_mask], orth_off=orth_off)
            elif learn_lvs == 'untrained':
                n_untrained = gen.input_dim - sum(vae.learn_lvs)
                lv_mask = np.logical_not(vae.learn_lvs)
                out = da.generate_partition_functions(
                    n_untrained, n_funcs=n_iters, orth_vec=orth_vec[lv_mask],
                    orth_off=orth_off)
            else:
                raise IOError('{} is not an understood option for '
                              'learn_lvs'.format(learn_lvs))
        else:
            out = da.generate_partition_functions(gen.input_dim, n_funcs=n_iters)
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
        train_samples = train_distrib.rvs(n_train_samples)
        if repl_mean is not None:
            train_samples[:, repl_mean] = train_distrib.mean[repl_mean]
        train_labels = train_func[i](train_samples[:, lv_mask])
        inp_reps = gen.generator(train_samples)
        train_rep = vae.get_representation(inp_reps)
        print('tr', train_rep)
        print('tl', train_labels)
        if not np.any(np.isnan(train_rep)):
            c = classifier(max_iter=100000, **classifier_params)
            ops = [skp.StandardScaler(), c]
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

def evaluate_multiple_models_dims(dg_use, models, *args, **kwargs):
    ps, cs = [], []
    for m in models:
        p, c = evaluate_multiple_models(dg_use, m, *args, **kwargs)
        ps.append(p)
        cs.append(c)
    return np.array(ps), np.array(cs)

def train_multiple_models(dg_use, model_kinds, layer_spec, n_reps=10, batch_size=32,
                          n_train_samps=10**6, epochs=5, hide_print=False,
                          input_dim=None, use_mp=False, standard_loss=True,
                          val_set=True, save_and_discard=False,
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
            m = mk(dg_use.output_dim, layer_spec, input_dim, **kwargs)
            if hide_print:
                with u.HiddenPrints():
                    th = m.fit_sets(train_set, eval_set=eval_set, epochs=epochs,
                                    batch_size=batch_size,
                                    standard_loss=standard_loss,
                                    use_multiprocessing=use_mp)
            else:
                th = m.fit_sets(train_set, eval_set=eval_set, epochs=epochs,
                                    batch_size=batch_size,
                                    standard_loss=standard_loss,
                                    use_multiprocessing=use_mp)
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
    for i in range(models.shape[0]):
        for j in range(models.shape[1]):
            bvae = models[i, j]
            for k, td in enumerate(test_distributions):
                train_d_k = train_distributions[k]
                out = classifier_generalization(dg_use, bvae, train_func,
                                                test_distrib=td,
                                                train_distrib=train_d_k,
                                                mean=mean, n_iters=n_iters,
                                                **classifier_args)
                performance[i, j, k] = out[0]
                chance[i, j, k]= out[1]
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
                         source_view_init=None, model_view_init=None, **kwargs):
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
        axs[0].set_xlabel('feature 1 (au)')
        axs[0].set_ylabel('feature 2 (au)')
    axs[ind].set_xlabel('PCA 1 (au)')
    axs[ind].set_ylabel('PCA 2 (au)')
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

def plot_diagnostics(dg_use, model, rs, n_arcs, ax=None, n=1000, dim_red=True,
                     n_dim_red=10**4, pt_size=2, line_style='solid',
                     markers=True, line_alpha=1, use_arc_dim=False,
                     use_circ_dim=False, arc_col=(.8, .8, .8), scale_mag=.5,
                     fwid=2.5, set_inds=(0, 1), plot_partitions=False,
                     plot_source=False, square=True, start_vals=(0,),
                     supply_range=1, plot_3d=False, dim_red_func=None,
                     compute_pr=False, ret_dim_red=False, buff=0,
                     model_trs=_model_pca, view_init=None, **pca_args):
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
                        alpha=line_alpha, color=arc_col)
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
                    linestyle=line_style, alpha=line_alpha)
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
            ax.plot(xs + v_o[0], ys + v_o[1], color='r')

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
                                 set_title=True, **kwargs):
    if len(scores.shape) == 4:
        scores = np.mean(scores, axis=3)
    n_ds, n_mks, n_reps = scores.shape
    if axs is None:
        f, axs = plt.subplots(n_ds, 1, sharey=True,
                              figsize=(fwid, n_ds*fwid))
    for i, sc in enumerate(scores):
        title = plot_labels.format(n_plots[i])
        if collapse_plots:
            plot_ind = 0
            legend = title
            kwargs['label'] = legend
        else:
            plot_ind = i
        if not set_title and collapse_plots:
            kwargs['label'] = ''
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

def find_linear_mappings(dg_use, model_arr, n_samps=10**4, half_ns=100, half=True,
                         **kwargs):
    inds = it.product(*(range(x) for x in model_arr.shape))
    if half:
        scores_shape = model_arr.shape + (half_ns,)
    else:
        scores_shape = model_arr.shape
    scores = np.zeros(scores_shape, dtype=float)
    sims = np.zeros_like(scores, dtype=object)
    lintrans = np.zeros(model_arr.shape + (2,), dtype=object)
    for ind in inds:
        lr, sc, sim = find_linear_mapping(dg_use, model_arr[ind], n_samps=n_samps,
                                          **kwargs)
        scores[ind] = sc
        lintrans[ind] = lr
        sims[ind] = sim
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

def find_linear_mapping_single(dg_use, model, n_samps=10**4, half=True,
                               get_parallelism=True, train_stim_set=None,
                               train_labels=None, test_stim_set=None,
                               test_labels=None, feat_mask=None,
                               lr_type=sklm.Ridge,
                               correct=False, repl_mean=None,
                               partition_vec=None, learn_lvs='ignore',
                               **kwargs):
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
            stim = src.rvs(n_samps)
            if repl_mean is not None:
                stim[:, repl_mean] = src.mean[repl_mean]
        else:
            stim = dg_use.source_distribution.rvs(n_samps)
            if repl_mean is not None:
                stim[:, repl_mean] = dg_use.source_distribution.mean[repl_mean]

        enc_pts = model.get_representation(dg_use.generator(stim))
        if half:
            flipped = src.flip()
            test_stim = flipped.rvs(n_samps)
        else:
            test_stim = dg_use.source_distribution.rvs(n_samps)
        test_enc_pts = model.get_representation(dg_use.generator(test_stim))
    if feat_mask is None:
        feat_mask = np.ones(stim.shape[1], dtype=bool)
    lr = lr_type(**kwargs)
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
                            compute_untrained=True):
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

    if train_models_blind:
        train_d2 = da.HalfMultidimensionalNormal(np.zeros(inp_dim), dg_source_var)
        test_d2 = train_d2.flip()
        dg_use.source_distribution = train_d2

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
    
    if models_ths is None:
        models, th = train_multiple_models_dims(*models_args, **models_kwargs)
    else:
        models, th = models_ths

    if th is not None and plot:
        plot_training_progress(th, use_x)
    if plot:
        plot_model_dimensionality(dg_use, models, use_x, log_x=models_log_x)

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
        train_ds, test_ds = train_test_distr

    if gpu_samples:
        n_train_samples = 2*10**3
        n_test_samples = 10**3
        n_save_samps = int(n_save_samps/10)
    else:
        n_train_samples = 2*10**3
        n_test_samples = 10**3
        
    if p_c is None:
        if compute_trained_lvs:
            pt, ct = evaluate_multiple_models_dims(
                dg_use, models, None, test_ds, train_distributions=train_ds,
                n_iters=eval_n_iters, n_train_samples=n_train_samples,
                n_test_samples=n_test_samples, mean=p_mean, learn_lvs='trained')
            if compute_untrained:
                pu, cu = evaluate_multiple_models_dims(
                    dg_use, models, None, test_ds, train_distributions=train_ds,
                    n_iters=eval_n_iters, n_train_samples=n_train_samples,
                    n_test_samples=n_test_samples, mean=p_mean,
                    learn_lvs='untrained')
                p = np.stack((pt, pu), axis=0)
                c = np.stack((ct, cu), axis=0)
            else:
                p = pt
                c = ct
        else:
            p, c = evaluate_multiple_models_dims(dg_use, models, None, test_ds,
                                                 train_distributions=train_ds,
                                                 n_iters=eval_n_iters,
                                                 n_train_samples=n_train_samples,
                                                 n_test_samples=n_test_samples,
                                                 mean=p_mean)
    else:
        p, c = p_c

    print('p')
    print(p)
    if plot:
        plot_generalization_performance(use_x, p, log_x=models_log_x)
        plot_model_manifolds(dg_use, models)

    if lts_scores is None:
        if compute_trained_lvs:
            lts_t = find_linear_mappings(dg_use, models, half=True,
                                         n_samps=n_test_samples,
                                         learn_lvs='trained')
            if compute_untrained:
                lts_u = find_linear_mappings(dg_use, models, half=True,
                                             n_samps=n_test_samples,
                                             learn_lvs='untrained')
                lts_scores = list(np.stack((lts_ti, lts_u[i]), axis=0)
                                  for i, lts_ti in enumerate(lts_t))
            else:
                lts_scores = lts_t
        else:
            lts_scores = find_linear_mappings(dg_use, models, half=True,
                                              n_samps=n_test_samples)
    print(lts_scores[1])
    print(np.mean(lts_scores[1]))
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

    return dg_use, (models, th), (p, c), lts_scores, gd

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

def plot_recon_gen_summary(run_ind, f_pattern, fwid=3, log_x=True,
                           dg_type=dg.FunctionalDataGenerator,
                           model_type=dd.FlexibleDisentanglerAE, axs=None,
                           folder='disentangled/simulation_data/partition/',
                           ret_info=False, collapse_plots=False,  pv_mask=None,
                           xlab='tasks', ret_fig=False, legend='',
                           print_args=True, set_title=True, color=None,
                           plot_hline=True, distr_parts=None, linestyle='solid',
                           double_ind=None, **kwargs):
    if double_ind is not None:
        merge_axis = 2
    else:
        merge_axis = 1
    data, info = da.load_full_run(folder, run_ind, merge_axis=merge_axis,
                                  dg_type=dg_type, model_type=model_type,
                                  file_template=f_pattern, analysis_only=True,
                                  **kwargs) 
    n_parts, _, _, _, p, c, _, sc, _ = data
    if 'beta_mult' in info['args'][0].keys():
        n_parts = np.array(n_parts)*info['args'][0]['beta_mult']
    if ('l2pr_weights_mult' in info['args'][0].keys()
        and info['args'][0]['l2pr_weights'] is not None):
        n_parts = np.array(n_parts)*info['args'][0]['l2pr_weights_mult']*100
    if print_args:
        print(info['args'][0])
    p = p[..., 1]
    print(p.shape, sc.shape)
    if double_ind is not None:
        p = p[double_ind]
        sc = sc[double_ind]
    if distr_parts is not None:
        n_parts = np.array(list(info_i[distr_parts] for info_i in info['args']))
        sort_inds = np.argsort(n_parts)
        n_parts = n_parts[sort_inds]
        p = p[:, sort_inds]
        sc = sc[:, sort_inds]
    panel_vals = np.logspace(*info['training_eg_args'], dtype=int)
    if pv_mask is not None:
        panel_vals = panel_vals[pv_mask]
        p = p[pv_mask]
        sc = sc[pv_mask]
    ylims = ((.5, 1), (0, 1))
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
                                      linestyle=linestyle, thresh=thresh)
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
        ylims = ((0, 1),)*n_plots
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
                        leave_const=(2,),
                        **kwargs):
    ld = gen.img_params[learn_dim]
    ld_u = np.unique(gen.data_table[ld])
    np.random.shuffle(ld_u)
    test_ld = ld_u[0]
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
            
