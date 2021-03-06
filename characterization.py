import numpy as np
import scipy.stats as sts
import pickle
import os
import itertools as it
import tensorflow as tf

import sklearn.decomposition as skd
import sklearn.svm as skc
import sklearn.linear_model as sklm

import general.utility as u
import general.plotting as gpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import disentangled.aux as da
import disentangled.data_generation as dg
import disentangled.disentanglers as dd

    
def classifier_generalization(gen, vae, train_func=None, train_distrib=None,
                              test_distrib=None, test_func=None,
                              n_train_samples=2*10**4, n_test_samples=10**4,
                              classifier=skc.SVC, kernel='linear', n_iters=10,
                              shuffle=False, use_orthogonal=True, 
                              **classifier_params):
    if train_func is None:
        if use_orthogonal and hasattr(train_distrib, 'partition'):
            orth_vec = train_distrib.partition
            orth_off = train_distrib.offset
            out = da.generate_partition_functions(gen.input_dim, n_funcs=n_iters,
                                                  orth_vec=orth_vec,
                                                  orth_off=orth_off)
        else:
            out = da.generate_partition_functions(gen.input_dim, n_funcs=n_iters)
        train_func, _, _ = out
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
        train_labels = train_func[i](train_samples)
        inp_reps = gen.generator(train_samples)
        train_rep = vae.get_representation(inp_reps)
        c = classifier(kernel=kernel, **classifier_params)
        c.fit(train_rep, train_labels)

        test_samples = test_distrib.rvs(n_test_samples)
        test_labels = test_func[i](test_samples)
        if shuffle:
            snp.random.shuffle(test_labels)

        test_rep = vae.get_representation(gen.generator(test_samples))
        scores[i] = c.score(test_rep, test_labels)
        chances[i] = .5 
    return np.mean(scores), np.mean(chances)

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
                          input_dim=None, use_mp=False, standard_loss=False,
                          val_set=True, **kwargs):
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
            training_history[i, j] = th
            models[i, j] = m
    return models, training_history


def evaluate_multiple_models(dg_use, models, train_func, test_distributions,
                             train_distributions=None, **classifier_args):
    performance = np.zeros(models.shape + (len(test_distributions),))
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
               use_circ_dim=False, **pca_args):
    if use_arc_dim:
        distrib_pts = np.zeros((n_dim_red, dg_use.input_dim))
        x0_pts = np.linspace(0, dg_use.source_distribution.cov[0, 0], n_dim_red)
        x1_pts = np.linspace(0, dg_use.source_distribution.cov[1, 1], n_dim_red)
        distrib_pts[:, 0] = x0_pts
        distrib_pts[:, 1] = x1_pts
    elif use_circ_dim:
        r = np.sqrt(dg_use.source_distribution.cov[0, 0])
        distrib_pts = _get_circle_pts(n_dim_red, dg_use.input_dim, r=r)
    else:
        distrib_pts = dg_use.source_distribution.rvs(n_dim_red)
    distrib_reps = dg_use.get_representation(distrib_pts)
    mod_distrib_reps = model.get_representation(distrib_reps)
    p = skd.PCA(**pca_args)
    p.fit(mod_distrib_reps)
    return p

def _get_circle_pts(n, inp_dim, r=1):
    angs = np.linspace(0, 2*np.pi, n)
    pts = np.stack((np.cos(angs), np.sin(angs),) +
                   (np.zeros_like(angs),)*(inp_dim - 2), axis=1)
    return r*pts

def plot_diagnostics(dg_use, model, rs, n_arcs, ax=None, n=1000, dim_red=True,
                     n_dim_red=10**4, pt_size=2, line_style='solid',
                     markers=True, line_alpha=.5, use_arc_dim=False,
                     use_circ_dim=False, arc_col=(.8, .8, .8), scale_mag=.5,
                     fwid=2.5, set_inds=(0, 1), **pca_args):
    if ax is None:
        f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))

    angs = np.linspace(0, 2*np.pi, n)
    pts = np.zeros((n, dg_use.input_dim))
    pts[:, set_inds[0]] = np.cos(angs)
    pts[:, set_inds[1]] = np.sin(angs)
    
    if dim_red:
        p = _model_pca(dg_use, model, n_dim_red=n_dim_red, use_arc_dim=use_arc_dim,
                       use_circ_dim=use_circ_dim, **pca_args)
        
    if n_arcs > 0:
        skips = int(np.round(n/n_arcs))
        sub_pts = rs[-1]*pts[::skips]
        y = np.expand_dims(np.linspace(0, 1, n), 1)
        for sp in sub_pts:
            sp = np.expand_dims(sp, 0)
            s_reps = dg_use.generator(sp*y)
            mod_reps = model.get_representation(s_reps)
            if dim_red:
                mod_reps = p.transform(mod_reps)
            l = ax.plot(mod_reps[:, 0], mod_reps[:, 1], linestyle=line_style,
                        alpha=line_alpha, color=arc_col)
            if markers:
                ax.plot(mod_reps[:, 0], mod_reps[:, 1], 'o', markersize=pt_size,
                        color=l[0].get_color())

    for r in rs:
        s_reps = dg_use.generator(r*pts)
        mod_reps = model.get_representation(s_reps)
        if dim_red:
            mod_reps = p.transform(mod_reps)
        l = gpl.plot_trace_werr(mod_reps[:, 0], mod_reps[:, 1], ax=ax,
                                linestyle=line_style, alpha=line_alpha)
        if markers:
            ax.plot(mod_reps[:, 0], mod_reps[:, 1], 'o', markersize=pt_size,
                    color=l[0].get_color())


    ax.set_aspect('equal')
    gpl.make_xaxis_scale_bar(ax, scale_mag)
    gpl.make_yaxis_scale_bar(ax, scale_mag)
    return ax

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
                model_dims[i, j, k] = p.n_components_
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
                                 xlab='partitions', **kwargs):
    if len(scores.shape) == 4:
        scores = np.mean(scores, axis=3)
    n_ds, n_mks, n_reps = scores.shape
        
    if axs is None:
        f, axs = plt.subplots(n_ds, 1, sharey=True,
                              figsize=(fwid, n_ds*fwid))
    for i, sc in enumerate(scores):
        plot_recon_accuracy_partition(sc, ax=axs[i], mks=xs, **kwargs)
        axs[i].set_ylabel(ylabel)
        if n_plots is not None and len(plot_labels) > 0:
            axs[i].set_title(plot_labels.format(n_plots[i]))
        if ylim is not None:
            axs[i].set_ylim(ylim)
        if num_dims is not None:
            gpl.add_vlines(num_dims, axs[i])
    axs[i].set_xlabel(xlab)
    return axs        

def plot_recon_accuracy_partition(scores, mks=None, ax=None, indiv_pts=True,
                                  log_x=False, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    n_mks, n_reps = scores.shape
    if mks is None:
        mks = np.arange(n_mks)
    l = gpl.plot_trace_werr(mks, scores.T, ax=ax, log_x=log_x, **kwargs)
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

def find_linear_mapping_single(dg_use, model, n_samps=10**4, half=True,
                               get_parallelism=True, train_stim_set=None,
                               train_labels=None, test_stim_set=None,
                               test_labels=None, feat_mask=None, **kwargs):
    if train_stim_set is not None and test_stim_set is not None:
        enc_pts = model.get_representation(train_stim_set)
        test_enc_pts = model.get_representation(test_stim_set)
        stim = train_labels
        test_stim = test_labels
    else:
        if half:
            try:
                src = dg_use.source_distribution.make_partition()
            except AttributeError:
                src = da.HalfMultidimensionalNormal.partition(
                    dg_use.source_distribution)
            stim = src.rvs(n_samps)
        else:
            stim = dg_use.source_distribution.rvs(n_samps)

        enc_pts = model.get_representation(dg_use.generator(stim))
        if half:
            flipped = src.flip()
            test_stim = flipped.rvs(n_samps)
        else:
            test_stim = dg_use.source_distribution.rvs(n_samps)
        test_enc_pts = model.get_representation(dg_use.generator(test_stim))
    if feat_mask is None:
        feat_mask = np.ones(stim.shape[1], dtype=bool)
    lr = sklm.Ridge(**kwargs)
    lr.fit(enc_pts, stim[:, feat_mask])
    score = lr.score(test_enc_pts, test_stim[:, feat_mask])
    params = lr.get_params()
    if get_parallelism:
        lr2 = sklm.Ridge(**kwargs)
        lr2.fit(test_enc_pts, test_stim[:, feat_mask])
        sim = u.cosine_similarity(lr.coef_, lr2.coef_)
    else:
        lr2 = None
        sim = None
    return (lr, lr2), score, sim

def test_generalization_new(dg_use=None, models_ths=None, lts_scores=None,
                            train_models_blind=False, inp_dim=2,
                            p_c=None, dg_kind=dg_kind_default,
                            dg_args=None, dg_kwargs=None, dg_source_var=1,
                            dg_train_epochs=25, models_args=None,
                            models_kwargs=None, models_log_x=True,
                            use_samples_x=True, models_n_diffs=6,
                            models_n_bounds=(2, 6.5),
                            hide_print=True, est_inp_dim=None,
                            eval_n_iters=10, use_mp=False,
                            train_test_distrs=None, n_reps=5,
                            model_kinds=model_kinds_default,
                            layer_spec=None, model_n_epochs=60,
                            plot=True, gpu_samples=False):
    # train data generator
    if dg_args is None:
        out_dim = 50
        layers =  (20, 50, 50, 50, 30)
        dg_args = (inp_dim, layers, out_dim)
    if dg_kwargs is None:
        noise = .1
        reg_weight = (0, .15)
        dg_kwargs = {'noise':noise, 'l2_weight':reg_weight}
    
    if dg_use is None:
        dg_use = dg_kind(*dg_args, **dg_kwargs)

        source_distr = sts.multivariate_normal(np.zeros(inp_dim), dg_source_var)
        dg_use.fit(source_distribution=source_distr, epochs=dg_train_epochs,
               use_multiprocessing=use_mp)
    else:
        source_distr = dg_use.source_distribution

    # test dimensionality

    pdims = dg_use.representation_dimensionality()[0]
    print('participation ratio', np.sum(pdims)**2/np.sum(pdims**2))
    if plot:
        plot_representation_dimensionality(dg_use, source_distr=source_distr)

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
            layer_spec = ((40,), (40,), (25,), (est_inp_dim,))
        models_args = (input_dims, dg_use, model_kinds, layer_spec)
    print(layer_spec)
    if models_kwargs is None:
        batch_size = 50
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
            train_d2 = da.HalfMultidimensionalNormal.partition(
                dg_use.source_distribution)
        train_ds = (None, train_d2)
        test_ds = (None, train_d2.flip())
    else:
        train_ds, test_ds = train_test_distr

    if gpu_samples:
        n_train_samples = 2*10**3
        n_test_samples = 10**3
    else:
        n_train_samples = 2*10**4
        n_test_samples = 10**4
        
    if p_c is None:
        p, c = evaluate_multiple_models_dims(dg_use, models, None, test_ds,
                                             train_distributions=train_ds,
                                             n_iters=eval_n_iters,
                                             n_train_samples=n_train_samples,
                                             n_test_samples=n_test_samples)
        print('corr', p)
        print('chance', c)
    else:
        p, c = p_c

    if plot:
        plot_generalization_performance(use_x, p, log_x=models_log_x)
        plot_model_manifolds(dg_use, models)

    if lts_scores is None:
        lts_scores = find_linear_mappings(dg_use, models, half=True,
                                          n_samps=n_test_samples)
        print(np.mean(lts_scores[1]))
    if plot:
        plot_recon_accuracy(lts_scores[1], use_x=use_x, log_x=models_log_x)

    return dg_use, (models, th), (p, c), lts_scores

def plot_recon_gen_summary(run_ind, f_pattern, fwid=3, log_x=True,
                           dg_type=dg.FunctionalDataGenerator,
                           model_type=dd.FlexibleDisentanglerAE,
                           folder='disentangled/simulation_data/partition/'):
    data, info = da.load_full_run(folder, run_ind, 
                                  dg_type=dg_type, model_type=model_type,
                                  file_template=f_pattern, analysis_only=True) 
    n_parts, _, _, _, p, c, _, sc = data
    print(info)
    p = p[..., 1]
    panel_vals = np.logspace(*info['training_eg_args'], dtype=int)
    plot_recon_gen_summary_data((p, sc), n_parts, ylims=((.5, 1), (0, 1)),
                                labels=('gen classifier', 'gen regression'),
                                info=info, log_x=log_x, panel_vals=panel_vals)

def plot_recon_gen_summary_data(quants_plot, x_vals, panel_vals=None,
                                ylims=None, labels=None, x_ax=1, panel_ax=0,
                                fwid=3, info=None, log_x=True, label='',
                                panel_labels='train egs = {}',
                                xlab='partitions', axs=None):
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
        f, axs = plt.subplots(n_panels, n_plots, figsize=fsize, squeeze=False)

    for i, qp in enumerate(quants_plot):
        if info is not None:
            nd = info.get('input_dimensions', None)
        else:
            nd = None
        if x_ax == 0 and panel_ax == 1:
            qp = np.swapaxes(qp, 0, 1)
        else:
            qp = np.swapaxes(qp, x_ax, 1)
            qp = np.swapaxes(qp, panel_ax, 0)
        axs_i = plot_recon_accuracies_ntrain(qp, central_tendency=np.nanmedian,
                                             axs=axs[:, i], xs=x_vals,
                                             log_x=log_x, n_plots=panel_vals,
                                             ylabel=labels[i], ylim=ylims[i],
                                             num_dims=nd, xlab=xlab,
                                             label=label,
                                             plot_labels=panel_labels)
