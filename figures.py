
import numpy as np
import scipy.stats as sts
import functools as ft
import sklearn.decomposition as skd
import sklearn.svm as skc
import scipy.linalg as spla
import itertools as it
import matplotlib.pyplot as plt

import tensorflow as tf

import general.plotting as gpl
import general.plotting_styles as gps
import general.paper_utilities as pu
import general.utility as u
import disentangled.data_generation as dg
import disentangled.disentanglers as dd
import disentangled.characterization as dc
import disentangled.aux as da
import disentangled.theory as dt
import disentangled.multiverse_options as dmo
import disentangled.simulation_db as dsb

config_path = 'disentangled/figures.conf'

colors = np.array([(127,205,187),
                   (65,182,196),
                   (29,145,192),
                   (34,94,168),
                   (37,52,148),
                   (8,29,88)])/256

tuple_int = lambda x: (int(x),)

def _make_cgp_ax(ax):
    ax.set_yticks([.5, 1])
    ax.set_ylabel('classifier')
    gpl.add_hlines(.5, ax)
    ax.set_ylim([.5, 1])

def _make_rgp_ax(ax):
    ax.set_yticks([0, .5, 1])
    ax.set_ylabel('regression')
    gpl.add_hlines(0, ax)
    ax.set_ylim([0, 1])

def plot_cgp(results, ax, **kwargs):
    plot_single_gen(results, ax, **kwargs)
    _make_cgp_ax(ax)

def plot_rgp(results, ax, **kwargs):
    plot_single_gen(results, ax, **kwargs)
    _make_rgp_ax(ax)

def plot_bgp(res_c, res_r, ax_c, ax_r, **kwargs):
    plot_cgp(res_c, ax_c, **kwargs)
    plot_rgp(res_r, ax_r, **kwargs)

def plot_multi_bgp(res_list_c, res_list_r, ax_c, ax_r, legend_labels=None,
                   **kwargs):
    plot_multi_gen(res_list_c, ax_c, **kwargs)
    plot_multi_gen(res_list_r, ax_r, legend_labels=legend_labels, **kwargs)
    _make_cgp_ax(ax_c)
    _make_rgp_ax(ax_r)

def plot_single_gen(results, ax, xs=None, color=None,
                    labels=('trained', 'tested'), legend_label='',
                    marker='o', linestyle='None', rotation=0, **kwargs):
    if xs is None:
        xs = [0, 1]
    gpl.violinplot(results.T, xs, ax=ax, color=(color, color),
                   showextrema=False)
    ax.plot(xs, np.mean(results, axis=0), marker=marker, color=color,
            label=legend_label, linestyle=linestyle, **kwargs)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=rotation)
    gpl.clean_plot(ax, 0)
    gpl.clean_plot_bottom(ax, keeplabels=True)
    return ax

def plot_multi_gen(res_list, ax, xs=None, labels=('trained', 'tested'),
                   sep=.2, colors=None, legend_labels=None, rotation=0,
                   markers=None, **kwargs):
    if xs is None:
        xs = np.array([0, 1])
    if colors is None:
        colors = (None,)*len(res_list)
    if legend_labels is None:
        legend_labels = ('',)*len(res_list)
    if markers is None:
        markers = ('o',)*len(res_list)
    start_xs = xs - len(res_list)*sep/4
    n_seps = (len(res_list) - 1)/2
    use_xs = np.linspace(-sep*n_seps, sep*n_seps, len(res_list))
    
    for i, rs in enumerate(res_list):
        plot_single_gen(rs, ax, xs=xs + use_xs[i], color=colors[i],
                        legend_label=legend_labels[i], marker=markers[i],
                        **kwargs)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=rotation)
    ax.legend(frameon=False)
    gpl.clean_plot(ax, 0)
    gpl.clean_plot_bottom(ax, keeplabels=True)
    return ax

def train_eg_bvae(dg, params):
    beta_eg = params.getfloat('beta_eg')
    latent_dim = params.getint('latent_dim')
    n_epochs = params.getint('n_epochs')
    n_train_eg = params.getint('n_train_eg')
    layer_spec = params.getlist('layers', typefunc=tuple_int)
    batch_size = params.getint('batch_size')
    hide_print = params.getboolean('hide_print')
    eg_model = (ft.partial(dd.BetaVAE, beta=beta_eg),)

    out = dc.train_multiple_models(dg, eg_model,
                                   layer_spec, epochs=n_epochs,
                                   input_dim=latent_dim,
                                   n_train_samps=n_train_eg,
                                   use_mp=True, n_reps=1,
                                   batch_size=batch_size,
                                   hide_print=hide_print)
    return out

def train_eg_fd(dg, params, offset_var=True, **kwargs):
    n_part = params.getint('n_part_eg')
    latent_dim = params.getint('latent_dim')
    n_epochs = params.getint('n_epochs')
    n_train_eg = params.getint('n_train_eg')
    layer_spec = params.getlist('layers', typefunc=tuple_int)
    batch_size = params.getint('batch_size')
    hide_print = params.getboolean('hide_print')
    no_autoenc = params.getboolean('no_autoencoder')
    
    if offset_var:
        offset_var_eg = params.getfloat('offset_var_eg')
        offset_distr = sts.norm(0, offset_var_eg)
    else:
        offset_distr = None
    eg_model = (ft.partial(dd.FlexibleDisentanglerAE,
                           true_inp_dim=dg.input_dim, 
                           n_partitions=n_part,
                           offset_distr=offset_distr,
                           no_autoenc=no_autoenc, **kwargs),)

    out = dc.train_multiple_models(dg, eg_model,
                                   layer_spec, epochs=n_epochs,
                                   input_dim=latent_dim,
                                   n_train_samps=n_train_eg,
                                   use_mp=True, n_reps=1,
                                   batch_size=batch_size,
                                   hide_print=hide_print)
    return out

def explore_autodisentangling_layers(latents, layers, inp_dim, dims, **kwargs):
    out_dict = {}
    for i in range(len(layers) + 1):
        layers_i = layers[:i]
        out = explore_autodisentangling_latents(latents, dims, inp_dim,
                                                layers_i, **kwargs)
        out_dict[layers_i] = out
    return out_dict

def explore_autodisentangling_latents(latents, *args, n_class=10, **kwargs):
    classes = np.zeros((len(latents), n_class, 2))
    regrs = np.zeros_like(classes)
    for i, latent in enumerate(latents):
        full_args = args + (latent,)
        out = explore_autodisentangling(*full_args, n_class=n_class, **kwargs)
        classes[i], regrs[i] = out
    return classes, regrs

def explore_autodisentangling(dims, inp_dim, layers, latent, n_samps=10000,
                              epochs=200, n_class=10, ret_m=False,
                              use_rf=False, low_thr=.001, rf_width=3):
    if use_rf:
        rbf_dg = dg.RFDataGenerator(dims, inp_dim, total_out=True,
                                    low_thr=low_thr, input_noise=0,
                                    noise=0, width_scaling=rf_width)
    else:
        rbf_dg = dg.KernelDataGenerator(dims, None, inp_dim,
                                        low_thr=low_thr)
    print(rbf_dg.representation_dimensionality(participation_ratio=True))
    fdae = dd.FlexibleDisentanglerAE(rbf_dg.output_dim, layers, latent,
                                     n_partitions=0)
    y, x = rbf_dg.sample_reps(n_samps)
    fdae.fit(x, y, epochs=epochs, verbose=False)
    class_p, regr_p = characterize_generalization(rbf_dg,
                                                  dd.IdentityModel(),
                                                  n_class)
    class_m, regr_m = characterize_generalization(rbf_dg, fdae, n_class)
    if ret_m:
        out = class_m, regr_m, (rbf_dg, fdae)
    else:
        out = (class_m, regr_m)
    return out

def characterize_gaussian_process(inp_dim, out_dim, length_scales, eps=1e-3,
                                  max_scale=10, n_reps=10, fit_samples=100):
    prs = np.zeros_like(length_scales)
    class_perf = np.zeros_like(length_scales)
    class_gen = np.zeros_like(length_scales)
    regr_perf = np.zeros_like(length_scales)
    regr_gen = np.zeros_like(length_scales)
    for i, ls in enumerate(length_scales):
        dg_gp = dg.GaussianProcessDataGenerator(inp_dim, 100, out_dim,
                                                length_scale=ls)
        dg_gp.fit(train_samples=fit_samples)
        out = characterize_generalization(dg_gp, dd.IdentityModel(), n_reps)
        pr = dg_gp.representation_dimensionality(participation_ratio=True)
        class_res, regr_res = out
        prs[i] = pr
        class_perf[i], class_gen[i] = np.mean(class_res, axis=0)
        regr_perf[i], regr_gen[i] = np.mean(regr_res, axis=0)
    return prs, class_perf, class_gen, regr_perf, regr_gen        

def characterize_generalization(dg, model, c_reps, train_samples=1000,
                                test_samples=500, bootstrap_regr=True,
                                n_boots=1000, norm=True, cut_zero=True,
                                repl_mean=None, norm_samples=False,
                                **kwargs):
    results_class = np.zeros((c_reps, 2))
    results_regr = np.zeros((c_reps, 2))
    for i in range(c_reps):
        if norm:
            train_distr = da.HalfMultidimensionalNormal.partition(
                dg.source_distribution)
        else:
            train_distr = dg.source_distribution.make_partition()

        test_distr = train_distr.flip()
        results_class[i, 0] = dc.classifier_generalization(
            dg, model, n_train_samples=train_samples,
            n_test_samples=test_samples,
            n_iters=1, repl_mean=repl_mean, **kwargs)[0]
        results_class[i, 1] = dc.classifier_generalization(
            dg, model, train_distrib=train_distr,
            test_distrib=test_distr, n_train_samples=train_samples,
            n_test_samples=test_samples, n_iters=1, repl_mean=repl_mean,
            **kwargs)[0]
        
        results_regr[i, 0] = dc.find_linear_mapping_single(
            dg, model, half=False, n_train_samps=train_samples,
            n_test_samps=test_samples, norm_samples=norm_samples,
            repl_mean=repl_mean, **kwargs)[1]
        results_regr[i, 1] = dc.find_linear_mapping_single(
            dg, model, n_train_samps=train_samples, norm_samples=norm_samples,
            n_test_samps=test_samples,
            repl_mean=repl_mean, **kwargs)[1]
    if cut_zero:
        results_regr[results_regr < 0] = 0
    if False and bootstrap_regr:
        results_regr_b = np.zeros((n_boots, 2))
        results_regr_b[:, 0] = u.bootstrap_list(results_regr[:, 0],
                                                np.mean, n=n_boots)
        results_regr_b[:, 1] = u.bootstrap_list(results_regr[:, 1],
                                                np.mean, n=n_boots)
        results_regr = results_regr_b
    return results_class, results_regr

class DisentangledFigure(pu.Figure):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, find_panel_keys=False, **kwargs)

    def _generate_panel_training_rep_data(self, use_gpdg=False, retrain=False,
                                          use_rf_dg=False, **kwargs):
        if self.data.get(self.models_key) is None or retrain:
            if use_gpdg:
                fdg = self.make_gpdg(**kwargs)
            elif use_rf_dg:
                fdg = self.make_random_rfdg(**kwargs)
            else:
                fdg = self.make_fdg()
            n_parts = self.params.getlist('n_parts', typefunc=int)
            latent_dim = self.params.getint('latent_dim')
            n_reps = self.params.getint('n_reps')
            dg_epochs = self.params.getint('dg_epochs')
            n_epochs = self.params.getint('n_epochs')
            n_train_bounds = self.params.getlist('n_train_eg_bounds',
                                                 typefunc=float)
            n_train_diffs = self.params.getint('n_train_eg_diffs')
            layer_spec = self.params.getlist('layers', typefunc=tuple_int)
            no_autoencoder = self.params.getboolean('no_autoencoder')
            task_offset_var = self.params.getfloat('task_offset_var')
            context_bounds = self.params.getboolean('contextual_boundaries')
            activation_func = self.params.get('activation_func')
            act_func_dict = {
                'relu':tf.nn.relu,
                'none':None,
            }
            act_func = act_func_dict[activation_func]
            task_offset_distr = sts.norm(0, task_offset_var)
            
            model_kinds = list(ft.partial(dd.FlexibleDisentanglerAE,
                                          true_inp_dim=fdg.input_dim, 
                                          n_partitions=num_p,
                                          no_autoenc=no_autoencoder,
                                          noise=.1,
                                          act_func=act_func,
                                          contextual_partitions=context_bounds,
                                          offset_distr=task_offset_distr,
                                          orthog_context=True,
                                          use_early_stopping=True) 
                               for num_p in n_parts)
        
            out = dc.test_generalization_new(
                dg_use=fdg, layer_spec=layer_spec,
                est_inp_dim=latent_dim,
                inp_dim=fdg.output_dim,
                dg_train_epochs=dg_epochs,
                model_n_epochs=n_epochs,
                n_reps=n_reps, model_kinds=model_kinds,
                models_n_diffs=n_train_diffs,
                models_n_bounds=n_train_bounds,
                p_mean=False, plot=False)
            self.data[self.models_key] = (out, (n_parts, n_epochs))
        return self.data[self.models_key]

    def make_gpdg(self, retrain=False, dg_dim=None, gp_ls=None):
        try:
            assert not retrain
            gpdg = self.gpdg
        except:
            inp_dim = self.params.getint('inp_dim')
            if dg_dim is None:
                dg_dim = self.params.getint('dg_dim')
            
            dg_epochs = self.params.getint('dg_epochs')
            dg_noise = self.params.getfloat('dg_noise')
            dg_regweight = self.params.getlist('dg_regweight', typefunc=float)
            dg_layers = self.params.getlist('dg_layers', typefunc=int)
            dg_source_var = self.params.getfloat('dg_source_var')
            dg_train_egs = self.params.getint('gp_dg_train_egs')
            dg_bs = self.params.getint('dg_batch_size')
            if gp_ls is None:
                gp_ls = self.params.getfloat('gpdg_length_scale')
            
            source_distr = sts.multivariate_normal(np.zeros(inp_dim),
                                                   dg_source_var)
            gpdg = dg.GaussianProcessDataGenerator(inp_dim, dg_layers, dg_dim,
                                                   noise=dg_noise,
                                                   length_scale=gp_ls)
            gpdg.fit(source_distribution=source_distr, epochs=dg_epochs,
                    train_samples=dg_train_egs, batch_size=dg_bs,
                    verbose=False)
            self.gpdg = gpdg 
        return gpdg

    def make_random_rfdg(self, retrain=False, dg_dim=None, gp_ls=None):
        try:
            assert not retrain
            gpdg = self.gpdg
        except:
            inp_dim = self.params.getint('inp_dim')
            if dg_dim is None:
                dg_dim = self.params.getint('dg_dim')
            
            dg_epochs = self.params.getint('dg_epochs')
            dg_train_egs = self.params.getint('gp_dg_train_egs')
            dg_bs = self.params.getint('dg_batch_size')
            
            source_distr = da.MultivariateUniform(inp_dim, (-1, 1))
            rfdg = dg.RFDataGenerator(inp_dim, dg_dim,
                                      source_distribution=source_distr,
                                      use_random_rfs=True,
                                      total_out=False)
            self.rfdg = rfdg 
        return rfdg
    
    def make_fdg(self, retrain=False, dg_dim=None):
        try:
            assert not retrain
            fdg = self.fdg
        except:
            inp_dim = self.params.getint('inp_dim')
            if dg_dim is None:
                dg_dim = self.params.getint('dg_dim')
            
            dg_epochs = self.params.getint('dg_epochs')
            dg_noise = self.params.getfloat('dg_noise')
            dg_regweight = self.params.getlist('dg_regweight', typefunc=float)
            dg_layers = self.params.getlist('dg_layers', typefunc=int)
            dg_source_var = self.params.getfloat('dg_source_var')
            dg_train_egs = self.params.getint('dg_train_egs')
            dg_pr_reg = self.params.getboolean('dg_pr_reg')
            dg_bs = self.params.getint('dg_batch_size')
            
            source_distr = sts.multivariate_normal(np.zeros(inp_dim),
                                                   dg_source_var)
            fdg = dg.FunctionalDataGenerator(inp_dim, dg_layers, dg_dim,
                                             noise=dg_noise,
                                             use_pr_reg=dg_pr_reg,
                                             l2_weight=dg_regweight)
            fdg.fit(source_distribution=source_distr, epochs=dg_epochs,
                    train_samples=dg_train_egs, batch_size=dg_bs,
                    verbose=False)
            self.fdg = fdg 
        return fdg

    def load_run(self, run_ind, f_pattern_key='f_pattern', double_ind=None,
                 multi_train=False, analysis_only=True, **kwargs):
        f_pattern = self.params.get(f_pattern_key)
        folder = self.params.get('mp_simulations_path')
        if double_ind is not None:
            merge_axis = 2
        else:
            merge_axis = 1
        dg_type = None
        model_type = None
        data, info = da.load_full_run(folder, run_ind, merge_axis=merge_axis,
                                      dg_type=dg_type, model_type=model_type,
                                      file_template=f_pattern,
                                      analysis_only=analysis_only,
                                      multi_train=multi_train, **kwargs)
        n_parts, _, _, th, p, c, _, sc, _, other = data
        return n_parts, p, c, sc, th, info, other

    def _abstraction_panel(self, run_inds, res_axs, f_pattern=None,
                           folder=None, labels=None, colors=None,
                           multi_num=1, set_lims=True,
                           collapse_plots=False, **kwargs):
        if f_pattern is None:
            f_pattern = self.params.get('f_pattern')
        if folder is None:
            folder = self.params.get('mp_simulations_path')
        if labels is None:
            labels = ('',)*len(run_inds)
        if colors is None:
            colors = (None,)*len(run_inds)
        if multi_num > 1:
            double_inds = np.concatenate(list((i,)*len(run_inds)
                                              for i in range(multi_num)))
            run_inds = run_inds*multi_num
            labels = labels*multi_num
            colors=colors*multi_num
        else:
            double_inds = (None,)*len(run_inds)
        for i, ri in enumerate(run_inds):
            dc.plot_recon_gen_summary(ri, f_pattern, log_x=False, 
                                      collapse_plots=collapse_plots,
                                      folder=folder,
                                      axs=res_axs, legend=labels[i],
                                      print_args=False, set_title=False,
                                      color=colors[i], double_ind=double_inds[i],
                                      set_lims=set_lims, **kwargs)
        res_axs[0, 0].set_yticks([.5, 1])
        res_axs[0, 1].set_yticks([0, .5, 1])

    def _manifold_panel(self, fdg, model, axs, rep_scale_mag=5,
                        source_scale_mag=.5, x_label=True, y_label=True,
                        view_init=None, **kwargs):
        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        vis_3d = self.params.getboolean('vis_3d')

        # print(characterize_generalization(fdg, model, 10))
        dc.plot_source_manifold(fdg, model, rs, n_arcs, 
                                source_scale_mag=source_scale_mag,
                                rep_scale_mag=rep_scale_mag,
                                markers=False, axs=axs,
                                titles=False, plot_model_3d=vis_3d,
                                model_view_init=view_init, **kwargs)


    def _standard_panel(self, fdg, model, run_inds, f_pattern=None, folder=None,
                        axs=None, labels=None, rep_scale_mag=5,
                        source_scale_mag=.5, x_label=True, y_label=True,
                        colors=None, view_init=None,
                        multi_num=1, set_lims=True, collapse_plots=True,
                        **kwargs):
        model = model[0, 0]
        if len(axs) == 3:
            ax_break = 1
        else:
            ax_break = 2
        manifold_axs = axs[:ax_break]
        res_axs = np.expand_dims(axs[ax_break:], 0)

        self._manifold_panel(fdg, model, manifold_axs,
                             rep_scale_mag=rep_scale_mag,
                             source_scale_mag=source_scale_mag,
                             x_label=x_label, y_label=y_label,
                             view_init=view_init)
                             
        self._abstraction_panel(run_inds, res_axs, f_pattern, folder, 
                                labels=labels, colors=colors,
                                multi_num=multi_num, set_lims=set_lims,
                                collapse_plots=collapse_plots, **kwargs)
        

class Figure1(DisentangledFigure):
    
    def __init__(self, fig_key='figure1', colors=colors, **kwargs):
        fsize = (6, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('partition_schematic',
                           'representation_schematic',
                           'encoder_schematic',
                           'encoder_visualization', 'metric_schematic')
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}

        part_schem_grid = self.gs[:80, :30]
        gss[self.panel_keys[0]] = self.get_axs((part_schem_grid,))

        metric_schem_grid = self.gs[:80, 36:55]
        gss[self.panel_keys[4]] = self.get_axs((metric_schem_grid,))

        rep_schem_grid = pu.make_mxn_gridspec(self.gs, 2, 2,
                                                25, 100, 0, 55,
                                                0, 0)
        gss[self.panel_keys[1]] = self.get_axs(rep_schem_grid, all_3d=True)
        
        encoder_schem_grid = self.gs[:40, 70:]
        gss[self.panel_keys[2]] = self.get_axs((encoder_schem_grid,))

        plot_3d_axs = np.zeros((2, 2), dtype=bool)
        plot_3d_axs[0, 1] = self.params.getboolean('vis_3d')
        ev1_grid = pu.make_mxn_gridspec(self.gs, 1, 2,
                                        50, 70, 65, 100,
                                        8, 0)
        ev2_grid = pu.make_mxn_gridspec(self.gs, 1, 2,
                                        79, 100, 65, 100,
                                        8, 15)
        ev_grid = np.concatenate((ev1_grid, ev2_grid), axis=0)
        gss[self.panel_keys[3]] = self.get_axs(ev_grid,
                                               plot_3ds=plot_3d_axs)

        self.gss = gss

    def _make_nonlin_func(self, cents, wids=2):
        def f(x):
            cs = np.expand_dims(cents, 0)
            xs = np.expand_dims(x, 1)
            d = np.sum(-(xs - cs)**2, axis=2)
            r = np.exp(d/2*wids)
            return r
        return f

    def _plot_schem(self, pts, f, ax, corners=None, corner_color=None,
                    **kwargs):
        pts_trs = f(pts)
        l = ax.plot(pts_trs[:, 0], pts_trs[:, 1], pts_trs[:, 2],
                    **kwargs)
        if corners is not None:
            if corner_color is not None:
                kwargs['color'] = corner_color
            corners_trs = f(corners)
            ax.plot(corners_trs[:, 0], corners_trs[:, 1],
                    corners_trs[:, 2], 'o', **kwargs)
            
    def _plot_hyperplane(self, pts, lps, f, ax):
        pts_f = f(pts)
        lps_f = f(lps)

        cats = [0, 1]
        c = skc.SVC(kernel='linear', C=1000)
        c.fit(lps_f, cats)
        n_vecs = spla.null_space(c.coef_)
        v1 = np.linspace(-1, 1, 2)
        v2 = np.linspace(-1, 1, 2)
        x, y = np.meshgrid(v1, v2)
        x_ns = np.expand_dims(x, 0)*np.expand_dims(n_vecs[:, 0], (1, 2))
        y_ns = np.expand_dims(y, 0)*np.expand_dims(n_vecs[:, 1], (1, 2))
        offset = np.expand_dims(c.coef_[0]*c.intercept_,
                                (1, 2))
        proj_pts = x_ns + y_ns
        proj_pts = proj_pts - offset
        ax.plot_surface(*proj_pts, alpha=1)
            
    def panel_representation_schematic(self):
        key = self.panel_keys[1]
        ax_lin, ax_nonlin = self.gss[key][0]
        ax_lin_h, ax_nonlin_h = self.gss[key][1]
        rpt = 1
        lpt = -1
        pts, corners = dc.make_square(100, lpt=lpt, rpt=rpt)

        pts_h1, corners_h1 = dc.make_half_square(100, lpt=lpt, rpt=rpt)
        pts_h2, corners_h2 = dc.make_half_square(100, lpt=rpt, rpt=lpt)
        trs = u.make_unit_vector(np.array([[1, 1],
                                           [-1, 1],
                                           [-1, .5]]))
        lin_func = lambda x: np.dot(x, trs.T)
        cents = np.array([[rpt, rpt],
                          [lpt, lpt],
                          [.5*rpt, 0]])
        nonlin_func = self._make_nonlin_func(cents)
        rads = self.params.getlist('manifold_radii', typefunc=float)
        grey_col = self.params.getcolor('grey_color')
        pt_color = self.params.getcolor('berry_color')
        h1_color = self.params.getcolor('train_color')
        h2_color = self.params.getcolor('test_color')
        alpha = self.params.getfloat('schem_alpha')
        ms = 3
        elev_lin = 20
        az_lin = -10
        elev_nonlin = 50
        az_nonlin = -120
        colors = (grey_col,)*(len(rads) - 1) + (grey_col,)
        alphas = (alpha,)*(len(rads) - 1) + (1,)
        
        for i, r in enumerate(rads):
            if i == len(rads) - 1:
                corners_p = r*corners
            else:
                corners_p = None
            self._plot_schem(r*pts, lin_func, ax_lin, corners=corners_p, 
                             color=colors[i], corner_color=pt_color,
                             alpha=alphas[i], markersize=ms)
            self._plot_schem(r*pts, nonlin_func, ax_nonlin, corners=corners_p,
                             color=colors[i], corner_color=pt_color,
                             alpha=alphas[i], markersize=ms)
            
            self._plot_schem(r*pts_h1, lin_func, ax_lin_h, corners=corners_p,
                             color=h1_color, corner_color=pt_color,
                             alpha=alphas[i], markersize=ms)
            self._plot_schem(r*pts_h2, lin_func, ax_lin_h, corners=None,
                             color=h2_color, corner_color=pt_color,
                             alpha=alphas[i], markersize=ms)
            
            self._plot_schem(r*pts_h1, nonlin_func, ax_nonlin_h,
                             corners=corners_p,
                             color=h1_color, corner_color=pt_color,
                             alpha=alphas[i], markersize=ms)
            self._plot_schem(r*pts_h2, nonlin_func, ax_nonlin_h,
                             corners=None,
                             color=h2_color, corner_color=pt_color,
                             alpha=alphas[i], markersize=ms)
            
        self._plot_hyperplane(r*pts_h1, corners_p[:2], lin_func,
                              ax_lin_h)
        self._plot_hyperplane(r*pts_h1, corners_p[:2], nonlin_func,
                              ax_nonlin_h)

        ax_lin.view_init(elev_lin, az_lin)
        ax_nonlin.view_init(elev_nonlin, az_nonlin)
        ax_lin_h.view_init(elev_lin, az_lin)
        ax_nonlin_h.view_init(elev_nonlin, az_nonlin)
        gpl.set_3d_background(ax_nonlin)
        gpl.set_3d_background(ax_lin)
        gpl.remove_ticks_3d(ax_nonlin)
        gpl.remove_ticks_3d(ax_lin)
        gpl.set_3d_background(ax_nonlin_h)
        gpl.set_3d_background(ax_lin_h)
        gpl.remove_ticks_3d(ax_nonlin_h)
        gpl.remove_ticks_3d(ax_lin_h)

    def panel_encoder_visualization(self):
        key = self.panel_keys[3]
        axs = self.gss[key]
        vis_axs = axs[0]
        class_ax, regr_ax = axs[1]
        if self.data.get(key) is None:
            fdg = self.make_fdg()
            exp_dim = fdg.representation_dimensionality(
                participation_ratio=True)
            pass_model = dd.IdentityModel()

            c_reps = self.params.getint('dg_classifier_reps')
            gen_perf = characterize_generalization(fdg, pass_model,
                                                   c_reps)
            self.data[key] = (fdg, pass_model, exp_dim, gen_perf)
        fdg, pass_model, exp_dim, gen_perf = self.data[key]

        print('PR = {}'.format(exp_dim))
        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        vis_3d = self.params.getboolean('vis_3d')
        
        dc.plot_source_manifold(fdg, pass_model, rs, n_arcs, 
                                source_scale_mag=.5,
                                rep_scale_mag=.03,
                                markers=False, axs=vis_axs,
                                titles=False, plot_model_3d=vis_3d,
                                l_axlab_str='latent dim {} (au)')
        dg_color = self.params.getcolor('dg_color')
        plot_bgp(gen_perf[0], gen_perf[1], class_ax, regr_ax, color=dg_color)
        # plot_single_gen(gen_perf[0], class_ax, color=dg_color)
        # plot_single_gen(gen_perf[1], regr_ax, color=dg_color)
        # class_ax.set_ylabel('classifier\ngeneralization')
        # regr_ax.set_ylabel('regression\ngeneralization')
        # gpl.add_hlines(.5, class_ax)
        # gpl.add_hlines(0, regr_ax)
        # class_ax.set_ylim([.5, 1])
        # regr_ax.set_ylim([0, 1])

class Figure1Alt(Figure1):

    def make_gss(self):
        gss = {}

        part_schem_grid = self.gs[:80, :30]
        gss[self.panel_keys[0]] = self.get_axs((part_schem_grid,))

        metric_schem_grid = self.gs[:80, 36:55]
        gss[self.panel_keys[4]] = self.get_axs((metric_schem_grid,))

        rep_schem_grid = pu.make_mxn_gridspec(self.gs, 2, 2,
                                                25, 100, 0, 55,
                                                0, 0)
        gss[self.panel_keys[1]] = self.get_axs(rep_schem_grid, all_3d=True)
        
        encoder_schem_grid = self.gs[:40, 70:]
        gss[self.panel_keys[2]] = self.get_axs((encoder_schem_grid,))

        plot_3d_axs = np.zeros((2, 2), dtype=bool)
        plot_3d_axs[0, 1] = self.params.getboolean('vis_3d')

        latent_vis = pu.make_mxn_gridspec(self.gs, 1, 2,
                                          50, 70, 65, 75,
                                          2, 0)
        lv_axs = self.get_axs(latent_vis)
        
        rep_vis = pu.make_mxn_gridspec(self.gs, 3, 3,
                                       50, 70, 80, 100,
                                       2, 2)
        rep_axs = self.get_axs(rep_vis)

        perf_vis = pu.make_mxn_gridspec(self.gs, 1, 2,
                                        79, 100, 65, 100,
                                        8, 15)
        perf_axs = self.get_axs(perf_vis)
        
        gss[self.panel_keys[3]] = (lv_axs, rep_axs, perf_axs)

        self.gss = gss

    def panel_encoder_visualization(self):
        key = self.panel_keys[3]
        axs = self.gss[key]
        vis_axs = axs[0]
        lv_axs, rep_axs, (class_ax, regr_ax) = axs[1]
        if self.data.get(key) is None:
            fdg = self.make_fdg()
            exp_dim = fdg.representation_dimensionality(
                participation_ratio=True)
            pass_model = dd.IdentityModel()

            c_reps = self.params.getint('dg_classifier_reps')
            gen_perf = characterize_generalization(fdg, pass_model,
                                                   c_reps)
            self.data[key] = (fdg, pass_model, exp_dim, gen_perf)
        fdg, pass_model, exp_dim, gen_perf = self.data[key]

        print('PR = {}'.format(exp_dim))
        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        vis_3d = self.params.getboolean('vis_3d')
        
        dc.plot_source_manifold(fdg, pass_model, rs, n_arcs, 
                                source_scale_mag=.5,
                                rep_scale_mag=.03,
                                markers=False, axs=vis_axs,
                                titles=False, plot_model_3d=vis_3d,
                                l_axlab_str='latent dim {} (au)')
        dg_color = self.params.getcolor('dg_color')
        plot_bgp(gen_perf[0], gen_perf[1], class_ax, regr_ax, color=dg_color)
        # plot_single_gen(gen_perf[0], class_ax, color=dg_color)
        # plot_single_gen(gen_perf[1], regr_ax, color=dg_color)
        # class_ax.set_ylabel('classifier\ngeneralization')
        # regr_ax.set_ylabel('regression\ngeneralization')
        # gpl.add_hlines(.5, class_ax)
        # gpl.add_hlines(0, regr_ax)
        # class_ax.set_ylim([.5, 1])
        # regr_ax.set_ylim([0, 1])


class FigureInp(DisentangledFigure):

    def __init__(self, fig_key='figure_inp', colors=colors, **kwargs):
        fsize = (6, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('schem', 'rfs', 'info', 'rep_vis',
                           'abstraction', 'gen_vis')
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        schem_grid = self.gs[:40, :50]
        gss[self.panel_keys[0]] = self.get_axs((schem_grid,))

        rfs_grid = pu.make_mxn_gridspec(self.gs, 5, 5,
                                        0, 45, 60, 100,
                                        2, 2)
        gss[self.panel_keys[1]] = self.get_axs(rfs_grid, aspect='equal')

        rest_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 55, 100, 70, 100,
                                         10, 10)
        axs = self.get_axs(rest_grid)
        gss[self.panel_keys[2]] = axs[0]
        gss[self.panel_keys[4]] = axs[1]

        vis_grid = pu.make_mxn_gridspec(self.gs, 2, 2, 45, 100, 0, 60,
                                        0, 10)
        plot_3d_axs = np.zeros((2, 2), dtype=bool)
        plot_3d_axs[0, 1] = self.params.getboolean('vis_3d')
        vis_axs = self.get_axs(vis_grid, plot_3ds=plot_3d_axs)
        gss[self.panel_keys[3]] = vis_axs[0]
        gss[self.panel_keys[5]] = vis_axs[1]

        self.gss = gss

    def panel_rfs(self):
        key = self.panel_keys[1]
        axs = self.gss[key]

        fdg = self.make_fdg()
        cmap = self.params.get('rf_cmap')
        dc.plot_dg_rfs(fdg, axs=axs, cmap=cmap, rasterized=True)
        for ind in u.make_array_ind_iterator(axs.shape):
            if ind[1] > 0:
                axs[ind].set_yticks([])
            if ind[0] < axs.shape[0] - 1:
                axs[ind].set_xticks([])
        axs[-1, 0].set_xlabel('latent dim 1')
        axs[-1, 0].set_ylabel('latent dim 2')

    def panel_info(self):
        key = self.panel_keys[2]
        (sp_ax, dim_ax) = self.gss[key]

        fdg = self.make_fdg()
        stim, reps = fdg.sample_reps()
        sparseness = dc.quantify_sparseness(reps)
        dimensionality = fdg.representation_dimensionality(
            participation_ratio=True)
        print('sparseness', np.nanmean(sparseness))
        print('dimensionality', dimensionality)
        dg_color = self.params.getcolor('dg_color')
        lv_color = self.params.getcolor('lv_color')
        ms = self.params.getfloat('markersize')

        gpl.plot_trace_werr([1], np.expand_dims(sparseness, 1), points=True,
                              ax=sp_ax, color=dg_color, markersize=ms)
        sp_ax.plot([0], [0], 's', color=lv_color, markersize=ms)
        sp_ax.set_ylim([-.1, 1.1])
        dim_ax.plot([1], dimensionality, 'o', color=dg_color, markersize=ms)
        dim_ax.plot([0], [5], 's', color=lv_color, markersize=ms)
        dim_ax.set_ylim([-1, 210])
        sp_ax.set_ylabel('sparseness')
        dim_ax.set_ylabel('dimensionality')
        gpl.clean_plot(sp_ax, 0)
        gpl.clean_plot_bottom(sp_ax)
        gpl.clean_plot(dim_ax, 0)
        gpl.clean_plot_bottom(dim_ax)
        sp_ax.set_xlim([-1, 2])
        dim_ax.set_xlim([-1, 2])

    def panel_gen_vis(self):
        key = self.panel_keys[5]
        class_ax, regr_ax = self.gss[key]

        fdg = self.make_fdg()
        pass_model = dd.IdentityModel()

        grid_pts = self.params.getint('grid_pts')
        
        dc.plot_class_grid(fdg, pass_model, grid_pts=30,
                           ax=class_ax, col_eps=.15)
        dc.plot_regr_grid(fdg, pass_model, grid_pts=30,
                          ax=regr_ax)
        
        
    def panel_rep_vis(self):
        key = self.panel_keys[3]
        vis_axs = self.gss[key]

        fdg = self.make_fdg()
        pass_model = dd.IdentityModel()

        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        vis_3d = self.params.getboolean('vis_3d')

        dc.plot_source_manifold(fdg, pass_model, rs, n_arcs, 
                                source_scale_mag=.5,
                                rep_scale_mag=.03,
                                markers=False, axs=vis_axs,
                                titles=False, plot_model_3d=vis_3d,
                                l_axlab_str='latent dim {} (au)')

    def panel_abstraction(self):
        key = self.panel_keys[4]
        (class_ax, regr_ax) = self.gss[key]

        if self.data.get(key) is None:
            fdg = self.make_fdg()
            pass_model = dd.IdentityModel()
            ident_dg = dg.IdentityDG(fdg.source_distribution)

            c_reps = self.params.getint('dg_classifier_reps')
            gen_perf = characterize_generalization(fdg, pass_model,
                                                   c_reps)
            lv_perf = characterize_generalization(ident_dg, pass_model,
                                                  c_reps)
            
            self.data[key] = (lv_perf, gen_perf)
        lv_perf, gen_perf = self.data[key]
        dg_color = self.params.getcolor('dg_color')
        lv_color = self.params.getcolor('lv_color')
        ms = self.params.getfloat('markersize')
        plot_multi_bgp((lv_perf[0], gen_perf[0]), (lv_perf[1], gen_perf[1]),
                       class_ax, regr_ax, colors=(lv_color, dg_color),
                       legend_labels=('latent variables', 'input'),
                       markers=('s', 'o'), markersize=ms,
                       rotation=45)
        class_ax.set_ylim([.5, 1.05])
        regr_ax.set_ylim([0, 1.05])
        
        

class SIFigureGPTask(DisentangledFigure):

    def __init__(self, fig_key='sifigure_gp_task', colors=colors, **kwargs):
        fsize = (3, 4.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('sample_efficiency',)
        self.models_key = 'trained_models'
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}

        res_grids = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 100, 0, 100,
                                         10, 25)
        res_axs = self.get_axs(res_grids)
        
        # rep_classifier_grid = self.gs[75:, 60:75]
        # rep_regression_grid = self.gs[75:, 85:]
        # gss[self.panel_keys[2]] = self.get_axs((rep_classifier_grid,
        #                                         rep_regression_grid))
        gss[self.panel_keys[0]] = res_axs

        self.gss = gss

    def panel_sample_efficiency(self, recompute=False):
        key = self.panel_keys[0]
        axs = self.gss[key]

        if self.data.get(key) is None or recompute:
            fdg = self.make_fdg()
            run_inds = self.params.getlist('efficiency_run_inds')
            out_dict = {}
            for i, run_ind in enumerate(run_inds):
            
                n_parts, p, c, sc, _, info, _ = self.load_run(run_ind, double_ind=0,
                                                           multi_train=True)
                
                lg_args = info['args'][0]['training_samples_seq']
                gp_len = info['args'][0]['gp_test_task_length_scale']
                n_train_samples = np.logspace(*lg_args[:2], int(lg_args[2]),
                                              dtype=int)
                ident_models = [np.array([dd.IdentityModel()])]
                ident_dg = dg.IdentityDG(fdg.source_distribution)
                
                out_trad = dc.evaluate_multiple_models_dims(
                    fdg, ident_models, None, (fdg.source_distribution,),
                    n_iters=10, n_train_samples=n_train_samples,
                    gp_task_ls=gp_len)
                out_trad_asymp = dc.evaluate_multiple_models_dims(
                    ident_dg, ident_models,
                    None, (fdg.source_distribution,),
                    n_iters=10, n_train_samples=n_train_samples,
                    gp_task_ls=gp_len)

                train_distr = da.HalfMultidimensionalNormal.partition(
                    fdg.source_distribution)
                test_distr = train_distr.flip()

                out_gen = dc.evaluate_multiple_models_dims(
                    fdg, ident_models, None, (test_distr,),
                    train_distributions=(train_distr,), n_iters=10,
                    n_train_samples=n_train_samples,
                    gp_task_ls=gp_len)
                out_gen_asymp = dc.evaluate_multiple_models_dims(
                    ident_dg, ident_models, None, (test_distr,),
                    train_distributions=(train_distr,), n_iters=10,
                    n_train_samples=n_train_samples,
                    gp_task_ls=gp_len)
                
                standard = (out_trad[0], out_trad_asymp[0])
                gen = (out_gen[0], out_gen_asymp[0])
                save = (fdg, n_parts, n_train_samples, standard,
                        gen, p)
                out_dict[gp_len] = save
            self.data[key] = out_dict
        out_dict = self.data[key]

        ub_color = self.params.getcolor('upper_bound_color')
        lb_color = self.params.getcolor('lower_bound_color')
        run_color = self.params.getcolor('partition_color')
        plot_n_parts = self.params.getint('plot_n_parts')
        for i, (gp_len, save) in enumerate(out_dict.items()):
            (fdg, n_parts, n_train_samples, standard, gen, p) = save
            p_standard = p[..., 0]
            p_gen = p[..., 1]
            n_part_ind = np.argmin(np.abs(np.array(plot_n_parts) - n_parts))

            gpl.plot_trace_wpts(n_train_samples, p_standard[1, :, n_part_ind].T,
                                ax=axs[i, 0], color=run_color)
            axs[i, 0].plot(n_train_samples, np.squeeze(standard[0]), color=lb_color)
            axs[i, 0].plot(n_train_samples, np.squeeze(standard[1]), color=ub_color)

            gpl.plot_trace_wpts(n_train_samples, p_gen[1, :, n_part_ind].T,
                                ax=axs[i, 1],
                                color=run_color)
            axs[i, 1].plot(n_train_samples, np.squeeze(gen[0]), color=lb_color,
                           label='lower bound')
            axs[i, 1].plot(n_train_samples, np.squeeze(gen[1]), color=ub_color,
                           label='upper bound')

            axs[i, 0].set_xscale('log')
            axs[i, 1].set_xscale('log')
            axs[i, 0].set_ylim([.5, 1])
            axs[i, 1].set_ylim([.5, 1])
            axs[i, 0].set_ylabel('novel task\nperformance')
            axs[i, 1].set_ylabel('novel task\ngeneralization')
            axs[i, 0].set_title('LS = {}'.format(gp_len))
            
        axs[i, 0].set_xlabel('novel samples')
        axs[i, 1].legend(frameon=False)

class Figure2(DisentangledFigure):
    
    def __init__(self, fig_key='figure2', colors=colors, **kwargs):
        fsize = (7, 8)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('order_disorder', 'training_rep', 'rep_summary',
                           'sample_efficiency', 'resp_distrib', 'training_vis')
        self.models_key = 'trained_models'
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}

        # ordered_rep_grid = self.gs[:25, :30]
        # class_perf_grid = self.gs[75:, :15]
        # regr_perf_grid = self.gs[75:, 30:45]

        # inp_grid = pu.make_mxn_gridspec(self.gs, 1, 2,
        #                                     50, 68, 10, 50,
        #                                     5, 10)
        # high_d_grid = pu.make_mxn_gridspec(self.gs, 1, 3,
        #                                    75, 100, 0, 10,
        #                                    5, 2)
        # high_d_grid = (self.gs[75:, :5],)
        # hypoth_grids = pu.make_mxn_gridspec(self.gs, 1, 2,
        #                                     75, 100, 10, 50,
        #                                     5, 5)
        # gss[self.panel_keys[0]] = (self.get_axs(inp_grid),
        #                            self.get_axs(high_d_grid),
        #                            self.get_axs(hypoth_grids))
        
        train_grid = self.gs[:15, 35:55]
        train_ax = self.get_axs((train_grid,))[0, 0]
        n_parts = len(self.params.getlist('n_parts'))
        rep_grids = pu.make_mxn_gridspec(self.gs, n_parts, 5,
                                         45, 100, 0, 100,
                                         5, 5)
        plot_3d_axs = np.zeros((n_parts, 5), dtype=bool)
        plot_3d_axs[:, 1] = self.params.getboolean('vis_3d')
        rep_axs = self.get_axs(rep_grids, sharex='vertical',
                               sharey='vertical', plot_3ds=plot_3d_axs)
        gss[self.panel_keys[1]] = train_ax, rep_axs[:, :2]
        gss[self.panel_keys[5]] = rep_axs[:, 2:4]        
        gss[self.panel_keys[4]] = rep_axs[:, 4]


        res_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 40, 60, 100,
                                         10, 10)
        res_axs = self.get_axs(res_grids)
        
        # rep_classifier_grid = self.gs[75:, 60:75]
        # rep_regression_grid = self.gs[75:, 85:]
        # gss[self.panel_keys[2]] = self.get_axs((rep_classifier_grid,
        #                                         rep_regression_grid))
        gss[self.panel_keys[2]] = res_axs[0]

        gss[self.panel_keys[3]] = res_axs[1]
        self.gss = gss

    def panel_training_vis(self):
        key = self.panel_keys[5]
        axs = self.gss[key]

        out = self._generate_panel_training_rep_data()            
        fdg, (models, th), (p, _), (_, scrs, _) = out[0][:4]
        n_parts, n_epochs = out[1]

        grid_pts = self.params.getint('grid_pts')
        for i, num_p in enumerate(n_parts):
            model = models[0, i, 0]
            dc.plot_class_grid(fdg, model, grid_pts=grid_pts,
                               ax=axs[i, 0], col_eps=.15)
            dc.plot_regr_grid(fdg, model, grid_pts=grid_pts,
                              ax=axs[i, 1])
            

    def panel_rep_distribution(self):
        key = self.panel_keys[4]
        axs = self.gss[key]

        out = self._generate_panel_training_rep_data()
        fdg, (models, th), (p, _), (_, scrs, _) = out[0][:4]
        n_parts, n_epochs = out[1]

        vis_3d = self.params.getboolean('vis_3d')
        colormap = self.params.get('vis_colormap')
        cmap = plt.get_cmap(colormap)
        task_colors = cmap([.1, .9])
        c1_color = self.params.getcolor('c1_color')
        c2_color = self.params.getcolor('c2_color')
        task_colors = (c1_color, c2_color)
        for i, num_p in enumerate(n_parts):
            dc.plot_task_reps(fdg, models[0, i, 0], axs=(axs[i],),
                              plot_tasks=(0,), colors=task_colors,
                              density=True)
            gpl.clean_plot(axs[i], 1)
            gpl.make_xaxis_scale_bar(axs[i], 20)
            gpl.make_xaxis_scale_bar(axs[i], 20)
            gpl.make_yaxis_scale_bar(axs[i], .02, double=False)
        axs[-1].set_xlabel('projection onto\ntask vector')

        
    def panel_sample_efficiency(self, recompute=False):
        key = self.panel_keys[3]
        axs = self.gss[key]

        if self.data.get(key) is None or recompute:
            fdg = self.make_fdg()
            run_ind = self.params.get('efficiency_run_ind')
            n_parts, p, c, sc, _, info = self.load_run(run_ind, double_ind=0,
                                                       multi_train=True)

            lg_args = info['args'][0]['training_samples_seq']
            n_train_samples = np.logspace(*lg_args[:2], int(lg_args[2]),
                                          dtype=int)
            ident_models = [np.array([dd.IdentityModel()])]
            ident_dg = dg.IdentityDG(fdg.source_distribution)
            out_trad = dc.evaluate_multiple_models_dims(
                fdg, ident_models, None, (fdg.source_distribution,),
                n_iters=10, n_train_samples=n_train_samples)
            out_trad_asymp = dc.evaluate_multiple_models_dims(
                ident_dg, ident_models,
                None, (fdg.source_distribution,),
                n_iters=10, n_train_samples=n_train_samples)

            train_distr = da.HalfMultidimensionalNormal.partition(
                fdg.source_distribution)
            test_distr = train_distr.flip()

            out_gen = dc.evaluate_multiple_models_dims(
                fdg, ident_models, None, (test_distr,),
                train_distributions=(train_distr,), n_iters=10,
                n_train_samples=n_train_samples)
            out_gen_asymp = dc.evaluate_multiple_models_dims(
                ident_dg, ident_models, None, (test_distr,),
                train_distributions=(train_distr,), n_iters=10,
                n_train_samples=n_train_samples)
            standard = (out_trad[0], out_trad_asymp[0])
            gen = (out_gen[0], out_gen_asymp[0])
            self.data[key] = (fdg, n_parts, n_train_samples, standard,
                              gen, p)
        (fdg, n_parts, n_train_samples, standard, gen, p) = self.data[key]
        p_standard = p[..., 0]
        p_gen = p[..., 1]


        ub_color = self.params.getcolor('upper_bound_color')
        lb_color = self.params.getcolor('lower_bound_color')
        run_color = self.params.getcolor('partition_color')
        plot_n_parts = self.params.getint('plot_n_parts')
        n_part_ind = np.argmin(np.abs(np.array(plot_n_parts) - n_parts))

        gpl.plot_trace_wpts(n_train_samples, p_standard[1, :, n_part_ind].T,
                            ax=axs[0], color=run_color)
        axs[0].plot(n_train_samples, np.squeeze(standard[0]), color=lb_color)
        axs[0].plot(n_train_samples, np.squeeze(standard[1]), color=ub_color)

        gpl.plot_trace_wpts(n_train_samples, p_gen[1, :, n_part_ind].T, ax=axs[1],
                            color=run_color)
        axs[1].plot(n_train_samples, np.squeeze(gen[0]), color=lb_color,
                    label='lower bound')
        axs[1].plot(n_train_samples, np.squeeze(gen[1]), color=ub_color,
                    label='upper bound')

        axs[0].set_xscale('log')
        axs[1].set_xscale('log')
        axs[0].set_ylim([.5, 1])
        axs[1].set_ylim([.5, 1])
        axs[0].set_xlabel('novel samples')
        axs[0].set_ylabel('novel task\nperformance')
        axs[1].set_ylabel('novel task\ngeneralization')
        axs[1].legend(frameon=False)
        
    def panel_learning_history(self, recompute=False):
        key = 'blah'
        if self.data.get(key) is None or recompute:
            fdg = self.make_fdg()
            run_ind = self.params.get('efficiency_run_ind')
            n_parts, p, c, sc, hist, info = self.load_run(run_ind, double_ind=0,
                                                          multi_train=True,
                                                          add_hist=True)
            self.data[key] = (n_parts, hist, info)
        n_parts, hist, info = self.data[key]        
        
    def panel_order_disorder(self):
        key = self.panel_keys[0]
        (ax_inp, ax_hd, axs) = self.gss[key]
        if self.data.get(key) is None:
            fdg = self.make_fdg()
            exp_dim = fdg.representation_dimensionality(
                participation_ratio=True)
            pass_model = dd.IdentityModel()

            self.data[key] = (fdg, pass_model, exp_dim)
        fdg, pass_model, exp_dim = self.data[key]

        map_dims = self.params.getint('map_dims')
        map_parts = self.params.getint('map_parts')
        
        samps, targs, targs_scal, _ = dt.generate_binary_map(map_dims,
                                                             map_parts)
        p = skd.PCA()
        p.fit(targs)
        targs_dim = p.transform(targs)
        p_scal = skd.PCA()
        p_scal.fit(targs_scal)
        
        partition_color = self.params.getcolor('partition_color')
        theory_color = self.params.getcolor('theory_color')
        ax_inp[0, 0].plot(p.explained_variance_ratio_, 'o', label='actual',
                          color=partition_color)
        ax_inp[0, 0].plot(p_scal.explained_variance_ratio_, 'o',
                          label='linear theory', color=theory_color)
        ax_inp[0, 0].legend(frameon=False)
        ax_inp[0, 0].set_xlabel('PC number')
        ax_inp[0, 0].set_ylabel('proportion\nexplained')
        gpl.clean_plot(ax_inp[0, 0], 0)
        ax_inp[0, 1].plot(targs_dim[:, 0], targs_dim[:, 1], 'o',
                          color=partition_color)
        gpl.clean_plot(ax_inp[0, 1], 0)
        gpl.make_yaxis_scale_bar(ax_inp[0, 1], .8)
        gpl.make_xaxis_scale_bar(ax_inp[0, 1], .8)
        ax_inp[0, 1].set_xlabel('PC 1')
        ax_inp[0, 1].set_ylabel('PC 2')
        
        eps = [-.1, -.05, 0, .05, .1]
        for i, eps_i in enumerate(eps):
            ax_hd[0].plot([0, 0], [1 + eps_i, -1 - eps_i], 'o')
        gpl.clean_plot(ax_hd[0], 0)
        gpl.clean_plot_bottom(ax_hd[0])
        gpl.make_yaxis_scale_bar(ax_hd[0], .8)
        ax_hd[0].set_ylabel('PC P')
        # for i, eps_i in enumerate(eps):
        #     ax_hd[0, 0].plot([0, 0], [1 + eps_i, -1 - eps_i], 'o')
        #     ax_hd[0, 2].plot([0, 0], [1 + eps_i, -1 - eps_i], 'o')
        # gpl.clean_plot(ax_hd[0, 1], 1)
        # gpl.clean_plot(ax_hd[0, 0], 0)
        # gpl.clean_plot(ax_hd[0, 2], 0)
        # gpl.clean_plot_bottom(ax_hd[0, 1])
        # gpl.clean_plot_bottom(ax_hd[0, 0])
        # gpl.clean_plot_bottom(ax_hd[0, 2])
        # gpl.make_yaxis_scale_bar(ax_hd[0, 0], .8)
        # ax_hd[0, 0].set_ylabel('PC 1')
        # gpl.make_yaxis_scale_bar(ax_hd[0, 2], .8)        
        # ax_hd[0, 2].set_ylabel('PC P')
            
        rs_close = self.params.getlist('manifold_radii_close', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        dc.plot_diagnostics(fdg, pass_model, rs_close, n_arcs, plot_source=True,
                            dim_red=False, square=False,
                            scale_mag=.2, markers=False, ax=axs[0, 0])
        axs[0, 0].set_xlabel('PC 1')
        axs[0, 0].set_ylabel('PC 2')
        rs = self.params.getlist('manifold_radii', typefunc=float)
        dc.plot_diagnostics(fdg, pass_model, rs, n_arcs, plot_source=True,
                            dim_red=False, 
                            scale_mag=.2, markers=False, ax=axs[0, 1])
        axs[0, 1].set_xlabel('PC 1')
        
    def panel_training_rep(self):
        key = self.panel_keys[1]
        train_ax, rep_axs = self.gss[key]

        out = self._generate_panel_training_rep_data()
            
        fdg, (models, th), (p, _), (_, scrs, _) = out[0][:4]
        n_parts, n_epochs = out[1]

        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        npart_signifier = self.params.get('npart_signifier')
        mid_i = np.floor(len(n_parts)/2)
        vis_3d = self.params.getboolean('vis_3d')
        view_inits = (None, (50, 30), (40, -20), (40, -20), (40, -20))
        for i, num_p in enumerate(n_parts):
            hist = th[0, i, 0]['loss']
            epochs = np.arange(1, len(hist) + 1)
            train_ax.plot(epochs, hist,
                          label='r${} = {}$'.format(npart_signifier,
                                                    num_p))
            dc.plot_source_manifold(fdg, models[0, i, 0], rs, n_arcs, 
                                    source_scale_mag=.5,
                                    rep_scale_mag=10, plot_model_3d=vis_3d,
                                    markers=False, axs=rep_axs[i],
                                    titles=False, model_view_init=view_inits[i])
            if mid_i != i:
                rep_axs[i, 0].set_ylabel('')
                rep_axs[i, 1].set_ylabel('')
            if i < len(n_parts) - 1:
                rep_axs[i, 0].set_xlabel('')
                rep_axs[i, 1].set_xlabel('')
        gpl.clean_plot(train_ax, 0)
        train_ax.set_yscale('log')

    def panel_rep_summary(self):
        key = self.panel_keys[2]
        axs = self.gss[key]
        
        run_ind = self.params.get('rep_summary_run')
        f_pattern = self.params.get('f_pattern')
        path = self.params.get('mp_simulations_path')
        part_color = self.params.getcolor('partition_color')

        pv_mask = np.array([False, True, False])
        axs = np.expand_dims(axs, 0)
        dc.plot_recon_gen_summary(run_ind, f_pattern, log_x=False, 
                                  collapse_plots=True, folder=path,
                                  axs=axs, print_args=False,
                                  pv_mask=pv_mask,
                                  set_title=False, color=part_color)

class Figure2Context(Figure2):
    
    def __init__(self, fig_key='figure2_context', **kwargs):
        super().__init__(fig_key=fig_key, **kwargs)
        

class Figure2TwoD(Figure2):
    
    def __init__(self, fig_key='figure2_2d', **kwargs):
        super().__init__(fig_key=fig_key, **kwargs)

        
class Figure2Alt(Figure2):

    def make_gss(self):
        gss = {}

        inp_grid = pu.make_mxn_gridspec(self.gs, 1, 2,
                                            50, 68, 10, 50,
                                            5, 10)
        high_d_grid = (self.gs[75:, :5],)
        hypoth_grids = pu.make_mxn_gridspec(self.gs, 1, 2,
                                            75, 100, 10, 50,
                                            5, 5)
        gss[self.panel_keys[0]] = (self.get_axs(inp_grid),
                                   self.get_axs(high_d_grid),
                                   self.get_axs(hypoth_grids))
        
        train_grid = self.gs[:15, 35:55]
        train_ax = self.get_axs((train_grid,))[0]
        n_parts = len(self.params.getlist('n_parts'))
        rep_grids = pu.make_mxn_gridspec(self.gs, n_parts, 2,
                                         0, 65, 60, 100,
                                         5, 0)
        rep_axs = self.get_axs(rep_grids, sharex='vertical',
                               sharey='vertical')
        gss[self.panel_keys[1]] = train_ax, rep_axs


        res_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 60, 100, 55, 100,
                                         10, 10)
        res_axs = self.get_axs(res_grids)
        
        gss[self.panel_keys[2]] = res_axs[0]

        gss[self.panel_keys[3]] = res_axs[1]
        self.gss = gss

    
    def panel_training_rep(self):
        key = self.panel_keys[1]
        train_ax, rep_axs = self.gss[key]
        
        out = self._generate_panel_training_rep_data()
        fdg, (models, th), (p, _), (_, scrs, _) = out[0][:4]
        n_parts, n_epochs = out[1]

        vis_3d = self.params.getboolean('vis_3d')
        colormap = self.params.get('vis_colormap')
        cmap = plt.get_cmap(colormap)
        task_colors = cmap([.1, .9])
        for i, num_p in enumerate(n_parts):
            dc.plot_grid(fdg, models[0, i, 0], ms=.5, ax=rep_axs[i, 0],
                         colormap=colormap)

            dc.plot_task_reps(fdg, models[0, i, 0], axs=(rep_axs[i, 1],),
                              plot_tasks=(0,), colors=task_colors,
                              density=True)
            gpl.clean_plot(rep_axs[i, 1], 1)
            gpl.make_xaxis_scale_bar(rep_axs[i, 1], 10)
            gpl.make_xaxis_scale_bar(rep_axs[i, 1], 10)
            gpl.make_yaxis_scale_bar(rep_axs[i, 1], .02, double=False)
            
        #     hist = th[0, i, 0]['loss']
        #     epochs = np.arange(1, len(hist) + 1)
        #     train_ax.plot(epochs, hist,
        #                   label='r${} = {}$'.format(npart_signifier,
        #                                             num_p))
        #     dc.plot_source_manifold(fdg, models[0, i, 0], rs, n_arcs, 
        #                             source_scale_mag=.5,
        #                             rep_scale_mag=10, plot_model_3d=vis_3d,
        #                             markers=False, axs=rep_axs[i],
        #                             titles=False, model_view_init=view_inits[i])
        #     if mid_i != i:
        #         rep_axs[i, 0].set_ylabel('')
        #         rep_axs[i, 1].set_ylabel('')
        #     if i < len(n_parts) - 1:
        #         rep_axs[i, 0].set_xlabel('')
        #         rep_axs[i, 1].set_xlabel('')
        # gpl.clean_plot(train_ax, 0)
        # train_ax.set_yscale('log')
    

class Figure4Beta(DisentangledFigure):

    def __init__(self, fig_key='figure4beta', colors=colors, **kwargs):
        fsize = (5.5, 3.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.panel_keys = ('bvae_schematic', 'bvae_performance')
        super().__init__(fsize, params, colors=colors, **kwargs)        
        
    def make_gss(self):
        gss = {}

        bvae_schematic_grid = self.gs[:, :45]
        
        bv1_perf = pu.make_mxn_gridspec(self.gs, 1, 2, 0, 44, 55, 100,
                                        8, 0)
        bv2_perf = pu.make_mxn_gridspec(self.gs, 1, 2, 56, 100, 55, 100,
                                        8, 15)
        bv_perf = np.concatenate((bv1_perf, bv2_perf), axis=0)
        vis_3d = self.params.getboolean('vis_3d')
        axs_3ds = np.zeros((2, 2), dtype=bool)
        axs_3ds[0, 1] = vis_3d
        gss[self.panel_keys[0]] = self.get_axs((bvae_schematic_grid,))
        gss[self.panel_keys[1]] = self.get_axs(bv_perf, plot_3ds=axs_3ds)

        self.gss = gss

    def panel_bvae_performance(self):
        key = self.panel_keys[1]
        axs = self.gss[key]
        if not key in self.data.keys():
            fdg = self.make_fdg()

            out = train_eg_bvae(fdg, self.params)

            c_reps = self.params.getint('dg_classifier_reps')
            m = out[0]
            gen_perf = characterize_generalization(fdg, m[0, 0], c_reps)
            self.data[key] = (fdg, m, gen_perf)
        fdg, m, gen_perf = self.data[key]

        run_inds = (self.params.get('beta_eg_ind'),)
        f_pattern = self.params.get('beta_f_pattern')
        folder = self.params.get('beta_simulations_path')
        labels = (r'$\beta$VAE',)

        bvae_color = self.params.getcolor('bvae_color')
        colors = (bvae_color,)
        
        m[0, 0].p_vectors = []
        m[0, 0].p_offsets = []
        pv_mask = np.array([False, True, False])
        axs_flat = np.concatenate((axs[0], axs[1]))
        self._standard_panel(fdg, m, run_inds, f_pattern, folder, axs_flat,
                             labels=labels, pv_mask=pv_mask,
                             xlab=r'$\beta$', colors=colors,
                             rep_scale_mag=.01)
        
        
class Figure3(DisentangledFigure):
    
    def __init__(self, fig_key='figure3prf', colors=colors, **kwargs):
        fsize = (5.5, 3.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.panel_keys = ('unbalanced_partitions', 'contextual_partitions',
                           'partial_information')
        super().__init__(fsize, params, colors=colors, **kwargs)        
        
    def make_gss(self):
        gss = {}

        unbalanced_latent_grid = self.gs[:30, :20]
        unbalanced_rep_grid = self.gs[:30:, 30:45]
        unbalanced_class_grid = self.gs[:30, 55:70]
        unbalanced_regress_grid = self.gs[:30, 80:]

        axs_3d = np.zeros((1, 4), dtype=bool)
        axs_3d[0, 1] = self.params.getboolean('vis_3d')
        axs_left = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 100, 0, 40,
                                        3, 0)
        axs_right = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 100, 54, 100,
                                         5, 15)
        gss[self.panel_keys[0]] = self.get_axs(np.concatenate((axs_left[0],
                                                               axs_right[0])),
                                               plot_3ds=axs_3d,
                                               squeeze=True)
        gss[self.panel_keys[1]] = self.get_axs(np.concatenate((axs_left[1],
                                                               axs_right[1])),
                                               plot_3ds=axs_3d,
                                               squeeze=True)
        gss[self.panel_keys[2]] = self.get_axs(np.concatenate((axs_left[2],
                                                               axs_right[2])),
                                               plot_3ds=axs_3d,
                                               squeeze=True)

        self.gss = gss

    def panel_unbalanced_partitions(self):
        key = self.panel_keys[0]
        axs = self.gss[key]
        if not key in self.data.keys():
            fdg = self.make_fdg()
            out = train_eg_fd(fdg, self.params)
            self.data[key] = (fdg, out)
        fdg, out = self.data[key]
        m, _ = out

        run_inds = self.params.getlist('unbalanced_eg_inds')
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        labels = ('balanced', 'unbalanced', 'very unbalanced')

        part_color = self.params.getcolor('partition_color')
        unbal1_color = self.params.getcolor('unbalance_color1')
        unbal2_color = self.params.getcolor('unbalance_color2')
        colors = (part_color, unbal1_color, unbal2_color)

        rep_scale_mag = 20

        pv_mask = np.array([False, False, True])
        self._standard_panel(fdg, m, run_inds, f_pattern, folder, axs,
                             labels=labels, pv_mask=pv_mask, 
                             rep_scale_mag=rep_scale_mag, colors=colors)
        for ax in axs:
            ax.set_xlabel('')
            ax.set_ylabel('')

    def panel_contextual_partitions(self):
        key = self.panel_keys[1]
        axs = self.gss[key]
        if not key in self.data.keys():
            fdg = self.make_fdg()
            out = train_eg_fd(fdg, self.params, contextual_partitions=True,
                              offset_var=True)
            self.data[key] = (fdg, out)
        fdg, out = self.data[key]
        m, _ = out

        run_inds = self.params.getlist('contextual_eg_inds')
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        rep_scale_mag = 20

        part_color = self.params.getcolor('partition_color')
        context_color = self.params.getcolor('contextual_color')
        context_offset_color = self.params.getcolor('contextual_offset_color')
        colors = (part_color, context_color, context_offset_color)
        
        labels = ('full tasks', 'contextual tasks',
                  'offset contextual tasks')
        # pv_mask = np.array([False, False, False, False, True, False, False,
        #                     False])
        pv_mask = np.array([False, False, True])
        
        self._standard_panel(fdg, m, run_inds, f_pattern, folder, axs,
                             labels=labels, pv_mask=pv_mask,
                             rep_scale_mag=rep_scale_mag, colors=colors,
                             view_init=(45, -30))
        for ax in axs[:2]:
            ax.set_xlabel('')

    def panel_partial_information(self):
        key = self.panel_keys[2]
        axs = self.gss[key]

        nan_salt_eg = self.params.getfloat('nan_salt_eg')
        if not key in self.data.keys():
            fdg = self.make_fdg()
            out = train_eg_fd(fdg, self.params, nan_salt=nan_salt_eg,
                              offset_var=True)
            self.data[key] = (fdg, out)
        fdg, out = self.data[key]
        m, _ = out

        run_inds = self.params.getlist('partial_eg_inds')
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        rep_scale_mag = 20

        part_color = self.params.getcolor('partition_color')
        partial_color1 = self.params.getcolor('partial_color1')
        partial_color2 = self.params.getcolor('partial_color2')
        colors = (part_color, partial_color1, partial_color2)
        
        labels = ('full information', '50% missing', 'single task')
        # pv_mask = np.array([False, False, False, True, False])
        pv_mask = np.array([False, False, True])
        self._standard_panel(fdg, m, run_inds, f_pattern, folder, axs,
                             labels=labels, pv_mask=pv_mask,
                             rep_scale_mag=rep_scale_mag, colors=colors)
        for ax in axs[:2]:
            ax.set_xlabel('')
            
class Figure3Grid(DisentangledFigure):

    def __init__(self, fig_key='figure3grid', colors=colors, **kwargs):
        fsize = (5.5, 4.8)
        cf = u.ConfigParserColor()
        cf.read(config_path)

        params = cf[fig_key]
        self.panel_keys = ('task_manipulations', 'irrel_variables',
                           'correlation_decay', 'grid_only',
                           'mixed')
        super().__init__(fsize, params, colors=colors, **kwargs)
        # self.fdg = self.data.get('fdg')

    def make_gss(self):
        gss = {}
        
        gs_schem = pu.make_mxn_gridspec(self.gs, 4, 2, 0, 100, 0, 40,
                                        3, 0)
        axs_3d = np.zeros((4, 2), dtype=bool)
        axs_3d[:, 1] = self.params.getboolean('vis_3d')
        axs_schem = self.get_axs(gs_schem, plot_3ds=axs_3d)
        
        gs_res = pu.make_mxn_gridspec(self.gs, 4, 2, 0, 100, 54, 100,
                                       8, 12)
        axs_res = self.get_axs(gs_res)
        axs_res2 = np.concatenate((axs_schem[:, 1:], axs_res), axis=1)
        axs_schem2 = axs_schem[3, 0]

        gss[self.panel_keys[0]] = axs_res2[0]
        gss[self.panel_keys[1]] = axs_res2[1]
        gss[self.panel_keys[2]] = axs_schem2
        gss[self.panel_keys[3]] = axs_res2[2]
        gss[self.panel_keys[4]] = axs_res2[3]
        self.gss = gss

    def panel_task_manipulations(self):
        key = self.panel_keys[0]
        axs = self.gss[key]

        if not key in self.data.keys():
            fdg = self.make_fdg()
            out = train_eg_fd(fdg, self.params, contextual_partitions=True,
                              offset_var=True)
            self.data[key] = (fdg, out)
        fdg, out = self.data[key]
        m, _ = out

        run_inds = self.params.getlist('manip_eg_inds')
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        rep_scale_mag = 20

        unbal_color = self.params.getcolor('unbalance_color1')
        context_color = self.params.getcolor('contextual_color')
        partial_color = self.params.getcolor('partial_color2')
        colors = (unbal_color, context_color, partial_color)
        
        labels = ('unbalanced tasks', 'contextual tasks',
                  'single task examples')
        # pv_mask = np.array([False, False, False, False, True, False, False,
        #                     False])
        pv_mask = np.array([False, False, True])
        
        self._standard_panel(fdg, m, run_inds, f_pattern, folder, axs,
                             labels=labels, pv_mask=pv_mask,
                             rep_scale_mag=rep_scale_mag, colors=colors,
                             view_init=(45, -30))
        for ax in axs[:2]:
            ax.set_xlabel('')

        
    def panel_irrel_variables(self):
        key = self.panel_keys[1]
        axs = self.gss[key]

        if not key in self.data.keys():
            fdg = self.make_fdg()
            irrel_dims = self.params.getlist('irrel_dims', typefunc=int)
            irrel_dims = np.array(irrel_dims).astype(bool)
            out = train_eg_fd(fdg, self.params, offset_var=False,
                              no_learn_lvs=irrel_dims)
            self.data[key] = (fdg, out)
        fdg, out = self.data[key]
        m, _ = out

        run_inds = self.params.getlist('no_learn_eg_ind')
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        multi_num = self.params.getint('multi_num')
        rep_scale_mag = 20

        grid2_color = self.params.getcolor('partition_color')
        grid3_color = self.params.getcolor('untrained_color')
        colors = (grid2_color, grid3_color)
        
        labels = ('trained dimensions', 'untrained dimensions')
        pv_mask = np.array([False, False, True])
        
        self._standard_panel(fdg, m, run_inds, f_pattern, folder, axs,
                             labels=labels, pv_mask=pv_mask,
                             rep_scale_mag=rep_scale_mag, colors=colors,
                             multi_num=multi_num,
                             view_init=(45, -30))
        # for ax in axs:
        #     ax.set_xlabel('')
        #     ax.set_xticks([])
        
        
    def panel_correlation_decay(self):
        key = self.panel_keys[2]
        ax = self.gss[key]

        eg_dim = self.params.getint('inp_dim')
        n_samples = self.params.getint('n_corr_samples')
        partition_color = self.params.getcolor('partition_color')
        grid2_color = self.params.getcolor('grid2_color')
        grid3_color = self.params.getcolor('grid3_color')
        
        part_corr = dt.norm_dot_product(eg_dim)
        grid2_corr = dt.binary_dot_product(2, eg_dim)
        grid3_corr = dt.binary_dot_product(3, eg_dim)
        ax.hist(part_corr, histtype='step', color=partition_color,
                label='partition tasks')
        ax.hist(grid2_corr, histtype='step', color=grid2_color,
                label=r'$N_{C} = 2^{D}$')
        ax.hist(grid3_corr, histtype='step', color=grid3_color,
                label=r'$N_{C} = 3^{D}$')
        ax.legend(frameon=False)
        gpl.clean_plot(ax, 0)
        ax.set_xlabel('task alignment')
        
    def panel_grid_only(self):
        key = self.panel_keys[3]
        axs = self.gss[key]
        
        if not key in self.data.keys():
            fdg = self.make_fdg()
            n_grids = self.params.getint('n_grid_eg')
            out = train_eg_fd(fdg, self.params, offset_var=False,
                              grid_coloring=True, n_granules=3)
            self.data[key] = (fdg, out)
        fdg, out = self.data[key]
        m, _ = out

        run_inds = self.params.getlist('grid_eg_inds')
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        rep_scale_mag = 20

        grid2_color = self.params.getcolor('grid2_color')
        grid3_color = self.params.getcolor('grid3_color')
        grid_style = self.params.get('grid_style')
        colors = (grid2_color, grid3_color)
        
        labels = ('grid = 2', 'grid = 3')
        pv_mask = np.array([False, False, True])
        
        self._standard_panel(fdg, m, run_inds, f_pattern, folder, axs,
                             labels=labels, pv_mask=pv_mask,
                             rep_scale_mag=rep_scale_mag, colors=colors,
                             view_init=(45, -30), linestyle=grid_style,
                             set_lims=False)
        axs[1].set_ylim([.5, 1])
        axs[2].set_ylim([-1, 1])
        axs[2].set_yticks([-1, 0, 1])
        gpl.add_hlines(0, axs[2], linewidth=1)
        gpl.add_vlines(fdg.input_dim, axs[1])
        gpl.add_vlines(fdg.input_dim, axs[2])
        for ax in axs:
            ax.set_xlabel('')
            # ax.set_xticks([])


    def panel_mixed(self):
        key = self.panel_keys[4]
        axs = self.gss[key]
        if not key in self.data.keys():
            fdg = self.make_fdg()
            n_grids = self.params.getint('n_grid_eg')
            out = train_eg_fd(fdg, self.params, n_grids=n_grids)
            self.data[key] = (fdg, out)
        fdg, out = self.data[key]
        m, _ = out

        run_inds = self.params.getlist('mixed_eg_inds')
        f_pattern = self.params.get('f_mixed_pattern')
        folder = self.params.get('mp_simulations_path')
        rep_scale_mag = 20

        mixed2_color = self.params.getcolor('grid2_color')
        mixed3_color = self.params.getcolor('grid3_color')
        colors = (mixed2_color, mixed3_color)

        marker_color = self.params.getcolor('marker_color')
        
        labels = (r'$N_{C} = 2^{D}$',
                  r'$N_{C} = 3^{D}$')
        pv_mask = np.array([False, False, True])

        self._standard_panel(fdg, m, run_inds, f_pattern, folder, axs,
                             labels=labels, pv_mask=pv_mask,
                             rep_scale_mag=rep_scale_mag, colors=colors,
                             view_init=(45, -30), distr_parts='n_grids',
                             plot_hline=False)
        for ax in axs[1:]:
            ax.set_xlabel('grid tasks')
            gpl.add_vlines(15, ax=ax, linestyle='dashed')
        

class FigureGP(DisentangledFigure):

    def __init__(self, fig_key='figure_gp', colors=colors, **kwargs):
        fsize = (7, 6.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.panel_keys = ('gp_inputs', 'dp_tasks', 'gp_results',
                           'gp_input_characterization',
                           'gp_task_characterization',
                           'gp_task_schem',
                           'gp_input_schem')
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gp_dgs(self, retrain=False):
        if self.data.get('gp_dgs') is not None and not retrain:
            gp_dgs = self.data.get('gp_dgs')
        else:
            inp_dim = self.params.getint('inp_dim')
            dg_dim = self.params.getint('dg_dim')
            fit_samples = self.params.getint('fit_samples')
            length_scales = self.params.getlist('dg_scales', typefunc=float)
            gp_dgs = np.zeros(len(length_scales), dtype=object)
            for i, ls in enumerate(length_scales):
                dg_gp = dg.GaussianProcessDataGenerator(inp_dim, 100, dg_dim,
                                                        length_scale=ls)
                dg_gp.fit(train_samples=fit_samples)
                gp_dgs[i] = dg_gp
            self.data['gp_dgs'] = gp_dgs
        return gp_dgs

    def make_gp_tasks(self, retrain=False):
        if self.data.get('gp_tasks') is not None and not retrain:
            gp_tasks = self.data.get('gp_tasks')
        else:
            task_dim = self.params.getint('task_dim')
            length_scales = self.params.getlist('task_scales', typefunc=float)
            n_tasks = self.params.getint('n_tasks_per_dim')
            
            gp_tasks = np.zeros((len(length_scales), n_tasks), dtype=object)
            for i, ls in enumerate(length_scales):
                out = dd.make_tasks(task_dim, n_tasks, use_gp_tasks=True,
                                    gp_task_length_scale=ls)
                func, _, _ = out
                gp_tasks[i] = func
            self.data['gp_tasks'] = gp_tasks
        return gp_tasks

    def make_gss(self):
        gss = {}

        n_dgs = len(self.params.getlist('dg_scales', typefunc=float))
        dg_grid = pu.make_mxn_gridspec(self.gs, 1, n_dgs, 33, 60,
                                       0, 50, 5, 0)
        axs_3ds = np.ones((1, n_dgs), dtype=bool)
        gss[self.panel_keys[0]] = self.get_axs(dg_grid, plot_3ds=axs_3ds,
                                               squeeze=True)

        n_tasks = len(self.params.getlist('task_scales', typefunc=float))
        task_grid = pu.make_mxn_gridspec(self.gs, 1, n_tasks, 33, 60,
                                         55, 100, 5, 2)
        gss[self.panel_keys[1]] = self.get_axs(task_grid, squeeze=True,
                                               aspect='equal')

        res_c_grid = self.gs[75:100, 25:50]
        res_c_cb_grid = self.gs[75:100, 52:54]
        res_r_grid = self.gs[75:100, 65:90]
        res_r_cb_grid = self.gs[75:100, 92:94]
        res_axs = self.get_axs(((res_c_grid, res_r_grid),
                                (res_c_cb_grid, res_r_cb_grid)))
        gss[self.panel_keys[2]] = res_axs

        inp_char_grid = pu.make_mxn_gridspec(self.gs, 1, 4, 60, 70,
                                             20, 100, 5, 10)
        char_axs = self.get_axs(inp_char_grid, squeeze=True)
        gss[self.panel_keys[3]] = char_axs[:-1]
        gss[self.panel_keys[4]] = char_axs[-1]

        task_schem = pu.make_mxn_gridspec(self.gs, 2, 1, 5, 25,
                                          68, 80, 16, 5)
        task_bar = self.get_axs(task_schem, squeeze=False,
                                sharex=True, sharey=True)
        task_schem = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 30,
                                          90, 100, 10, 5)
        resp_bar = self.get_axs(task_schem, squeeze=False,
                                sharex=True, sharey=True)
        ts_arr = np.concatenate((task_bar, resp_bar), axis=1)
        gss[self.panel_keys[5]] = ts_arr

        input_schem = pu.make_mxn_gridspec(self.gs, 3, 1, 0, 30,
                                           15, 30, 5, 5)
        input_gp_axs = self.get_axs(input_schem, squeeze=True, sharex=True,
                                    sharey=True)
        input_comb_ax = self.get_axs((self.gs[0:30, 30:60],),
                                     plot_3ds=np.ones((1, 1), dtype=bool),
                                     squeeze=False)
        lv_ax = self.get_axs((self.gs[0:30, :10],),
                             squeeze=False)
        gss[self.panel_keys[6]] = (lv_ax[0, 0], input_gp_axs, input_comb_ax[0, 0])

        self.gss = gss

    def panel_gp_inputs(self, **kwargs):
        key = self.panel_keys[0]
        dg_axs = self.gss[key]

        length_scales = self.params.getlist('dg_scales', typefunc=float)
        dgs = self.make_gp_dgs(**kwargs)
        pass_model = dd.IdentityModel()
        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        vis_3d = self.params.getboolean('vis_3d')

        for i, dg in enumerate(dgs):
            ls = length_scales[i]
            ax = dg_axs[i]
            ax.set_title('LS = {}'.format(ls))
            dc.plot_diagnostics(dg, pass_model, rs, n_arcs, scale_mag=.5,
                                markers=False, ax=ax, plot_3d=vis_3d)


    def panel_gp_input_schem(self, retrain=False):
        key = self.panel_keys[6]
        lv_ax, gp_axs, comb_ax = self.gss[key]
        
        ls = self.params.getfloat('eg_input_ls')
        if self.data.get(key) is None or retrain:
            gp_inp = dg.GaussianProcessDataGenerator(1, 1, 3,
                                                     length_scale=ls)
            gp_inp.fit()

            mesh_pts = self.params.getint('mesh_pts')
            task_range = self.params.getfloat('task_range')
            
            lv1 = np.zeros((mesh_pts, 1))
            lv1[:, 0] = np.linspace(-task_range, task_range, mesh_pts)

            rep1 = gp_inp.get_representation(lv1)

            self.data[key] = ((lv1, rep1), gp_inp)
        (lv1, rep1), gp_inp = self.data[key]

        cmap1, _ = self.params.getlist('schem_cmaps')
        
        gpl.plot_colored_line(np.zeros(len(lv1)), lv1[:, 0], ax=lv_ax,
                              cmap=cmap1)

        for i in range(rep1.shape[1]):
            gpl.plot_colored_line(lv1[:, 0], rep1[:, i], ax=gp_axs[i],
                                  cmap=cmap1)
            gpl.clean_plot(gp_axs[i], 0)
            gp_axs[i].set_ylabel('GP {}'.format(i+1))

        gp_axs[-1].set_xlabel('LV 1')
        gpl.plot_colored_line(rep1[:, 0], rep1[:, 1], rep1[:, 2], ax=comb_ax,
                              cmap=cmap1)

        lv_ax.set_aspect('equal')
        gpl.clean_plot(lv_ax, 0)
        # gpl.make_xaxis_scale_bar(lv_ax, 1)
        lv_ax.set_xlim([-.2, .2])
        gpl.make_yaxis_scale_bar(lv_ax, 1)
        gpl.clean_plot_bottom(lv_ax)

        lv_ax.set_ylabel('LV 1')

        gpl.clean_3d_plot(comb_ax)
        gpl.make_3d_bars(comb_ax, bar_len=1)

    def _gen_eg_tasks(self, ls, n_tasks, task_dims=1, retrain=False):
        mesh_pts = self.params.getint('mesh_pts')
        task_range = self.params.getfloat('task_range')
        task_vals = np.linspace(-task_range, task_range, mesh_pts)
        task_coords = np.array(list(it.product(range(mesh_pts),
                                                   repeat=task_dims)))
        task_maps = []
        for i in range(n_tasks):
            task = dg.GaussianProcessDataGenerator(task_dims, 1, n_tasks,
                                                   length_scale=ls)
            task.fit()

            task_func = lambda x: task.get_representation(x)[:, i]
            if task_dims > 1:
                task_map = self._get_task_map(task_func, task_coords,
                                              task_vals)
            else:
                task_map = task_func(np.expand_dims(task_vals, 1))
                mu = np.mean(task_map)
                sigma = np.std(task_map)
                task_map = (task_map - mu)/sigma
            task_maps.append(task_map)
        return (task_range, task_vals, task_maps, task_dims)
        
    def panel_gp_task_schem(self, retrain=False):
        key = self.panel_keys[5]
        task_axs = self.gss[key]

        n_tasks = task_axs.shape[0]
        ls = self.params.getfloat('eg_task_ls')
        if self.data.get(key) is None or retrain:
            self.data[key] = self._gen_eg_tasks(ls, n_tasks)
            
        task_range, task_vals, task_maps, task_dims = self.data.get(key)
        if task_dims > 1:
            self._plot_task_map(task_range, task_vals, task_maps, task_axs)
        else:
            self._plot_task_line(task_range, task_vals, task_maps, task_axs)

    def _plot_task_line(self, task_range, task_vals, task_maps, task_axs):
        cmap = self.params.get('task_cmap')
        cmap = plt.get_cmap(cmap)
        for i, task_map in enumerate(task_maps):
            extreme = np.max(np.abs(task_map))

            col_func = lambda x, y: (y + extreme)/(2*extreme)
            gpl.plot_colored_line(task_vals, task_map, func=col_func,
                                  cmap=cmap, ax=task_axs[i, 1], norm=None)
            gpl.add_hlines(0, task_axs[i, 1])
            
            tm = np.expand_dims(task_map, 0)
            tm_new = np.zeros_like(tm)
            tm_new[tm > 0] = extreme
            tm_new[tm <= 0] = -extreme
            n_tiles = 3
            tm_tile = np.concatenate((tm_new,)*n_tiles, axis=0)
            m = gpl.pcolormesh(task_vals, np.linspace(-.1, .1, n_tiles),
                               tm_tile,
                               ax=task_axs[i, 0], cmap=cmap, vmin=-extreme,
                               vmax=extreme)
            task_axs[i, 1].set_xticks([-task_range, 0, task_range])
            task_axs[i, 0].set_xticks([-task_range, 0, task_range])
            task_axs[i, 1].set_yticks([-1, 0, 1])
            task_axs[i, 0].set_yticks([])
            gpl.clean_plot(task_axs[i, 0], 0)
            gpl.clean_plot(task_axs[i, 1], 0)
            task_axs[i, 0].spines['left'].set_visible(False)
            task_axs[i, 1].set_ylabel('GP response')
            task_axs[i, 0].set_xlabel('LV 1')
            if i < len(task_maps) - 1:
                gpl.clean_plot_bottom(task_axs[i, 1], 0)
            else:
                task_axs[i, 1].set_xlabel('LV 1')

    def _plot_task_map(self, task_range, task_vals, task_maps):
        for i, task_map in enumerate(task_maps):
            extreme = np.max(np.abs(task_map))
            tm = np.diff(task_map > 0, n=1, axis=0)
            tm_x, tm_y = np.where(tm)
            tm_x = task_vals[tm_x]
            tm_y = task_vals[tm_y]
            m = gpl.pcolormesh(task_vals, task_vals, task_map, ax=task_axs[i, 1],
                               cmap=cmap, vmin=-extreme, vmax=extreme)
            task_axs[i, 0].plot(tm_y, tm_x, color='k', linewidth=1)
            gpl.pcolormesh(task_vals, task_vals, task_map > 0, ax=task_axs[i, 0],
                           cmap=cmap, vmin=0, vmax=1)
            task_axs[i, 0].set_xticks([-task_range, 0, task_range])
            task_axs[i, 0].set_yticks([-task_range, 0, task_range])
            task_axs[i, 0].set_aspect('equal')
            task_axs[i, 1].set_xticks([-task_range, 0, task_range])
            task_axs[i, 1].set_yticks([-task_range, 0, task_range])
            task_axs[i, 1].set_aspect('equal')

        cb = self.f.colorbar(m, ax=task_axs)
        cb.set_label('GP response')
            
    def _get_task_map(self, func, task_coords, task_vals):
        task_map = np.zeros((len(task_vals), len(task_vals)))
        for (i, j) in task_coords:
            pt = np.array([[task_vals[i], task_vals[j]]])
            out = func(pt)
            task_map[i, j] = out
        sigma = np.std(task_map)
        mu = np.mean(task_map)
        task_map = (task_map - mu)/sigma
        return task_map            

    def panel_gp_tasks(self, **kwargs):
        key = self.panel_keys[1]
        task_axs = self.gss[key][::-1]

        length_scales = self.params.getlist('task_scales', typefunc=float)
        tasks = self.make_gp_tasks(**kwargs)
        mesh_pts = self.params.getint('mesh_pts')
        task_range = self.params.getfloat('task_range')
        task_vals = np.linspace(-task_range, task_range, mesh_pts)
        cmap = self.params.get('task_cmap')
        task_ind = self.params.getint('task_ind')
        task_dim = 2

        task_coords = np.array(list(it.product(range(mesh_pts),
                                               repeat=task_dim)))
        for i, task in enumerate(tasks):
            task = task[task_ind]
            ls = length_scales[i]
            ax = task_axs[-i]
            ax.set_xlabel('LS = {:.0f}'.format(ls))

            task_map = self._get_task_map(task, task_coords, task_vals)
            gpl.pcolormesh(task_vals, task_vals, task_map, ax=ax,
                           cmap=cmap, rasterized=True)            
            ax.set_yticks([])
            ax.set_xticks([-task_range, 0, task_range])
        task_axs[0].set_xticks([-task_range, 0, task_range])


    def panel_gp_results(self):
        key = self.panel_keys[2]
        ((ax_c, ax_r), (ax_c_cb, ax_r_cb)) = self.gss[key]

        pre_str = self.params.get('gp_str')
        folder = self.params.get('mp_simulations_path')
        cmap = self.params.get('results_cmap')

        if self.data.get(key) is None:
            load_out = dsb.read_in_files_from_folder(folder, pre_str=pre_str)
            part_arr = list(sp[0] for sp in load_out['partitions'])
            load_out['single_partition'] = np.array(part_arr)
            load_out['regr_gen_single'] = dsb.complex_field_func(
                load_out['regr_gen'], pre_ind=0)
            out = dsb.construct_matrix(load_out, 'class_gen',
                                       'gp_task_length_scale', 
                                       'gp_length_scale',
                                       'single_partition')
            ax_vals, gp_map_class = out
            out = dsb.construct_matrix(load_out, 'regr_gen_single', 
                                       'gp_task_length_scale', 
                                       'gp_length_scale',
                                       'single_partition')
            ax_vals, gp_map_regr = out
            self.data[key] = (ax_vals, gp_map_class, gp_map_regr)
        ax_vals, gp_map_class, gp_map_regr = self.data[key]

        part_ind = self.params.getint('part_ind')
        eg_ind = self.params.getint('eg_ind')
        comp_ind = self.params.getint('comp_ind')

        proc_gp_class = np.mean(gp_map_class[..., eg_ind, 0, :, comp_ind],
                                axis=-1)
        proc_gp_regr = np.mean(gp_map_regr[..., eg_ind, 0, :, :],
                               axis=(-1, -2))
        proc_gp_regr[proc_gp_regr < 0] = 0

        m = gpl.pcolormesh(ax_vals[1], ax_vals[0], proc_gp_class[..., part_ind],
                           ax_c, equal_bins=True, vmin=.5, vmax=1,
                           cmap=cmap)
        cb_c = self.f.colorbar(m, cax=ax_c_cb)
        cb_c.set_ticks([.5, 1])
        cb_c.set_label('classifier generalization')

        ax_c.set_ylabel('task length scale')
        ax_r.set_xlabel('input length scale')
        ax_c.set_xlabel('input length scale')

        m = gpl.pcolormesh(ax_vals[1], ax_vals[0], proc_gp_regr[..., part_ind],
                           ax_r, equal_bins=True, vmin=0, vmax=1,
                           cmap=cmap)
        cb_r = self.f.colorbar(m, cax=ax_r_cb)
        cb_r.set_ticks([0, .5, 1])
        cb_r.set_label('regression generalization')

    def panel_gp_input_baseline(self, retrain=False):
        key = self.panel_keys[3]
        ax_c, ax_r, ax_d = self.gss[key]
        
        inp_dim = self.params.getint('inp_dim')
        dg_dim = self.params.getint('dg_dim')
        fit_samples = self.params.getint('fit_samples')
        length_scales = self.params.getlist('dg_scales_perf', typefunc=float)
        cmap = self.params.get('results_cmap')

        if self.data.get(key) is None or retrain:
            out = characterize_gaussian_process(inp_dim, dg_dim,
                                                length_scales,
                                                fit_samples=fit_samples)
            self.data[key] = out
        out = self.data[key]
        x_vals = length_scales # np.arange(len(length_scales))
        gpl.plot_trace_werr(x_vals, out[1], ax=ax_c, label='trained')
        gpl.plot_trace_werr(x_vals, out[2], ax=ax_c, label='tested')

        gpl.plot_trace_werr(x_vals, out[3], ax=ax_r)
        gpl.plot_trace_werr(x_vals, out[4], ax=ax_r)

        gpl.plot_trace_werr(x_vals, out[0], ax=ax_d)

        # ax_r.set_xticks(x_vals)
        # ax_r.set_xticklabels(length_scales, rotation=90)
        # ax_c.set_xticks(x_vals)
        # ax_c.set_xticklabels(length_scales, rotation=90)
        # ax_d.set_xticks(x_vals)
        # ax_d.set_xticklabels(length_scales, rotation=90)

        ax_c.set_xscale('log')
        ax_r.set_xscale('log')
        ax_d.set_xscale('log')
        ax_r.set_xlabel('input length scale')
        ax_c.set_ylabel('classifier')
        ax_r.set_ylabel('regression')
        ax_d.set_ylabel('input dim')

    def panel_gp_task_characterization(self, retrain=False):
        key = self.panel_keys[4]
        ax = self.gss[key]

        tasks = self.make_gp_tasks(retrain=retrain)
        task_dim = self.params.getint('task_dim')
        task_scales = self.params.getlist('task_scales', typefunc=float)
        sd = sts.multivariate_normal(np.zeros(task_dim), 1)
        samps = sd.rvs(10000)
        dims = np.zeros(len(task_scales))
        for i, task in enumerate(tasks):
            resps = np.concatenate(list(t(samps) for t in task), axis=1)
            dims[i] = u.participation_ratio(resps)

        gpl.plot_trace_werr(task_scales, dims, ax=ax)
        gpl.add_hlines(task_dim, ax)
        ax.set_xlabel('task length scale')
        ax.set_ylabel('output dim')

class FigureGPRevised(FigureGP):

    
    def make_gss(self):
        gss = {}

        n_dgs = len(self.params.getlist('dg_scales', typefunc=float))
        dg_grid = pu.make_mxn_gridspec(self.gs, 1, n_dgs, 38, 50,
                                       0, 40, 5, 5)
        gss[self.panel_keys[0]] = self.get_axs(dg_grid, 
                                               squeeze=True)

        n_tasks = len(self.params.getlist('task_scales', typefunc=float))
        task_grid = pu.make_mxn_gridspec(self.gs, 1, n_tasks, 33, 60,
                                         65, 100, 5, 2)
        gss[self.panel_keys[1]] = self.get_axs(task_grid, squeeze=True,
                                               aspect='equal')

        res_c_grid = self.gs[75:100, 25:50]
        res_c_cb_grid = self.gs[75:100, 52:54]
        res_r_grid = self.gs[75:100, 65:90]
        res_r_cb_grid = self.gs[75:100, 92:94]
        res_axs = self.get_axs(((res_c_grid, res_r_grid),
                                (res_c_cb_grid, res_r_cb_grid)))
        gss[self.panel_keys[2]] = res_axs

        dim_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 38, 50,
                                        45, 60, 5, 2)
        char_axs = self.get_axs(dim_grid, squeeze=True)
        gss[self.panel_keys[4]] = char_axs[1]

        inp_c_quant_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 60, 70,
                                                25, 50, 5, 10)
        inp_r_quant_grid = pu.make_mxn_gridspec(self.gs, 2, 1, 60, 70,
                                                65, 90, 5, 10)
        inp_quant_grid = np.concatenate((inp_c_quant_grid,
                                         inp_r_quant_grid),
                                        axis=1)
        inp_quant_axs = self.get_axs(inp_quant_grid, share_ax_x=res_axs[0, 0])
        gss[self.panel_keys[3]] = (char_axs[0], inp_quant_axs)

        task_schem = pu.make_mxn_gridspec(self.gs, 2, 1, 5, 25,
                                          68, 80, 16, 5)
        task_bar = self.get_axs(task_schem, squeeze=False,
                                sharex=True, sharey=True)
        task_schem = pu.make_mxn_gridspec(self.gs, 2, 1, 0, 30,
                                          90, 100, 10, 5)
        resp_bar = self.get_axs(task_schem, squeeze=False,
                                sharex=True, sharey=True)
        ts_arr = np.concatenate((task_bar, resp_bar), axis=1)
        gss[self.panel_keys[5]] = ts_arr

        input_schem = pu.make_mxn_gridspec(self.gs, 3, 1, 0, 30,
                                           15, 30, 5, 5)
        input_gp_axs = self.get_axs(input_schem, squeeze=True, sharex=True,
                                    sharey=True)
        input_comb_ax = self.get_axs((self.gs[0:30, 30:60],),
                                     plot_3ds=np.ones((1, 1), dtype=bool),
                                     squeeze=False)
        lv_ax = self.get_axs((self.gs[0:30, :10],),
                             squeeze=False)
        gss[self.panel_keys[6]] = (lv_ax[0, 0], input_gp_axs, input_comb_ax[0, 0])

        self.gss = gss

    def panel_gp_tasks(self, retrain=False):
        key = self.panel_keys[1]
        task_axs = self.gss[key][::-1]

        length_scales = self.params.getlist('task_scales', typefunc=float)
        cmap = self.params.get('task_cmap')
        if self.data.get(key) is None or retrain:
            out = {}
            for ls in length_scales:
                tasks = self._gen_eg_tasks(ls, 1)
                out[ls] = tasks
                
            self.data[key] = out
        
        task_dict = self.data.get(key)
        for i, ls in enumerate(length_scales):
            task_range, task_vals, task_maps, task_dims = task_dict[ls]
            bin_map = task_maps[0] < 0
            gpl.pcolormesh(np.array([0, .2]), task_vals, bin_map, ax=task_axs[i],
                           cmap=cmap, rasterized=True)            
            task_axs[i].set_yticks([-task_range, 0, task_range])
            task_axs[i].set_xticks([])
            task_axs[i].set_xlabel('LS = {:.0f}'.format(ls))

    def panel_gp_inputs(self, **kwargs):
        key = self.panel_keys[0]
        dg_axs = self.gss[key]

        length_scales = self.params.getlist('dg_scales', typefunc=float)
        dgs = self.make_gp_dgs(**kwargs)
        pass_model = dd.IdentityModel()

        mesh_pts = self.params.getint('mesh_pts')
        s_stim = np.expand_dims(np.linspace(-2, 2, mesh_pts), 1)
        o_stim = np.zeros((mesh_pts, 4))
        stim = np.concatenate((s_stim, o_stim), axis=1)
        for i, dg in enumerate(dgs):
            ls = length_scales[i]
            ax = dg_axs[i]
            ax.set_title('LS = {}'.format(ls))
            rep = dg.get_representation(stim)
            ax.plot(s_stim, rep[:, 0])
            gpl.clean_plot(ax, 0)

        
        
    def panel_gp_input_baseline(self, retrain=False):
        key = self.panel_keys[3]
        ax_d, heat_axs = self.gss[key]
        
        inp_dim = self.params.getint('inp_dim')
        dg_dim = self.params.getint('dg_dim')
        fit_samples = self.params.getint('fit_samples')
        length_scales = self.params.getlist('dg_scales_perf', typefunc=float)
        cmap = self.params.get('results_cmap')

        if self.data.get(key) is None or retrain:
            out = characterize_gaussian_process(inp_dim, dg_dim,
                                                length_scales,
                                                fit_samples=fit_samples)
            self.data[key] = out
        out = self.data[key]
        x_vals = length_scales # np.arange(len(length_scales))

        gpl.pcolormesh(x_vals, [0, 1], out[1], equal_bins=True,
                       ax=heat_axs[0, 0], cmap=cmap, vmin=.5, vmax=1)
        gpl.pcolormesh(x_vals, [0, 1], out[2], ax=heat_axs[1, 0],
                       equal_bins=True, cmap=cmap, vmin=.5, vmax=1)

        gpl.pcolormesh(x_vals, [0, 1], out[3], equal_bins=True,
                       ax=heat_axs[0, 1], cmap=cmap, vmin=0, vmax=1)
        gpl.pcolormesh(x_vals, [0, 1], out[4], equal_bins=True,
                       ax=heat_axs[1, 1], cmap=cmap, vmin=0, vmax=1)

        gpl.plot_trace_werr(x_vals, out[0], ax=ax_d)

        ax_d.set_xscale('log')
        ax_d.set_ylabel('input dim')


        
class Figure4(DisentangledFigure):

    def __init__(self, fig_key='figure4', colors=colors, **kwargs):
        fsize = (7, 5.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.panel_keys = ('rf_input', 'disentangling_comparison')
        super().__init__(fsize, params, colors=colors, **kwargs)
        self.rfdg = self.data.get('rfdg')

    def make_rfdg(self, retrain=False, kernel=False):
        if self.rfdg is not None and not retrain:
            rfdg = self.rfdg
        else:
            inp_dim = self.params.getint('inp_dim')
            dg_dim = self.params.getint('dg_dim')
            in_noise = self.params.getfloat('in_noise')
            out_noise = self.params.getfloat('out_noise')
            width_scaling = self.params.getfloat('width_scaling')
            dg_source_var = self.params.getfloat('dg_source_var')
            
            source_distr = sts.multivariate_normal(np.zeros(inp_dim),
                                                   dg_source_var)
            if not kernel:
                rfdg = dg.RFDataGenerator(inp_dim, dg_dim, total_out=True, 
                                          input_noise=in_noise, noise=out_noise,
                                          width_scaling=width_scaling,
                                          source_distribution=source_distr,
                                          low_thr=.01)
            else:
                rfdg = dg.KernelDataGenerator(inp_dim, None, dg_dim,
                                              low_thr=.01)
            self.rfdg = rfdg
        self.data['rfdg'] = rfdg
        return rfdg
        
    def make_gss(self):
        gss = {}

        rf_schematic_grid = self.gs[:40, :20]
        rf_projection_grid = self.gs[:50, 28:45]
        rf_dec_grid = pu.make_mxn_gridspec(self.gs, 1, 2, 60, 100,
                                           0, 45, 5, 14)
        axs_3ds = np.zeros((1, 4), dtype=bool)
        axs_3ds[0, 1] = self.params.getboolean('vis_3d')
        gss[self.panel_keys[0]] = self.get_axs(((rf_schematic_grid,
                                                 rf_projection_grid,
                                                 rf_dec_grid[0, 0],
                                                 rf_dec_grid[0, 1]),),
                                               plot_3ds=axs_3ds)[0]

        rep_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 100,
                                         55, 100, 5, 5)
        axs_3ds = np.zeros((2, 2), dtype=bool)
        axs_3ds[0, :] = self.params.getboolean('vis_3d')

        gss[self.panel_keys[1]] = self.get_axs(rep_grids, plot_3ds=axs_3ds)
        
        self.gss = gss

    def panel_rf_input(self, kernel=False):
        key = self.panel_keys[0]
        schem_ax, proj_ax, dec_c_ax, dec_r_ax = self.gss[key]
        rfdg = self.make_rfdg(kernel=kernel)

        rf_eg_color = self.params.getcolor('rf_eg_color')
        if not kernel:
            rfdg.plot_rfs(schem_ax, color=rf_eg_color, thin=20,
                          x_scale=.5, y_scale=.5, lw=1)
            schem_ax.set_aspect('equal')

        pass_model = dd.IdentityModel()
        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        vis_3d = self.params.getboolean('vis_3d')
        dc.plot_diagnostics(rfdg, pass_model, rs, n_arcs,
                            scale_mag=.5, markers=False, ax=proj_ax,
                            plot_3d=vis_3d)

        if not key in self.data.keys():
            c_reps = self.params.getint('dg_classifier_reps')
            out = characterize_generalization(rfdg, pass_model,
                                              c_reps)
            self.data[key] = out
        results_class, results_regr = self.data[key]

        color = self.params.getcolor('dg_color')
        plot_bgp(results_class, results_regr, dec_c_ax, dec_r_ax,
                 color=color)
        # plot_single_gen(results_class, dec_c_ax, color=color)
        # dec_c_ax.set_ylim([.5, 1])
        # plot_single_gen(results_regr, dec_r_ax, color=color)
        # dec_r_ax.set_ylim([0, 1])
        # dec_c_ax.set_ylabel('classifier\ngeneralization')
        # dec_r_ax.set_ylabel('regression\ngeneralization')

    def panel_disentangling_comparison(self, kernel=None):
        key = self.panel_keys[1]
        axs = self.gss[key]
        rfdg = self.make_rfdg(kernel=kernel)

        if not key in self.data.keys():
            out_fd = train_eg_fd(rfdg, self.params)
            out_bvae = train_eg_bvae(rfdg, self.params)
            m_fd = out_fd[0][0, 0]
            m_bvae = out_bvae[0][0, 0]
            fd_gen = characterize_generalization(rfdg, m_fd, 10)
            bvae_gen = characterize_generalization(rfdg, m_bvae, 10)
            self.data[key] = (out_fd, out_bvae, fd_gen, bvae_gen)
        if len(self.data[key]) > 2:
            out_fd, out_bvae, fd_gen, bvae_gen = self.data[key]
        else:
            out_fd, out_bvae = self.data[key]
        m_fd = out_fd[0][0, 0]
        m_bvae = out_bvae[0][0, 0]

        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        vis_3d = self.params.getboolean('vis_3d')

        # print(np.mean(fd_gen[0], axis=0)) 
        # print(np.mean(fd_gen[1], axis=0))
        # print(np.mean(bvae_gen[0], axis=0))
        # print(np.mean(bvae_gen[1], axis=0))

        run_ind_fd = self.params.get('run_ind_fd')  
        run_ind_beta = self.params.get('run_ind_beta')
        f_pattern = self.params.get('f_pattern')
        beta_f_pattern = self.params.get('beta_f_pattern')
        folder = self.params.get('mp_simulations_path')
        beta_folder = self.params.get('beta_simulations_path')
        dc.plot_diagnostics(rfdg, m_fd, rs, n_arcs, 
                            scale_mag=20, markers=False,
                            ax=axs[0, 0], plot_3d=vis_3d)
        dc.plot_diagnostics(rfdg, m_bvae, rs, n_arcs, 
                            scale_mag=.01, markers=False,
                            ax=axs[0, 1], plot_3d=vis_3d)
        res_axs = axs[1:]

        part_color = self.params.getcolor('partition_color')
        bvae_color = self.params.getcolor('bvae_color')
        xlab = r'tasks / $\beta$'
        
        pv_mask = np.array([False, True, False])
        # pv_mask = np.array([False, False, False, True, False])
        o = dc.plot_recon_gen_summary(run_ind_fd, f_pattern, log_x=False, 
                                  collapse_plots=False, folder=folder,
                                  axs=res_axs, legend='multi-tasking model',
                                  print_args=False, pv_mask=pv_mask,
                                  set_title=False, color=part_color,
                                  xlab=xlab, set_lims=False, ret_info=True)[1]
        pv_mask = np.array([False, True, False])
        o2 = dc.plot_recon_gen_summary(run_ind_beta, beta_f_pattern, log_x=False, 
                                  collapse_plots=False, folder=beta_folder,
                                  axs=res_axs, legend=r'$\beta$VAE',
                                  print_args=False, pv_mask=pv_mask,
                                  set_title=False, color=bvae_color,
                                  xlab=xlab, set_lims=False, ret_info=True)[1]
        print(u.dict_diff(o['args'][0], o2['args'][0]))
        res_axs[0, 0].set_ylim([.5, 1])
        res_axs[0, 1].set_ylim([-30, 1])
        res_axs[0, 1].set_yticks([-30, 0])
        gpl.add_hlines(0, res_axs[0, 1], linewidth=1)
        gpl.add_vlines(rfdg.input_dim, res_axs[0, 0])
        gpl.add_vlines(rfdg.input_dim, res_axs[0, 1])


class SIFigureContext(DisentangledFigure):

    def __init__(self, fig_key='sifigure_context', colors=colors, **kwargs):
        fsize = (4.5, 8)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('extrapolation',
                           'egs',
                           'gen_performance',
                           )
        self.models_key = 'trained_models'
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        grids = pu.make_mxn_gridspec(self.gs, 6, 7, 0, 50, 0, 100,
                                     2, 2)
        grid_axs = self.get_axs(grids, aspect='equal')
        
        grids = pu.make_mxn_gridspec(self.gs, 2, 2, 55, 100, 0, 100,
                                     10, 20)
        axs = self.get_axs(grids)
        gss[self.panel_keys[1]] = grid_axs
        gss[self.panel_keys[0]] = axs[0]
        gss[self.panel_keys[2]] = axs[1]
        self.gss = gss

    def panel_egs(self):
        key = self.panel_keys[1]
        axs = self.gss[key]
        axs_rfs = axs[:3]
        axs_tasks = axs[3:]

        out = self._generate_panel_training_rep_data()
        fdg, (models, _) = out[0][:2]
        models = models[0, :, 0]

        c1_col = self.params.getcolor('c1_color')
        c2_col = self.params.getcolor('c2_color')
        task_cmap = gpl.make_linear_cmap(c2_col, c1_col)
        for i, m in enumerate(models):
            p = m.n_partitions

            func = lambda x: m.get_representation(x).numpy()
            m1, _ = dc.plot_dg_rfs(fdg, func, cmap='viridis',
                                   axs=axs_rfs[:, i], mask=False)
            
            func = lambda x: m.class_model(m.get_representation(x)).numpy()
            m2, _ = dc.plot_func_rfs(fdg, func, cmap='RdBu', mask=False,
                                     axs_flat=axs_tasks[:p, i])
    
            xs = np.linspace(-2, 2, 100)
            max_ax = np.min([p, len(axs_tasks)])
            for j, pf in enumerate(m.p_funcs[:max_ax]):
                mask = dc.make_task_mask(pf)
                gpl.pcolormesh(xs, xs, mask, ax=axs_tasks[j, i], alpha=.3,
                               cmap=task_cmap, rasterized=True)
                axs_tasks[j, i].set_xticks([])
                axs_tasks[j, i].set_yticks([])
        
            axs_rfs[0, i].set_title('P = {}'.format(p), rotation=45)
        self.f.colorbar(m1, ax=axs_rfs)
        
    def panel_gen_performance(self):
        key = self.panel_keys[2]
        axs = np.expand_dims(self.gss[key], 0)

        run_inds = self.params.getlist('extrap_inds')
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        con_color = self.params.getcolor('contextual_color')
        label = 'contextual tasks'
        pv_mask = np.array([True,])

        dc.plot_recon_gen_summary(run_inds, f_pattern, log_x=False, 
                                  collapse_plots=False, folder=folder,
                                  intermediate=False,
                                  pv_mask=pv_mask,
                                  axs=axs, legend=label,
                                  plot_hline=False, set_title=False,
                                  color=con_color,
                                  set_lims=True, list_run_ind=True)
        gpl.add_vlines(5, axs[0, 0])
        gpl.add_vlines(5, axs[0, 1])
        
    def panel_extrapolation(self):
        key = self.panel_keys[0]
        axs = self.gss[key]
        
        extrap_run_inds = self.params.getlist('extrap_inds')
        con_color = self.params.getcolor('contextual_color')
        
        pts = []
        pts_reg = []
        for ri in extrap_run_inds:
            out = self.load_run(ri)
            p = out[1][0]
            n_parts = out[0]
            extrap = out[-1]
            for i, n in enumerate(n_parts):
                reg, extr = extrap[i]['contextual_extrapolation']
                nps = (n,)*extr.shape[-1]
                es = extr[0, 0]
                rs = reg[0, 0]
                pts.extend(zip(nps, es))
                pts_reg.extend(zip(nps, rs))
        pts = np.array(pts)
        pts_reg = np.array(pts_reg)
        axs[1].plot(pts[:, 0], pts[:, 1], 'o', color=con_color)
        axs[0].plot(pts_reg[:, 0], pts_reg[:, 1], 'o', color=con_color)
        axs[0].set_xlabel('tasks')
        axs[1].set_xlabel('tasks')
        axs[0].set_ylabel('trained context accuracy')
        axs[1].set_ylabel('tested context accuracy')
        axs[0].set_ylim([.5, 1])
        axs[1].set_ylim([.5, 1])
        gpl.clean_plot(axs[0], 0)
        gpl.clean_plot(axs[1], 0)
        
        
class SIFigureRandomRF(Figure4):

    def __init__(self, fig_key='sifigure_random_rf', **kwargs):
        super().__init__(fig_key=fig_key, **kwargs)

    def make_gss(self):
        gss = {}

        rf_inp_grid = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 100,
                                           0, 45, 5, 14)
        axs_3ds = np.zeros((3, 2), dtype=bool)
        axs_3ds[0, 1] = self.params.getboolean('vis_3d')
        gss[self.panel_keys[0]] = self.get_axs(rf_inp_grid,
                                               plot_3ds=axs_3ds)

        rep_grids = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 100,
                                         55, 100, 5, 5)
        axs_3ds = np.zeros((3, 2), dtype=bool)
        axs_3ds[0, 1] = self.params.getboolean('vis_3d')
        gss[self.panel_keys[1]] = self.get_axs(rep_grids, plot_3ds=axs_3ds)
        
        self.gss = gss

        
    def make_rfdg(self, kernel=None, **kwargs):
        return self.make_random_rfdg(**kwargs)

    def panel_rf_input(self, kernel=None):
        key = self.panel_keys[0]
        axs = self.gss[key]
        schem_ax, proj_ax = axs[0]
        gen_axs = axs[1]
        dec_c_ax, dec_r_ax = axs[2]
        
        rfdg = self.make_rfdg(kernel=kernel)

        rf_eg_color = self.params.getcolor('rf_eg_color')
        if not kernel:
            rfdg.plot_rfs(schem_ax, color=rf_eg_color, thin=20,
                          x_scale=.5, y_scale=.5, lw=1)
            schem_ax.set_aspect('equal')

        pass_model = dd.IdentityModel()
        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        vis_3d = self.params.getboolean('vis_3d')
        dc.plot_diagnostics(rfdg, pass_model, rs, n_arcs,
                            scale_mag=.5, markers=False, ax=proj_ax,
                            plot_3d=vis_3d)

        grid_len = self.params.getfloat('grid_len')
        grid_pts = self.params.getint('grid_pts')
        dc.plot_class_grid(rfdg, pass_model, grid_pts=grid_pts,
                           grid_len=grid_len,
                           ax=gen_axs[0], col_eps=.15)
        dc.plot_regr_grid(rfdg, pass_model, grid_pts=grid_pts,
                          grid_len=grid_len,
                          ax=gen_axs[1])


        if not key in self.data.keys():
            c_reps = self.params.getint('dg_classifier_reps')
            out = characterize_generalization(rfdg, pass_model,
                                              c_reps)
            self.data[key] = out
        results_class, results_regr = self.data[key]

        color = self.params.getcolor('dg_color')
        plot_bgp(results_class, results_regr, dec_c_ax, dec_r_ax,
                 color=color)
        
    
    def panel_disentangling_comparison(self, kernel=None):
        key = self.panel_keys[1]
        axs = self.gss[key]
        vis_ax = axs[0, 1]
        gen_axs = axs[1]
        quant_axs = axs[2:]
        rfdg = self.make_rfdg()

        if not key in self.data.keys():
            out_fd = train_eg_fd(rfdg, self.params, use_early_stopping=True)
            m_fd = out_fd[0][0, 0]
            fd_gen = characterize_generalization(rfdg, m_fd, 10)
            self.data[key] = (rfdg, m_fd, fd_gen)
        rfdg, m_fd, fd_gen = self.data[key]
 
        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        vis_3d = self.params.getboolean('vis_3d')
        grid_pts = self.params.getint('grid_pts')
        grid_len = self.params.getfloat('grid_len')

        dc.plot_diagnostics(rfdg, m_fd, rs, n_arcs, 
                            scale_mag=20, markers=False,
                            ax=vis_ax, plot_3d=vis_3d)

        dc.plot_class_grid(rfdg, m_fd, grid_pts=grid_pts,
                           grid_len=grid_len,
                           ax=gen_axs[0], col_eps=.15)
        dc.plot_regr_grid(rfdg, m_fd, grid_pts=grid_pts,
                          grid_len=grid_len,
                          ax=gen_axs[1])


        run_ind_fd = self.params.getlist('run_ind_fd')  
        run_ind_beta = self.params.get('run_ind_beta')
        f_pattern = self.params.get('f_pattern')
        beta_f_pattern = self.params.get('beta_f_pattern')
        folder = self.params.get('mp_simulations_path')
        beta_folder = self.params.get('beta_simulations_path')

        part_color = self.params.getcolor('partition_color')
        bvae_color = self.params.getcolor('bvae_color')
        xlab = r'tasks'
        
        pv_mask = np.array([False, True, False])
        o = dc.plot_recon_gen_summary(run_ind_fd, f_pattern, log_x=False, 
                                      collapse_plots=False, folder=folder,
                                      axs=quant_axs, legend='multi-tasking model',
                                      print_args=False, pv_mask=pv_mask,
                                      set_title=False, color=part_color,
                                      xlab=xlab, set_lims=False, ret_info=True,
                                      list_run_ind=True,
                                      intermediate=True,
                                      plot_intermediate=False)[1]
        quant_axs[0, 0].set_ylim([.5, 1])
        quant_axs[0, 1].set_ylim([0, 1])
        quant_axs[0, 1].set_yticks([0, 1])
        gpl.add_hlines(0, quant_axs[0, 1], linewidth=1)
        gpl.add_vlines(rfdg.input_dim, quant_axs[0, 0])
        gpl.add_vlines(rfdg.input_dim, quant_axs[0, 1])

        
class FigureImg(DisentangledFigure):

    def __init__(self, fig_key='figure_img', colors=colors, **kwargs):
        fsize = (6, 6)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.panel_keys = ('preproc_img_learning', 'preproc_img_null', 'img_egs')
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_shape_dg(self, retrain=False):
        try:
            assert not retrain
            shape_dg = self.shape_dg
        except:
            twod_file = self.params.get('shapes_path')
            img_size = tuple(self.params.getlist('img_size', typefunc=int))
            shape_dg = dg.TwoDShapeGenerator(twod_file, img_size=img_size,
                                             max_load=10000,
                                             convert_color=False)
            self.shape_dg = shape_dg 
        return shape_dg

    def make_chair_dg(self, retrain=False):
        try:
            assert not retrain
            chair_dg = self.chair_dg
        except:
            chair_file = self.params.get('chairs_path')
            img_size = tuple(self.params.getlist('img_size', typefunc=int))
            n_chairs = self.params.getint('n_chairs')
            filter_edges = self.params.getfloat('filter_edges')
            chair_dg = dg.ChairGenerator(chair_file, img_size=img_size,
                                         max_load=np.inf,
                                         include_position=True,
                                         max_move=.6, filter_edges=filter_edges,
                                         n_unique_chairs=n_chairs)
            self.chair_dg = chair_dg 
        return chair_dg

    def make_preproc_model(self):
        try:
            preproc_model = self.preproc_model
        except AttributeError:
            preproc_path = self.params.get('preproc_path')
            img_size = tuple(self.params.getlist('img_size', typefunc=int))
            learned_lvs = self.params.getlist('learned_lvs', typefunc=bool)
            preproc_model = dd.PretrainedModel(img_size, preproc_path)
            preproc_model.learn_lvs = np.array(learned_lvs)
            self.preproc_model = preproc_model
        return preproc_model
        
    def make_gss(self):
        gss = {}

        eg_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 30,
                                        0, 30, 4, 4)        
        gss[self.panel_keys[2]] = self.get_axs(eg_grids)
        
        null_grids = pu.make_mxn_gridspec(self.gs, 4, 2, 40, 100,
                                          0, 30, 10, 10)
        null_axs = self.get_axs(null_grids, sharex=True)
        gss[self.panel_keys[1]] = null_axs
        
        preproc_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 50, 95,
                                             50, 100, 3, 10)
        preproc_axs = self.get_axs(preproc_grids, sharex=True)
        gss[self.panel_keys[0]] = preproc_axs

        self.gss = gss

    def make_ident_model(self):
        learned_lvs = self.params.getlist('learned_lvs', typefunc=bool)
        id_model = dd.IdentityModel(flatten=True)
        id_model.learn_lvs = np.array(learned_lvs)
        return id_model

    def panel_img_egs(self, resample=False):
        key = self.panel_keys[2]
        axs = self.gss[key]

        if key not in self.data.keys() or resample:
            shape_dg = self.make_shape_dg()
            _, shape_imgs = shape_dg.sample_reps(2)
            chair_dg = self.make_chair_dg()
            _, chair_imgs = chair_dg.sample_reps(2)
            self.data[key] = (shape_imgs, chair_imgs)

        shape_imgs, chair_imgs = self.data[key]
        all_imgs = np.concatenate((shape_imgs, chair_imgs), axis=0)
        for i, ind in enumerate(u.make_array_ind_iterator(axs.shape)):
            axs[ind].imshow(all_imgs[i])
            axs[ind].set_xticks([])
            axs[ind].set_yticks([])
    
    def panel_preproc_img_null(self):
        key = self.panel_keys[1]
        axs = self.gss[key]

        if not key in self.data.keys():
            shape_dg = self.make_shape_dg()
            chair_dg = self.make_chair_dg()
            preproc_model = self.make_preproc_model()
            ident_model = self.make_ident_model()
                        
            c_reps = self.params.getint('dg_classifier_reps')
            

            id_shape = characterize_generalization(shape_dg, ident_model,
                                                   c_reps, learn_lvs='trained',
                                                   norm=False)
            id_chair = characterize_generalization(chair_dg, ident_model,
                                                   c_reps, learn_lvs='trained',
                                                   norm=False)
            ppm_shape = characterize_generalization(shape_dg, preproc_model,
                                                    c_reps, learn_lvs='trained',
                                                    norm=False)
            ppm_chair = characterize_generalization(chair_dg, preproc_model,
                                                    c_reps, learn_lvs='trained',
                                                    norm=False)
            self.data[key] = (id_shape, id_chair, ppm_shape, ppm_chair)
        id_shape, id_chair, ppm_shape, ppm_chair = self.data[key]

        shape_col = self.params.getcolor('shape_color')
        chair_col = self.params.getcolor('chair_color')
        colors = (shape_col, chair_col)

        labels = ('shape', 'chair')

        plot_multi_bgp((id_shape[0],),
                       (id_shape[1],),
                       axs[0, 0], axs[0, 1], colors=colors,
                       legend_labels=labels, labels=('', ''))
        plot_multi_bgp((ppm_shape[0],),
                       (ppm_shape[1],),
                       axs[1, 0], axs[1, 1], colors=colors,
                       legend_labels=('', ''), rotation=45)

        plot_multi_bgp((id_chair[0],),
                       (id_chair[1],),
                       axs[2, 0], axs[2, 1], colors=colors,
                       legend_labels=labels, labels=('', ''))
        plot_multi_bgp((ppm_chair[0],),
                       (ppm_chair[1],),
                       axs[3, 0], axs[3, 1], colors=colors,
                       legend_labels=('', ''), rotation=45)

    def panel_preproc_img_learning(self):
        key = self.panel_keys[0]
        axs = self.gss[key]

        chair_nom_ind = self.params.getlist('chair_nom_ind')
        chair_gen_ind = self.params.getlist('chair_gen_ind')
        chair_extrap_ind = self.params.getlist('chair_extrap_ind')

        twod_nom_ind = self.params.getlist('twod_nom_ind')
        twod_gen_ind = self.params.getlist('twod_gen_ind')
        twod_extrap_ind = self.params.getlist('twod_extrap_ind')

        # run_inds = ((twod_nom_ind, chair_nom_ind),
        #             (twod_gen_ind, chair_gen_ind),
        #             (twod_extrap_ind, chair_extrap_ind))
        run_inds = ((twod_nom_ind, chair_nom_ind),)
        
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        shape_color = self.params.getcolor('shape_color')
        chair_color = self.params.getcolor('chair_color')

        labels = ('shapes', 'chairs')
        colors = (shape_color, chair_color)

        pv_mask = np.array([True])
        for i, (ri_shape, ri_chair) in enumerate(run_inds):
            dc.plot_recon_gen_summary(ri_shape, f_pattern, log_x=False, 
                                      collapse_plots=False, folder=folder,
                                      intermediate=False,
                                      pv_mask=pv_mask,
                                      axs=axs[i:i+1], legend=labels[0],
                                      plot_hline=False, 
                                      print_args=False, set_title=False,
                                      color=colors[0], double_ind=None,
                                      set_lims=True, list_run_ind=True)
            dc.plot_recon_gen_summary(ri_chair, f_pattern, log_x=False, 
                                      collapse_plots=False, folder=folder,
                                      intermediate=False,
                                      pv_mask=pv_mask,
                                      axs=axs[i+1:i+2], legend=labels[1],
                                      plot_hline=False, 
                                      print_args=False, set_title=False,
                                      color=colors[1], double_ind=None,
                                      set_lims=True, list_run_ind=True)
            gpl.add_vlines(3, axs[i, 0])
            gpl.add_vlines(3, axs[i, 1])
            gpl.add_vlines(3, axs[i + 1, 0])
            gpl.add_vlines(3, axs[i + 1, 1])

        # axs[0, 0].set_ylabel('')
        # axs[0, 1].set_ylabel('')
        # axs[2, 0].set_ylabel('')
        # axs[2, 1].set_ylabel('')

class SIFigureZeroshot(DisentangledFigure):

    def __init__(self, fig_key='figure_img_zeroshot', colors=colors, **kwargs):
        fsize = (6, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.panel_keys = ('zero_gen_flips', 'schem_grids_gen',
                           'schem_grids_extrap')
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_shape_dg(self, retrain=False):
        try:
            assert not retrain
            shape_dg = self.shape_dg
        except:
            twod_file = self.params.get('shapes_path')
            img_size = tuple(self.params.getlist('img_size', typefunc=int))
            shape_dg = dg.TwoDShapeGenerator(twod_file, img_size=img_size,
                                             max_load=10000,
                                             convert_color=False)
            self.shape_dg = shape_dg 
        return shape_dg

    def make_chair_dg(self, retrain=False):
        try:
            assert not retrain
            chair_dg = self.chair_dg
        except:
            chair_file = self.params.get('chairs_path')
            img_size = tuple(self.params.getlist('img_size', typefunc=int))
            n_chairs = self.params.getint('n_chairs')
            filter_edges = self.params.getfloat('filter_edges')
            chair_dg = dg.ChairGenerator(chair_file, img_size=img_size,
                                         max_load=np.inf,
                                         include_position=True,
                                         max_move=.6, filter_edges=filter_edges,
                                         n_unique_chairs=n_chairs)
            self.chair_dg = chair_dg 
        return chair_dg

    def make_preproc_model(self):
        try:
            preproc_model = self.preproc_model
        except AttributeError:
            preproc_path = self.params.get('preproc_path')
            img_size = tuple(self.params.getlist('img_size', typefunc=int))
            learned_lvs = self.params.getlist('learned_lvs', typefunc=bool)
            preproc_model = dd.PretrainedModel(img_size, preproc_path)
            preproc_model.learn_lvs = np.array(learned_lvs)
            self.preproc_model = preproc_model
        return preproc_model
        
    def make_gss(self):
        gss = {}

        spacing = 1
        schem_grids1 = pu.make_mxn_gridspec(self.gs, 4, 4, 0, 45,
                                            0, 23, spacing, spacing)
        schem_axs1 = self.get_axs(schem_grids1)
        schem_grids2 = pu.make_mxn_gridspec(self.gs, 4, 4, 0, 45,
                                            27, 50, spacing, spacing)
        schem_axs2 = self.get_axs(schem_grids2)
        gss[self.panel_keys[1]] = (schem_axs1, schem_axs2)

        
        schem_grids1 = pu.make_mxn_gridspec(self.gs, 4, 4, 55, 100,
                                            0, 23, spacing, spacing)
        schem_axs1 = self.get_axs(schem_grids1)
        schem_grids2 = pu.make_mxn_gridspec(self.gs, 4, 4, 55, 100,
                                            27, 50, spacing, spacing)
        schem_axs2 = self.get_axs(schem_grids2)
        gss[self.panel_keys[2]] = (schem_axs1, schem_axs2)

        
        
        res_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 100,
                                         65, 100, 15, 12)
        res_axs = self.get_axs(res_grids)
        gss[self.panel_keys[0]] = res_axs

        self.gss = gss

    def _mult_schem_grids(self, axs1, axs2, g_col=None, r_col=None):
        if g_col is None:
            g_col = axs2
        if r_col is None:
            r_col = axs2
        chair_dg = self.make_chair_dg()
        n_axs = np.product(axs1.shape)
        _, imgs = chair_dg.sample_reps(n_axs)

        n_rots, n_pos = axs1.shape
        rot_range = self.params.getfloat('rot_range')
        pos_range = self.params.getfloat('pos_range')
        tr_chair_ind = self.params.getint('tr_chair')
        te_chair_ind = self.params.getint('te_chair')

        rots = np.linspace(-rot_range, rot_range, n_rots)
        poss = np.linspace(-pos_range, pos_range, n_pos)

        pos_mask = poss < 0
        for k, ind in enumerate(u.make_array_ind_iterator(axs1.shape)):
            rot_i, pos_i = ind
            tr_img = chair_dg.get_representation([tr_chair_ind, rots[rot_i],
                                                  0, poss[pos_i], 0])
            te_img = chair_dg.get_representation([te_chair_ind, rots[rot_i],
                                                  0, poss[pos_i], 0])
            axs1[ind].imshow(tr_img[0])
            axs2[ind].imshow(te_img[0])
            axs1[ind].set_xticks([])
            axs1[ind].set_yticks([])
            axs2[ind].set_xticks([])
            axs2[ind].set_yticks([])
            if pos_mask[pos_i]:
                gpl.set_ax_color(g_col[ind], 'g')
            else:
                gpl.set_ax_color(r_col[ind], 'r')

    def panel_schem_grids_gen(self):
        key = self.panel_keys[1]
        axs1, axs2 = self.gss[key]
        self._mult_schem_grids(axs1, axs2)

    def panel_schem_grids_extrap(self):
        key = self.panel_keys[2]
        axs1, axs2 = self.gss[key]
        self._mult_schem_grids(axs1, axs2, g_col=axs1)

    def panel_zero_gen_flips(self):
        key = self.panel_keys[0]
        axs = self.gss[key]

        chair_nom_ind = self.params.getlist('chair_nom_ind')
        chair_gen_ind = self.params.getlist('chair_gen_ind')
        chair_extrap_ind = self.params.getlist('chair_extrap_ind')

        twod_nom_ind = self.params.getlist('twod_nom_ind')
        twod_gen_ind = self.params.getlist('twod_gen_ind')
        twod_extrap_ind = self.params.getlist('twod_extrap_ind')

        run_inds = ((twod_gen_ind, chair_gen_ind),
                    (twod_extrap_ind, chair_extrap_ind))
        
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        shape_color = self.params.getcolor('shape_color')
        chair_color = self.params.getcolor('chair_color')

        labels = ('shapes', 'chairs')
        colors = (shape_color, chair_color)

        pv_mask = np.array([True])
        for i, (ri_shape, ri_chair) in enumerate(run_inds):
            dc.plot_recon_gen_summary(ri_shape, f_pattern, log_x=False, 
                                      collapse_plots=False, folder=folder,
                                      intermediate=False,
                                      pv_mask=pv_mask,
                                      axs=axs[i:i+1], legend=labels[0],
                                      plot_hline=False, 
                                      print_args=False, set_title=False,
                                      color=colors[0], double_ind=None,
                                      set_lims=True, list_run_ind=True)
            dc.plot_recon_gen_summary(ri_chair, f_pattern, log_x=False, 
                                      collapse_plots=False, folder=folder,
                                      intermediate=False,
                                      pv_mask=pv_mask,
                                      axs=axs[i:i+1], legend=labels[1],
                                      plot_hline=False, 
                                      print_args=False, set_title=False,
                                      color=colors[1], double_ind=None,
                                      set_lims=True, list_run_ind=True)
            gpl.add_vlines(3, axs[i, 0])
            gpl.add_vlines(3, axs[i, 1])


        
class SIFigureGen(DisentangledFigure):

    def __init__(self, fig_key='si-fig-gen', colors=colors, **kwargs):
        fsize = (5.5, 4.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.panel_keys = ('img_egs', 'rep_geometry', 'traversal_comparison',
                           'preproc_img')
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_shape_dg(self, retrain=False):
        try:
            assert not retrain
            shape_dg = self.shape_dg
        except:
            twod_file = self.params.get('shapes_path')
            img_size = self.params.getlist('img_size', typefunc=int)
            shape_dg = dg.TwoDShapeGenerator(twod_file, img_size=img_size,
                                             max_load=np.inf,
                                             convert_color=False)
            self.shape_dg = shape_dg 
        return shape_dg
        
    def make_gss(self):
        gss = {}
        
        img_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 20,
                                         0, 40, 3, 1)        
        gss[self.panel_keys[0]] = self.get_axs(img_grids)

        rep_geom_fd = self.gs[20:60, :18]
        rep_geom_bvae = self.gs[20:60, 22:40]
        rep_geom_class_perf = self.gs[65:80, :15]
        rep_geom_regr_perf = self.gs[65:80, 25:40]
        axs_3d = np.zeros(4, dtype=bool)
        axs_3d[0:2] = self.params.getboolean('vis_3d')
        gss[self.panel_keys[1]] = self.get_axs((rep_geom_fd, rep_geom_bvae,
                                                rep_geom_class_perf,
                                                rep_geom_regr_perf),
                                               plot_3ds=axs_3d)

        recon_grids = pu.make_mxn_gridspec(self.gs, 5, 6, 0, 80,
                                           45, 100, 3, 1)        
        gss[self.panel_keys[2]] = self.get_axs(recon_grids)

        preproc_grids = pu.make_mxn_gridspec(self.gs, 1, 2, 80, 100,
                                             50, 100, 3, 14)        
        gss[self.panel_keys[3]] = self.get_axs(preproc_grids)

        self.gss = gss

    def panel_img_egs(self):
        key = self.panel_keys[0]
        axs = self.gss[key]
        shape_dg = self.make_shape_dg()

        cm = self.params.get('img_colormap')
        out = shape_dg.sample_reps(sample_size=np.product(axs.shape))
        _, sample_imgs = out
        for i, ind in enumerate(u.make_array_ind_iterator(axs.shape)):
            axs[ind].imshow(sample_imgs[i], cmap=cm)
            axs[ind].set_xticks([])
            axs[ind].set_yticks([])

    def _get_eg_models(self, reload_=False):
        try:
            assert not reload_
            m_fd, m_bvae = self._eg_models
        except:
            path_fd = self.params.get('fd_eg_path')
            path_bvae = self.params.get('bvae_eg_path')
            m_fd = dd.FlexibleDisentanglerAEConv.load(path_fd)
            m_bvae = dd.BetaVAEConv.load(path_bvae)
            self._eg_models = (m_fd, m_bvae)
        return m_fd, m_bvae
            
    def panel_rep_geometry(self):
        key = self.panel_keys[1]
        rep_fd_ax, rep_bvae_ax, class_ax, regr_ax = self.gss[key]
        shape_dg = self.make_shape_dg()

        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')

        m_fd, m_bvae = self._get_eg_models()
        if not key in self.data.keys():
            self.data[key] = {}
            fd_red_func, bvae_red_func = None, None
            
            c_reps = self.params.getint('dg_classifier_reps')
            ident_model = dd.IdentityModel(flatten=True)
            repl_mean = (2,)
            res_ident = characterize_generalization(shape_dg, ident_model,
                                                    c_reps, norm=False,
                                                    repl_mean=repl_mean)
            res_fd = characterize_generalization(shape_dg, m_fd,
                                                 c_reps, norm=False,
                                                 repl_mean=repl_mean)
            res_bvae = characterize_generalization(shape_dg, m_bvae,
                                                   c_reps, norm=False,
                                                   repl_mean=repl_mean)
            self.data[key]['gen'] = (res_ident, res_fd, res_bvae)

        if 'dr' in self.data[key].keys():
            fd_red_func, bvae_red_func = self.data[key]['dr']

        vis_3d = self.params.getboolean('vis_3d')
        out_f = dc.plot_diagnostics(shape_dg, m_fd, rs, n_arcs, n_dim_red=1000,
                                    ax=rep_fd_ax, set_inds=(3, 4),
                                    scale_mag=20,
                                    dim_red_func=fd_red_func, ret_dim_red=True,
                                    plot_3d=vis_3d)
        out_b = dc.plot_diagnostics(shape_dg, m_bvae, rs, n_arcs, n_dim_red=1000,
                                    ax=rep_bvae_ax, set_inds=(3, 4),
                                    dim_red_func=bvae_red_func, scale_mag=.2,
                                    ret_dim_red=True, plot_3d=vis_3d, view_init=(60, 20))
        if 'dr' not in self.data[key].keys():
            self.data[key]['dr'] = (out_f[1], out_b[1])
        res_ident, res_fd, res_bvae = self.data[key]['gen']
        dg_col = self.params.getcolor('dg_color')
        bvae_col = self.params.getcolor('bvae_color')
        fd_col = self.params.getcolor('partition_color')
        colors = (dg_col, fd_col, bvae_col)

        labels = ('input', 'multi-tasking model', r'$\beta$VAE')
        plot_multi_bgp((res_ident[0], res_fd[0], res_bvae[0]),
                       (res_ident[1], res_fd[1], res_bvae[1]),
                       class_ax, regr_ax, colors=colors,
                       legend_labels=labels)

    def _get_img_traversal(self, dg, dim, n):
        cent = dg.get_center()
        unique_inds = np.unique(dg.data_table[dg.img_params[dim]])
        cent_ind = int(np.floor(len(unique_inds)/2))
        x = np.zeros((n, len(cent)))
        off_ind = int(np.floor(n/2))
        x[:, dim] = unique_inds[cent_ind - off_ind:cent_ind + off_ind]
        imgs = dg.get_representation(x)
        return imgs
        
    def panel_traversal_comparison(self):
        key = self.panel_keys[2]
        axs = self.gss[key]
        shape_dg = self.make_shape_dg()
        m_fd, m_bvae = self._get_eg_models()

        traverse_dim = self.params.getint('traverse_dim')
        learn_dim = self.params.getint('learn_dim')
        n_pts = self.params.getint('training_pts')
        n_perts = axs.shape[1]
        
        fd_perturb = self.params.getfloat('fd_perturb')
        bvae_perturb = self.params.getfloat('bvae_perturb')
        eps_d = self.params.getfloat('eps_d')
        cm = self.params.get('img_colormap')
        
        out = dc.plot_traversal_plot(shape_dg, m_fd, full_perturb=fd_perturb,
                                     trav_dim=traverse_dim, n_pts=n_pts,
                                     eps_d=eps_d, learn_dim=learn_dim,
                                     n_dense_pts=n_pts, n_perts=n_perts)
        recs, _, dl, dr, lr = out

        di = self._get_img_traversal(shape_dg, traverse_dim, len(axs[0]))
        dc.plot_img_series(di, title='', axs=axs[0], cmap=cm)
        dc.plot_img_series(dr, title='', axs=axs[1], cmap=cm)
        dc.plot_img_series(recs, title='', axs=axs[2], cmap=cm)

        out = dc.plot_traversal_plot(shape_dg, m_bvae,
                                     full_perturb=bvae_perturb,
                                     trav_dim=traverse_dim, n_pts=n_pts,
                                     eps_d=eps_d, n_dense_pts=n_pts,
                                     learn_dim=learn_dim, n_perts=n_perts)
        recs, di, dl, dr, lr = out
        dc.plot_img_series(dr, title='', axs=axs[3], cmap=cm)
        dc.plot_img_series(recs, title='', axs=axs[4], cmap=cm)
        
def get_tasks_learned(task_hists, training_ind=-1, thresh=.8):
    frac_learned = np.zeros_like(task_hists)
    n_tasks = np.zeros(task_hists.shape[0])
    for ind in u.make_array_ind_iterator(task_hists.shape):
        _, th = task_hists[ind]
        th_arr = np.array(th)
        n_tasks[ind[0]] = th_arr.shape[1]
        frac_learned[ind] = np.sum(th_arr[training_ind] > thresh)/th_arr.shape[1]
    return n_tasks, frac_learned
        
class FigureRL(DisentangledFigure):

    def __init__(self, fig_key='figure_rl', colors=colors, **kwargs):
        fsize = (4, 4)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('panel_history', 'panel_gen_performance')
        super().__init__(fsize, params, colors=colors, **kwargs)

    def make_gss(self):
        gss = {}

        schem_grid = self.gs[:50, 30:]
        self.get_axs((schem_grid,))

        hist_grid = pu.make_mxn_gridspec(self.gs, 2, 1,
                                         30, 100, 0, 20,
                                         13, 3)
        hist_axs = self.get_axs(hist_grid)
        gss[self.panel_keys[0]] = hist_axs
        
        gen_grid = pu.make_mxn_gridspec(self.gs, 1, 2,
                                       60, 100, 38, 100,
                                        3, 16)
        gen_axs = self.get_axs(gen_grid)
        gss[self.panel_keys[1]] = gen_axs
        self.gss = gss

    def _get_rl_hists(self):
        run_inds = self.params.getlist('run_ind_comb')
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        for i, ri in enumerate(run_inds):
            out = da.load_full_run(folder, ri, merge_axis=0, file_template=f_pattern,
                                   analysis_only=False, add_hist=True)
            hist = out[0][3]
            parts = out[0][0]
            if i == 0:
                hist_all = hist
            else:
                hist_all = np.concatenate((hist_all, hist), axis=-1)
        hist_all = np.swapaxes(hist_all, 0, 1)
        return parts, hist_all
        
    def _get_rl_run(self, ind=0):
        
        ri = self.params.getlist('run_ind_comb')[ind]
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        out = da.load_full_run(folder, ri, merge_axis=1, file_template=f_pattern,
                               analysis_only=False, add_hist=True)
        order = ('inds', 'dg', 'model', 'history', 'p', 'c', 'ld', 'sc',
                 'other')
        out_dict = dict(zip(order, out[0]))
        return out_dict, out[1]
    
    def panel_history(self):
        key = self.panel_keys[0]
        axs = self.gss[key]

        rl_run, rl_info = self._get_rl_run()
        hist_all = rl_run['history']

        tasks_ind = self.params.getint('hist_task_ind')
        rep_ind = self.params.getint('hist_rep_ind')
        task_color = self.params.getcolor('rl_task_color')
        epoch_cut = self.params.getint('epoch_cutoff')
        init_collect_ind = self.params.getint('initial_collect_ind')

        parts, hist_all = self._get_rl_hists()
        
        n_tasks, task_fractions = get_tasks_learned(np.squeeze(hist_all))
        ax, frac_ax = axs[:, 0]
        use_tf = np.array(task_fractions[init_collect_ind], dtype=float).T
        gpl.plot_trace_werr(parts, use_tf,
                            color=task_color, ax=frac_ax)
        # frac_ax.plot(parts, np.mean(task_fractions[init_collect_ind, axis=1),
        #              color=task_color)
        frac_ax.set_ylabel('fraction of\ntasks learned')
        frac_ax.set_xlabel('tasks')
        gpl.clean_plot(frac_ax, 0)
        
        hist_full, hist_task = hist_all[0, tasks_ind, 0, rep_ind]
        task_perf = np.array(hist_task)
        epochs = np.arange(1, task_perf.shape[0] + 1)

        mask = epochs < epoch_cut
        ax.plot(epochs[mask], task_perf[mask], color=task_color)
        ax.set_xlabel('training epoch')
        ax.set_ylabel('average\ntask reward')
        gpl.clean_plot(ax, 0)

    def panel_gen_performance(self):
        key = self.panel_keys[1]
        axs = self.gss[key]

        ri = self.params.getlist('run_ind_comb')
        f_pattern = self.params.get('f_pattern')
        folder = self.params.get('mp_simulations_path')
        rl_color = self.params.getcolor('rl_gen_color')
        label = 'RL agent'
        pv_mask = np.array([True, False, False])
        print_args = True

        dc.plot_recon_gen_summary(ri, f_pattern, log_x=False, 
                                  collapse_plots=False, folder=folder,
                                  intermediate=False,
                                  pv_mask=pv_mask,
                                  axs=axs, legend=label,
                                  plot_hline=False, 
                                  print_args=print_args, set_title=False,
                                  color=rl_color, double_ind=0, 
                                  set_lims=True, list_run_ind=True,
                                  rl_model=True)
        gpl.add_vlines(5, axs[0, 0])
        gpl.add_vlines(5, axs[0, 1])
        
class SIFigureRepWidth(DisentangledFigure):

    def __init__(self, fig_key='sifigure_rep_width', colors=colors, **kwargs):
        fsize = (3, 1)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('panel_uniform',)
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}

        unif_grid = pu.make_mxn_gridspec(self.gs, 1, 2,
                                         0, 100, 0, 100,
                                         20, 20)
        unif_axs = self.get_axs(unif_grid, squeeze=False)
        gss[self.panel_keys[0]] = unif_axs
        self.gss = gss

    def panel_uniform(self):
        key = self.panel_keys[0]
        axs = self.gss[key]

        norm_ind = self.params.get('normative_run')
        unif_ind = self.params.get('unif_width_run')
        norm_col = self.params.getcolor('partition_color')
        unif_col = self.params.getcolor('unif_width_color')
        labels = ('standard layers',
                  'uniform width layers')
        pv_mask = np.array([1])

        run_inds = (norm_ind, unif_ind)
        colors = (norm_col, unif_col)
        print(norm_col, unif_col)

        self._abstraction_panel(run_inds, axs, labels=labels,
                                pv_mask=pv_mask,
                                colors=colors)

class SIFigureInpWidth(DisentangledFigure):

    def __init__(self, fig_key='sifigure_inp_width', colors=colors, **kwargs):
        fsize = (3.5, 2.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = (
            'panel_inps',
            'panel_inp_info',
        )
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}

        inps_grid = pu.make_mxn_gridspec(self.gs, 2, 2,
                                         0, 100, 0, 100,
                                         20, 20)
        inps_axs = self.get_axs(inps_grid, squeeze=False)
        gss[self.panel_keys[0]] = inps_axs[1:]
        gss[self.panel_keys[1]] = inps_axs[0]
        self.gss = gss

    def panel_inp_info(self):
        key = self.panel_keys[1]
        dim_ax, sparse_ax = self.gss[key]
        fdg_list = []

        if self.data.get(key) is None:
            inp_wids = self.params.getlist('input_widths',
                                           typefunc=int)
            for i, iw in enumerate(inp_wids):
                fdg_i = self.make_fdg(dg_dim=iw, retrain=True)
                fdg_list.append(fdg_i)
                
            self.data[key] = (inp_wids, fdg_list)
        inp_wids, fdg_list = self.data[key]
        dims = np.zeros(len(fdg_list))
        sparses = np.zeros_like(dims)
        for i, fdg in enumerate(fdg_list):
            dims[i] = fdg.representation_dimensionality(
                participation_ratio=True)
            _, reps = fdg.sample_reps()
            sparses[i] = np.nanmean(dc.quantify_sparseness(reps))

        inps_cmap = self.params.get('inp_color_cmap')
        inps_cmap = plt.get_cmap(inps_cmap)
        color = inps_cmap(.5)
        gpl.plot_trace_werr(inp_wids, dims, points=True, ax=dim_ax,
                            ms=2, lw=.2, color=color)
        gpl.plot_trace_werr(inp_wids, sparses, points=True, ax=sparse_ax,
                            ms=2, lw=.2, color=color)
        dim_ax.set_xlabel('input width')
        dim_ax.set_ylabel('dimensionality')
        sparse_ax.set_ylabel('sparseness')

    def panel_inps(self):
        key = self.panel_keys[0]
        axs = self.gss[key]

        norm_ind = self.params.get('normative_run')
        inps = self.params.getlist('inp_width_runs')
        norm_col = self.params.getcolor('partition_color')
        
        inps_cmap = self.params.get('inp_color_cmap')
        inps_cmap = plt.get_cmap(inps_cmap)
        labels = ('standard input, 500',
                  '750', '1000', '1250')
        pv_mask = np.array([1])

        run_inds = (norm_ind,) + tuple(inps)
        colors = (norm_col,) + tuple(inps_cmap([.6, .4, .2]))

        self._abstraction_panel(run_inds, axs, labels=labels,
                                pv_mask=pv_mask,
                                colors=colors)

class SIFigureIntermediate(DisentangledFigure):

    def __init__(self, fig_key='sifigure_intermediate', colors=colors, **kwargs):
        fsize = (4, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('panel_nominal', 'panel_uniform',)
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}

        abs_grid = pu.make_mxn_gridspec(self.gs, 2, 2,
                                        0, 100, 0, 100,
                                        20, 15)
        abs_axs = self.get_axs(abs_grid, squeeze=False)
        
        gss[self.panel_keys[0]] = abs_axs[:1]
        gss[self.panel_keys[1]] = abs_axs[1:2]
        self.gss = gss

    def panel_nominal(self):
        key = self.panel_keys[0]
        axs = self.gss[key]

        ri = self.params.get('nom_intermediate')
        cmap = gpl.make_linear_cmap(self.params.getcolor('partition_color'))        
        inter_colors = cmap(np.linspace(.4, 1, 3))
        inter_labels = ('hidden 1', 'hidden 2', 'representation layer')                                    
        pv_mask = np.array([1])

        self._abstraction_panel((ri,), axs, inter_labels=inter_labels,
                                pv_mask=pv_mask, inter_colors=inter_colors,
                                intermediate=True)

    def panel_uniform(self):
        key = self.panel_keys[1]
        axs = self.gss[key]

        ri = self.params.get('unif_intermediate')
        cmap = gpl.make_linear_cmap(self.params.getcolor('unif_width_color'))
        inter_colors = cmap(np.linspace(.4, 1, 3))
        inter_labels = ('hidden 1', 'hidden 2', 'representation layer')

        pv_mask = np.array([1])

        self._abstraction_panel((ri,), axs, inter_labels=inter_labels,
                                pv_mask=pv_mask, inter_colors=inter_colors,
                                intermediate=True)

class SIFigureReg(DisentangledFigure):

    def __init__(self, fig_key='sifigure_reg', colors=colors, **kwargs):
        fsize = (5, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('panel_l1', 'panel_l2', 'panel_sparseness_l1',
                           'panel_sparseness_l2')
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}

        reg_grid = pu.make_mxn_gridspec(self.gs, 2, 3,
                                        0, 100, 0, 100,
                                        20, 15)
        abs_axs = self.get_axs(reg_grid[:, 1:], squeeze=False)
        sparse_axs = self.get_axs(reg_grid[:, 0], squeeze=False,
                                  sharex=True, sharey=True)
        
        gss[self.panel_keys[0]] = abs_axs[:1]
        gss[self.panel_keys[1]] = abs_axs[1:2]
        gss[self.panel_keys[2]] = sparse_axs[0, 0]
        gss[self.panel_keys[3]] = sparse_axs[0, 1]
        self.gss = gss

    def panel_sparseness_l1(self):
        key = self.panel_keys[2]
        ax = self.gss[key]

        if self.data.get(key) is None:
            f_pattern = self.params.get('f_pattern')
            folder = self.params.get('mp_simulations_path')

            use_runs = self.params.getint('use_runs')
            norm_ind = self.params.get('normative_run')
            l1s = self.params.getlist('l1_reg_runs')[::-1]
            run_inds = (norm_ind,) + tuple(l1s)

            ind = (1, 12)
            l1_weights = []
            avg_sparse = []
            for i, ri in enumerate(run_inds):
                out = da.load_full_run(folder, ri, file_template=f_pattern,
                                       analysis_only=False, skip_gd=False)
                ls_l1, inps_l1, reps_l1 = out[0][-2]
                l1 = out[1]['args'][0]['l1_weight']
                if l1 is None:
                    l1 = 0 
                l1_weights.append(l1)
            
                s = dc.quantify_sparseness(reps_l1[ind], axis=1)
                avg_sparse.append(np.mean(s, axis=1)[:use_runs])
            self.data[key] = (np.array(l1_weights),
                              np.stack(avg_sparse, axis=0))
        l1_weights, avg_sparse = self.data[key]

        cmap = plt.get_cmap(self.params.get('l1_color_cmap'))
        norm_col = self.params.getcolor('partition_color')
        ms = self.params.getfloat('markersize')
        col_pts = np.linspace(.3, .9, len(l1_weights))
        colors = (norm_col,) + tuple(cmap(col_pts))
        gpl.plot_colored_pts(l1_weights, avg_sparse.T, ax=ax,
                             colors=colors, markersize=ms)
        ax.set_xlabel('L2 weight')
        ax.set_ylabel('sparseness')
        

    def panel_sparseness_l2(self):
        key = self.panel_keys[3]
        ax = self.gss[key]

        if self.data.get(key) is None:
            f_pattern = self.params.get('f_pattern')
            folder = self.params.get('mp_simulations_path')
            use_runs = self.params.getint('use_runs')

            norm_ind = self.params.get('normative_run')
            l2s = self.params.getlist('l2_reg_runs')[::-1]
            run_inds = (norm_ind,) + tuple(l2s)

            ind = (1, 12)
            l2_weights = []
            avg_sparse = []
            for i, ri in enumerate(run_inds):
                out = da.load_full_run(folder, ri, file_template=f_pattern,
                                       analysis_only=False, skip_gd=False)
                ls_l2, inps_l2, reps_l2 = out[0][-2]
                l2 = out[1]['args'][0]['l2_weight']
                if l2 is None:
                    l2 = 0 
                l2_weights.append(l2)
            
                s = dc.quantify_sparseness(reps_l2[ind], axis=1)
                avg_sparse.append(np.mean(s, axis=1)[:use_runs])
            self.data[key] = (np.array(l2_weights),
                              np.stack(avg_sparse, axis=0))
        l2_weights, avg_sparse = self.data[key]
        
        norm_col = self.params.getcolor('partition_color')
        cmap = plt.get_cmap(self.params.get('l2_color_cmap'))
        ms = self.params.getfloat('markersize')
        col_pts = np.linspace(.3, .9, len(l2_weights))
        colors = (norm_col,) + tuple(cmap(col_pts))
        gpl.plot_colored_pts(l2_weights, avg_sparse.T, ax=ax,
                             colors=colors, markersize=ms)
        # gpl.plot_trace_werr(l2_weights, avg_sparse.T, ax=ax,
        #                     color=cmap(.5))
        ax.set_xlabel('L2 weight')
        ax.set_ylabel('sparseness')

        
    def panel_l1(self):
        key = self.panel_keys[0]
        axs = self.gss[key]

        norm_ind = self.params.get('normative_run')
        norm_col = self.params.getcolor('partition_color')

        l1s = self.params.getlist('l1_reg_runs')[::-1]
        l1_cmap = self.params.get('l1_color_cmap')
        l1_cmap = plt.get_cmap(l1_cmap)
        labels = ('no regularization',) + ('',)*len(l1s)
        pv_mask = np.array([1])

        run_inds = (norm_ind,) + tuple(l1s)
        col_pts = np.linspace(.3, .9, len(l1s))
        colors = (norm_col,) + tuple(l1_cmap(col_pts))

        self._abstraction_panel(run_inds, axs, labels=labels,
                                pv_mask=pv_mask,
                                colors=colors,
                                label_field='l1_weight')

    def panel_l2(self):
        key = self.panel_keys[1]
        axs = self.gss[key]

        norm_ind = self.params.get('normative_run')
        norm_col = self.params.getcolor('partition_color')

        l2s = self.params.getlist('l2_reg_runs')[::-1]
        l2_cmap = self.params.get('l2_color_cmap')
        l2_cmap = plt.get_cmap(l2_cmap)
        labels = ('no regularization',) + ('',)*len(l2s)
        pv_mask = np.array([1])

        run_inds = (norm_ind,) + tuple(l2s)
        col_pts = np.linspace(.3, .9, len(l2s))
        colors = (norm_col,) + tuple(l2_cmap(col_pts))

        self._abstraction_panel(run_inds, axs, labels=labels,
                                pv_mask=pv_mask,
                                colors=colors,
                                label_field='l2_weight')

        
class SIFigureMultiverse(DisentangledFigure):
    
    def __init__(self, fig_key='sifigure_multi', colors=colors, **kwargs):
        fsize = (5.5, 3)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('panel_multiverse',)
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}
        m1_grid = pu.make_mxn_gridspec(self.gs, 1, 8,
                                       0, 48, 0, 100,
                                       20, 3)
        m1_axs = self.get_axs(m1_grid, sharey=True)
        m2_grid = pu.make_mxn_gridspec(self.gs, 1, 8,
                                       52, 100, 0, 100,
                                       20, 3)
        m2_axs = self.get_axs(m2_grid, sharey=True)
        gss[self.panel_keys[0]] = (m1_axs[0], m2_axs[0])
        self.gss = gss

    def panel_multiverse(self):
        key = self.panel_keys[0]
        axs = self.gss[key]

        fd_manifest_path = self.params.get('fd_manifest_path')
        fd_pattern = self.params.get('fd_pattern')
        
        bv_manifest_path = self.params.get('bv_manifest_path')
        bv_pattern = self.params.get('bv_pattern')

        results_folder = self.params.get('results_folder')
        fd_color = self.params.getcolor('partition_color')
        bv_color = self.params.getcolor('bvae_color')
        colors = (fd_color, bv_color)

        if self.data.get(key) is None:
            fd_manifest = {'fd':fd_manifest_path}
            mv_fd = dmo.load_multiverse(results_folder, fd_manifest,
                                        run_pattern=fd_pattern)
            bv_manifest = {'bv':bv_manifest_path}
            mv_bv = dmo.load_multiverse(results_folder, bv_manifest,
                                        run_pattern=bv_pattern)  

            out_fd1 = dmo.model_explanation(mv_fd, 'class_gen')
            out_fd2 = dmo.model_explanation(mv_fd, 'regr_gen')
            out_bv1 = dmo.model_explanation(mv_bv, 'class_gen')
            out_bv2 = dmo.model_explanation(mv_bv, 'regr_gen')
            self.data[key] = (mv_fd, mv_bv, (out_fd1, out_fd2), (out_bv1,
                                                                 out_bv2))
        mv_fd, mv_bv, out_fd, out_bv = self.data[key]

        title_dict = {'layer_spec':'depth', 'train_eg':'training data',
                      'use_tanh':'act function',
                      'input_dims':'latent variables',
                      'no_autoencoder':'autoencoder',
                      'betas':r'tasks / $\beta$',
                      'source_distr':'latent variable\ndistribution',
                      'partitions':r'tasks / $\beta$',
                      'latent_dims':'rep width'}
        for i, (r_fd_i, s_fd_i, lh_fd_i, l_fd_i, lv_fd_i) in enumerate(out_fd):
            r_bv_i, s_bv_i, lh_bv_i, l_bv_i, lv_bv_i = out_bv[i]
            axs_i = axs[i]
            axd_i = {'partitions':axs_i[0], 'betas':axs_i[0],
                     'layer_spec':axs_i[1], 'train_eg':axs_i[2],
                     'use_tanh':axs_i[3], 'input_dims':axs_i[4],
                     'source_distr':axs_i[5], 'latent_dims':axs_i[6],
                     'no_autoencoder':axs_i[7]}
            if i == len(out_fd) - 1:
                labels = True
            else:
                labels = False
            model_names = ('multi-tasking model', r'$\beta$VAE')
            dmo.plot_multiple_model_coefs((l_fd_i, l_bv_i), (r_fd_i, r_bv_i),
                                          (lh_fd_i, lh_bv_i), ax_dict=axd_i,
                                          title_dict=title_dict, colors=colors,
                                          labels=labels, model_names=model_names,
                                          v_dicts=(lv_fd_i, lv_bv_i))
            for i, ax in enumerate(axs_i):
                gpl.clean_plot(ax, i)
        axd_i['no_autoencoder'].set_xticks([0, 1])
        axd_i['no_autoencoder'].set_xticklabels(['with', 'without'],
                                                rotation='vertical')
        axd_i['use_tanh'].set_xticks([0, 1])
        axd_i['use_tanh'].set_xticklabels(['ReLU', 'tanh'],
                                          rotation='vertical')
        axd_i['layer_spec'].set_xticklabels([3, 4, 5])
        axd_i['source_distr'].set_xticklabels(['normal', 'uniform'],
                                              rotation='vertical')
        axd_i['train_eg'].set_xticks([10000, 100000])
        axd_i['train_eg'].set_xticklabels([r'$10^{4}$', r'$10^{5}$'])
        axd_i['input_dims'].set_xticks([2, 5, 8])
        axd_i['partitions'].legend(frameon=False)
        axs[0][0].set_ylabel('classifier generalization\ninfluence')
        axs[1][0].set_ylabel('regression generalization\ninfluence')
        
class SIFigureDim(DisentangledFigure):

    def __init__(self, fig_key='sifigure_dim', colors=colors, **kwargs):
        fsize = (4, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('dim_dependence',)
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}
        dims = self.params.getlist('dims', typefunc=int)
        dims_grid = pu.make_mxn_gridspec(self.gs, len(dims), 2,
                                         0, 100, 0, 100,
                                         5, 20)
        gss[self.panel_keys[0]] = self.get_axs(dims_grid)
        self.gss = gss

    def panel_dim_dependence(self):
        key = self.panel_keys[0]
        axs = self.gss[key]

        dims = self.params.getlist('dims', typefunc=int)
        fd_inds = self.params.getlist('fd_dims_inds')
        bv_inds = self.params.getlist('bv_dims_inds')

        f_pattern = self.params.get('f_pattern')
        beta_f_pattern = self.params.get('beta_f_pattern')

        folder = self.params.get('mp_simulations_path')
        beta_folder = self.params.get('beta_simulations_path')

        part_color = self.params.getcolor('partition_color')
        bvae_color = self.params.getcolor('bvae_color')
        xlab = r'tasks / $\beta$'

        pv_mask = np.array([False, True, False])

        for i, dim in enumerate(dims):
            fd_ri = fd_inds[i]
            bv_ri = bv_inds[i]
            if i == 0:
                fd_legend = 'partition'
                bv_legend = r'$\beta$VAE'
            else:
                fd_legend = ''
                bv_legend = ''
            dc.plot_recon_gen_summary(fd_ri, f_pattern, log_x=False, 
                                      collapse_plots=False, folder=folder,
                                      axs=axs[i:i+1], legend=fd_legend,
                                      print_args=False, pv_mask=pv_mask,
                                      set_title=False, color=part_color,
                                      xlab=xlab)
            dc.plot_recon_gen_summary(bv_ri, beta_f_pattern, log_x=False, 
                                      collapse_plots=False, folder=beta_folder,
                                      axs=axs[i:i+1], legend=bv_legend,
                                      print_args=False, pv_mask=pv_mask,
                                      set_title=False, color=bvae_color,
                                      xlab=xlab, plot_hline=False)
            axs[i, 0].text(35, .8, r'$D = {}$'.format(dim))
            if i < len(dims) - 1:
                axs[i, 0].set_xlabel('')
                axs[i, 1].set_xlabel('')
                axs[i, 0].set_xticklabels([])
                axs[i, 1].set_xticklabels([])
