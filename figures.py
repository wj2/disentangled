
import numpy as np
import scipy.stats as sts
import functools as ft
import sklearn.decomposition as skd
import sklearn.svm as skc
import scipy.linalg as spla

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
                    labels=('standard', 'gen'), legend_label=''):
    if xs is None:
        xs = [0, 1]
    gpl.violinplot(results.T, xs, ax=ax, color=(color, color),
                   showextrema=False)
    ax.plot(xs, np.mean(results, axis=0), 'o', color=color,
            label=legend_label)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    gpl.clean_plot(ax, 0)
    gpl.clean_plot_bottom(ax, keeplabels=True)
    return ax

def plot_multi_gen(res_list, ax, xs=None, labels=('standard', 'gen'),
                   sep=.2, colors=None, legend_labels=None):
    if xs is None:
        xs = np.array([0, 1])
    if colors is None:
        colors = (None,)*len(res_list)
    if legend_labels is None:
        legend_labels = ('',)*len(res_list)
    start_xs = xs - len(res_list)*sep/4
    n_seps = (len(res_list) - 1)/2
    use_xs = np.linspace(-sep*n_seps, sep*n_seps, len(res_list))
    
    for i, rs in enumerate(res_list):
        plot_single_gen(rs, ax, xs=xs + use_xs[i], color=colors[i],
                        legend_label=legend_labels[i])
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
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

def characterize_generalization(dg, model, c_reps, train_samples=1000,
                                test_samples=500, bootstrap_regr=True,
                                n_boots=1000, norm=True, cut_zero=True,
                                repl_mean=None):
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
            n_iters=1, repl_mean=repl_mean)[0]
        results_class[i, 1] = dc.classifier_generalization(
            dg, model, train_distrib=train_distr,
            test_distrib=test_distr, n_train_samples=train_samples,
            n_test_samples=test_samples, n_iters=1, repl_mean=repl_mean)[0]
        results_regr[i, 0] = dc.find_linear_mapping_single(
            dg, model, half=False, n_samps=train_samples,
            repl_mean=repl_mean)[1]
        results_regr[i, 1] = dc.find_linear_mapping_single(
            dg, model, n_samps=train_samples,
            repl_mean=repl_mean)[1]
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

    def make_fdg(self, retrain=False):
        try:
            assert not retrain
            fdg = self.fdg
        except:
            inp_dim = self.params.getint('inp_dim')
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
                    train_samples=dg_train_egs, batch_size=dg_bs)
            self.fdg = fdg 
        return fdg
    
    def _standard_panel(self, fdg, model, run_inds, f_pattern, folder, axs,
                        labels=None, rep_scale_mag=5, source_scale_mag=.5,
                        x_label=True, y_label=True, colors=None, view_init=None,
                        multi_num=1, **kwargs):
        model = model[0, 0]
        if labels is None:
            labels = ('',)*len(run_inds)
        if len(axs) == 3:
            ax_break = 1
        else:
            ax_break = 2
        manifold_axs = axs[:ax_break]
        res_axs = np.expand_dims(axs[ax_break:], 0)
        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        vis_3d = self.params.getboolean('vis_3d')

        # print(characterize_generalization(fdg, model, 10))
        dc.plot_source_manifold(fdg, model, rs, n_arcs, 
                                source_scale_mag=source_scale_mag,
                                rep_scale_mag=rep_scale_mag,
                                markers=False, axs=manifold_axs,
                                titles=False, plot_model_3d=vis_3d,
                                model_view_init=view_init)
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
                                      collapse_plots=False, folder=folder,
                                      axs=res_axs, legend=labels[i],
                                      print_args=False, set_title=False,
                                      color=colors[i], double_ind=double_inds[i],
                                      **kwargs)
        res_axs[0, 0].set_yticks([.5, 1])
        res_axs[0, 1].set_yticks([0, .5, 1])

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

class Figure2(DisentangledFigure):
    
    def __init__(self, fig_key='figure2', colors=colors, **kwargs):
        fsize = (6, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('order_disorder', 'training_rep', 'rep_summary')
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}

        # ordered_rep_grid = self.gs[:25, :30]
        # class_perf_grid = self.gs[75:, :15]
        # regr_perf_grid = self.gs[75:, 30:45]

        inp_grid = pu.make_mxn_gridspec(self.gs, 1, 2,
                                            50, 68, 10, 50,
                                            5, 10)
        # high_d_grid = pu.make_mxn_gridspec(self.gs, 1, 3,
        #                                    75, 100, 0, 10,
        #                                    5, 2)
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
        plot_3d_axs = np.zeros((n_parts, 2), dtype=bool)
        plot_3d_axs[:, 1] = self.params.getboolean('vis_3d')
        rep_axs = self.get_axs(rep_grids, sharex='vertical',
                               sharey='vertical', plot_3ds=plot_3d_axs)
        gss[self.panel_keys[1]] = train_ax, rep_axs
        
        rep_classifier_grid = self.gs[75:, 60:75]
        rep_regression_grid = self.gs[75:, 85:]
        gss[self.panel_keys[2]] = self.get_axs((rep_classifier_grid,
                                                rep_regression_grid))
        self.gss = gss
    
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
        
        if self.data.get(key) is None:
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

            model_kinds = list(ft.partial(dd.FlexibleDisentanglerAE,
                                          true_inp_dim=fdg.input_dim, 
                                          n_partitions=num_p,
                                          no_autoenc=no_autoencoder) 
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
            self.data[key] = (out, (n_parts, n_epochs))
        fdg, (models, th), (p, _), (_, scrs, _), _ = self.data[key][0]
        n_parts, n_epochs = self.data[key][1]

        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        npart_signifier = self.params.get('npart_signifier')
        mid_i = np.floor(len(n_parts)/2)
        vis_3d = self.params.getboolean('vis_3d')
        view_inits = (None, (50, 30), (40, -20))
        for i, num_p in enumerate(n_parts):
            hist = th[0, i, 0].history['loss']
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

        axs_3d = np.zeros(4, dtype=bool)
        axs_3d[1] = self.params.getboolean('vis_3d')
        axs_left = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 100, 0, 40,
                                        3, 0)
        axs_right = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 100, 54, 100,
                                         5, 15)
        gss[self.panel_keys[0]] = self.get_axs(np.concatenate((axs_left[0],
                                                               axs_right[0])),
                                               plot_3ds=axs_3d)
        gss[self.panel_keys[1]] = self.get_axs(np.concatenate((axs_left[1],
                                                               axs_right[1])),
                                               plot_3ds=axs_3d)
        gss[self.panel_keys[2]] = self.get_axs(np.concatenate((axs_left[2],
                                                               axs_right[2])),
                                               plot_3ds=axs_3d)

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
        self.fdg = self.data.get('fdg')

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
                             view_init=(45, -30), linestyle=grid_style)
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
        

class Figure4(DisentangledFigure):

    def __init__(self, fig_key='figure4', colors=colors, **kwargs):
        fsize = (6, 4)
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
        axs_3ds = np.zeros(4, dtype=bool)
        axs_3ds[1] = self.params.getboolean('vis_3d')
        gss[self.panel_keys[0]] = self.get_axs((rf_schematic_grid,
                                                rf_projection_grid,
                                                rf_dec_grid[0, 0],
                                                rf_dec_grid[0, 1]),
                                               plot_3ds=axs_3ds)

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
            rfdg.plot_rfs(schem_ax, color=rf_eg_color, thin=5)

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
        pv_mask = np.array([False, True, False])

        part_color = self.params.getcolor('partition_color')
        bvae_color = self.params.getcolor('bvae_color')
        xlab = r'tasks / $\beta$'
        
        dc.plot_recon_gen_summary(run_ind_fd, f_pattern, log_x=False, 
                                  collapse_plots=False, folder=folder,
                                  axs=res_axs, legend='multi-tasking model',
                                  print_args=False, pv_mask=pv_mask,
                                  set_title=False, color=part_color,
                                  xlab=xlab)
        dc.plot_recon_gen_summary(run_ind_beta, beta_f_pattern, log_x=False, 
                                  collapse_plots=False, folder=beta_folder,
                                  axs=res_axs, legend=r'$\beta$VAE',
                                  print_args=False, pv_mask=pv_mask,
                                  set_title=False, color=bvae_color,
                                  xlab=xlab)

class Figure5(DisentangledFigure):

    def __init__(self, fig_key='figure5', colors=colors, **kwargs):
        fsize = (5.5, 3.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.panel_keys = ('img_egs', 'rep_geometry', 'traversal_comparison')
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
        
        img_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 30,
                                         0, 40, 3, 1)        
        gss[self.panel_keys[0]] = self.get_axs(img_grids)

        rep_geom_fd = self.gs[30:70, :18]
        rep_geom_bvae = self.gs[30:70, 22:40]
        rep_geom_class_perf = self.gs[75:, :15]
        rep_geom_regr_perf = self.gs[75:, 25:40]
        axs_3d = np.zeros(4, dtype=bool)
        axs_3d[0:2] = self.params.getboolean('vis_3d')
        gss[self.panel_keys[1]] = self.get_axs((rep_geom_fd, rep_geom_bvae,
                                                rep_geom_class_perf,
                                                rep_geom_regr_perf),
                                               plot_3ds=axs_3d)

        recon_grids = pu.make_mxn_gridspec(self.gs, 5, 6, 0, 100,
                                           45, 100, 3, 1)        
        gss[self.panel_keys[2]] = self.get_axs(recon_grids)
        
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

class SIFigureMultiverse(DisentangledFigure):
    
    def __init__(self, fig_key='sifigure_multi', colors=colors, **kwargs):
        fsize = (5.5, 5)
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
