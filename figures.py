
import numpy as np
import scipy.stats as sts
import functools as ft
import sklearn.decomposition as skd

import general.plotting as gpl
import general.plotting_styles as gps
import general.paper_utilities as pu
import general.utility as u
import disentangled.data_generation as dg
import disentangled.disentanglers as dd
import disentangled.characterization as dc
import disentangled.aux as da
import disentangled.theory as dt

config_path = 'disentangled/figures.conf'

colors = np.array([(127,205,187),
                   (65,182,196),
                   (29,145,192),
                   (34,94,168),
                   (37,52,148),
                   (8,29,88)])/256

tuple_int = lambda x: (int(x),)

def plot_single_gen(results, ax, xs=None, color=None,
                    labels=('standard', 'gen')):
    if xs is None:
        xs = [0, 1]
    gpl.violinplot(results.T, xs, ax=ax, color=(color, color),
                   showextrema=False)
    ax.plot(xs, np.mean(results, axis=0), 'o', color=color)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    gpl.clean_plot(ax, 0)
    gpl.clean_plot_bottom(ax, keeplabels=True)
    return ax

def plot_multi_gen(res_list, ax, xs=None, labels=('standard', 'gen'),
                   sep=.2, colors=None):
    if xs is None:
        xs = np.array([0, 1])
    if colors is None:
        colors = (None,)*len(res_list)
    start_xs = xs - len(res_list)*sep/4

    for i, rs in enumerate(res_list):
        plot_single_gen(rs, ax, xs=start_xs + i*sep, color=colors[i])
        print(start_xs + i*sep)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
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
                              epochs=200, n_class=10, **kwargs):
    
    rbf_dg = dg.KernelDataGenerator(dims, None, inp_dim)
    fdae = dd.FlexibleDisentanglerAE(inp_dim, layers, latent, n_partitions=0,
                                     **kwargs)
    y, x = rbf_dg.sample_reps(n_samps)
    fdae.fit(x, y, epochs=epochs, verbose=False)
    class_p, regr_p = characterize_generalization(rbf_dg,
                                                  dd.IdentityModel(),
                                                  n_class)
    class_m, regr_m = characterize_generalization(rbf_dg, fdae, n_class)
    return class_m, regr_m    

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
            
            source_distr = sts.multivariate_normal(np.zeros(inp_dim),
                                                   dg_source_var)
            
            fdg = dg.FunctionalDataGenerator(inp_dim, dg_layers, dg_dim,
                                             noise=dg_noise,
                                             l2_weight=dg_regweight)
            fdg.fit(source_distribution=source_distr, epochs=dg_epochs,
                    train_samples=dg_train_egs)
            self.fdg = fdg 
        return fdg
    
    def _standard_panel(self, fdg, model, run_inds, f_pattern, folder, axs,
                        labels=None, rep_scale_mag=.2, source_scale_mag=.2,
                        x_label=True, y_label=True, colors=None, **kwargs):
        model = model[0, 0]
        if labels is None:
            labels = ('',)*len(run_inds)
        manifold_axs = axs[:2]
        res_axs = np.expand_dims(axs[2:], 0)
        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')        
        dc.plot_source_manifold(fdg, model, rs, n_arcs, 
                                source_scale_mag=source_scale_mag,
                                rep_scale_mag=rep_scale_mag,
                                markers=False, axs=manifold_axs,
                                titles=False)
        if colors is None:
            colors = (None,)*len(run_inds)
        for i, ri in enumerate(run_inds):
            dc.plot_recon_gen_summary(ri, f_pattern, log_x=False, 
                                      collapse_plots=False, folder=folder,
                                      axs=res_axs, legend=labels[i],
                                      print_args=False, set_title=False,
                                      color=colors[i], **kwargs)

class Figure1(DisentangledFigure):
    
    def __init__(self, fig_key='figure1', colors=colors, **kwargs):
        fsize = (6, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.fig_key = fig_key
        self.panel_keys = ('partition_schematic', 'encoder_schematic',
                           'encoder_visualization', 'metric_schematic')
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}

        part_schem_grid = self.gs[:, :30]
        gss[self.panel_keys[0]] = self.get_axs((part_schem_grid,))
        
        encoder_schem_grid = self.gs[:40, 70:]
        gss[self.panel_keys[1]] = self.get_axs((encoder_schem_grid,))

        metric_schem_grid = self.gs[:, 36:55]
        gss[self.panel_keys[3]] = self.get_axs((metric_schem_grid,))

        encoder_vis_grid = pu.make_mxn_gridspec(self.gs, 2, 2,
                                                50, 100, 65, 100,
                                                8, 15)
        gss[self.panel_keys[2]] = self.get_axs(encoder_vis_grid)

        self.gss = gss
    
    def panel_encoder_visualization(self):
        key = self.panel_keys[2]
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

        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')

        dc.plot_source_manifold(fdg, pass_model, rs, n_arcs, 
                                source_scale_mag=.2,
                                rep_scale_mag=.01,
                                markers=False, axs=vis_axs,
                                titles=False)
        dg_color = self.params.getcolor('dg_color')
        plot_single_gen(gen_perf[0], class_ax, color=dg_color)
        plot_single_gen(gen_perf[1], regr_ax, color=dg_color)
        class_ax.set_ylabel('classifier\ngeneralization')
        regr_ax.set_ylabel('regression\ngeneralization')
        gpl.add_hlines(.5, class_ax)
        gpl.add_hlines(0, regr_ax)
        class_ax.set_ylim([.5, 1])
        regr_ax.set_ylim([0, 1])
            
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
                                         5, 8)
        plot_3d_axs = np.zeros((n_parts, 2), dtype=bool)
        plot_3d_axs[:, 1] = True
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
        ax_inp[0, 0].plot(p.explained_variance_ratio_, 'o', label='actual')
        ax_inp[0, 0].plot(p_scal.explained_variance_ratio_, 'o',
                          label='linear theory')
        ax_inp[0, 0].legend(frameon=False)
        ax_inp[0, 0].set_xlabel('PC number')
        ax_inp[0, 0].set_ylabel('proportion\nexplained')
        gpl.clean_plot(ax_inp[0, 0], 0)
        ax_inp[0, 1].plot(targs_dim[:, 0], targs_dim[:, 1], 'o')
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
        for i, num_p in enumerate(n_parts):
            hist = th[0, i, 0].history['loss']
            epochs = np.arange(1, len(hist) + 1)
            train_ax.plot(epochs, hist,
                          label='r${} = {}$'.format(npart_signifier,
                                                    num_p))
            dc.plot_source_manifold(fdg, models[0, i, 0], rs, n_arcs, 
                                    source_scale_mag=.2,
                                    rep_scale_mag=5, plot_model_3d=True,
                                    markers=False, axs=rep_axs[i],
                                    titles=False)
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
        
        bvae_perf = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 100, 55, 100,
                                         8, 10)
        gss[self.panel_keys[0]] = self.get_axs((bvae_schematic_grid,))
        gss[self.panel_keys[1]] = self.get_axs(bvae_perf)

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
        print(gen_perf)

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
    
    def __init__(self, fig_key='figure3', colors=colors, **kwargs):
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

        axs_left = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 100, 0, 40,
                                        3, 10)
        axs_right = pu.make_mxn_gridspec(self.gs, 3, 2, 0, 100, 54, 100,
                                         5, 15)
        gss[self.panel_keys[0]] = self.get_axs(np.concatenate((axs_left[0],
                                                              axs_right[0])))
        gss[self.panel_keys[1]] = self.get_axs(np.concatenate((axs_left[1],
                                                               axs_right[1])))
        gss[self.panel_keys[2]] = self.get_axs(np.concatenate((axs_left[2],
                                                               axs_right[2])))

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

        rep_scale_mag = 3

        pv_mask = np.array([False, True, False])
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
        rep_scale_mag = 3

        part_color = self.params.getcolor('partition_color')
        context_color = self.params.getcolor('contextual_color')
        colors = (part_color, context_color)
        
        labels = ('full partitions', 'contextual partitions')
        pv_mask = np.array([False, False, False, False, True, False, False,
                            False])
        self._standard_panel(fdg, m, run_inds, f_pattern, folder, axs,
                             labels=labels, pv_mask=pv_mask,
                             rep_scale_mag=rep_scale_mag, colors=colors)
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
        rep_scale_mag = 3

        part_color = self.params.getcolor('partition_color')
        partial_color1 = self.params.getcolor('partial_color1')
        partial_color2 = self.params.getcolor('partial_color1')
        colors = (part_color, partial_color1, partial_color2)
        
        labels = ('full information', '50% missing', '90% missing')
        pv_mask = np.array([False, False, False, True, False])
        self._standard_panel(fdg, m, run_inds, f_pattern, folder, axs,
                             labels=labels, pv_mask=pv_mask,
                             rep_scale_mag=rep_scale_mag, colors=colors)
        for ax in axs[:2]:
            ax.set_xlabel('')
            
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
        gss[self.panel_keys[0]] = self.get_axs((rf_schematic_grid,
                                                rf_projection_grid,
                                                rf_dec_grid[0, 0],
                                                rf_dec_grid[0, 1]))

        rep_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 100,
                                         55, 100, 5, 5)
        gss[self.panel_keys[1]] = self.get_axs(rep_grids)
        
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
        dc.plot_diagnostics(rfdg, pass_model, rs, n_arcs,
                            scale_mag=.2, markers=False, ax=proj_ax)

        if not key in self.data.keys():
            c_reps = self.params.getint('dg_classifier_reps')
            out = characterize_generalization(rfdg, pass_model,
                                              c_reps)
            self.data[key] = out
        results_class, results_regr = self.data[key]

        color = self.params.getcolor('dg_color')
        plot_single_gen(results_class, dec_c_ax, color=color)
        dec_c_ax.set_ylim([.5, 1])
        plot_single_gen(results_regr, dec_r_ax, color=color)
        dec_r_ax.set_ylim([0, 1])
        dec_c_ax.set_ylabel('classifier\ngeneralization')
        dec_r_ax.set_ylabel('regression\ngeneralization')

    def panel_disentangling_comparison(self, kernel=None):
        key = self.panel_keys[1]
        axs = self.gss[key]
        rfdg = self.make_rfdg(kernel=kernel)

        if not key in self.data.keys():
            out_fd = train_eg_fd(rfdg, self.params)
            out_bvae = train_eg_bvae(rfdg, self.params)
            self.data[key] = (out_fd, out_bvae)
        out_fd, out_bvae = self.data[key]
        m_fd = out_fd[0][0, 0]
        m_bvae = out_bvae[0][0, 0]

        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')
        fd_gen = characterize_generalization(rfdg, m_fd, 10)
        bvae_gen = characterize_generalization(rfdg, m_bvae, 10)
        print(np.mean(fd_gen[0], axis=0)) 
        print(np.mean(fd_gen[1], axis=0))
        print(np.mean(bvae_gen[0], axis=0))
        print(np.mean(bvae_gen[1], axis=0))

        run_ind_fd = self.params.get('run_ind_fd')  
        run_ind_beta = self.params.get('run_ind_beta')
        f_pattern = self.params.get('f_pattern')
        beta_f_pattern = self.params.get('beta_f_pattern')
        folder = self.params.get('mp_simulations_path')
        beta_folder = self.params.get('beta_simulations_path')
        dc.plot_diagnostics(rfdg, m_fd, rs, n_arcs, 
                            scale_mag=20, markers=False,
                            ax=axs[0, 0])
        dc.plot_diagnostics(rfdg, m_bvae, rs, n_arcs, 
                            scale_mag=.01, markers=False,
                            ax=axs[0, 1])
        res_axs = axs[1:]
        pv_mask = np.array([False, True, False])

        part_color = self.params.getcolor('partition_color')
        bvae_color = self.params.getcolor('bvae_color')
        
        dc.plot_recon_gen_summary(run_ind_fd, f_pattern, log_x=False, 
                                  collapse_plots=False, folder=folder,
                                  axs=res_axs, legend='partition',
                                  print_args=False, pv_mask=pv_mask,
                                  set_title=False, color=part_color)
        dc.plot_recon_gen_summary(run_ind_beta, beta_f_pattern, log_x=False, 
                                  collapse_plots=False, folder=beta_folder,
                                  axs=res_axs, legend=r'$\beta$VAE',
                                  print_args=False, pv_mask=pv_mask,
                                  set_title=False, color=bvae_color)

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
        gss[self.panel_keys[1]] = self.get_axs((rep_geom_fd, rep_geom_bvae,
                                                rep_geom_class_perf,
                                                rep_geom_regr_perf))

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
        
        out_f = dc.plot_diagnostics(shape_dg, m_fd, rs, n_arcs, n_dim_red=100,
                                    ax=rep_fd_ax, set_inds=(3, 4),
                                    scale_mag=20,
                                    dim_red_func=fd_red_func, ret_dim_red=True)
        out_b = dc.plot_diagnostics(shape_dg, m_bvae, rs, n_arcs, n_dim_red=100,
                                    ax=rep_bvae_ax, set_inds=(3, 4),
                                    dim_red_func=bvae_red_func, scale_mag=.2,
                                    ret_dim_red=True)
        if 'dr' not in self.data[key].keys():
            self.data[key]['dr'] = (out_f[1], out_b[1])
        res_ident, res_fd, res_bvae = self.data[key]['gen']
        dg_col = self.params.getcolor('dg_color')
        bvae_col = self.params.getcolor('bvae_color')
        fd_col = self.params.getcolor('partition_color')
        colors = (dg_col, fd_col, bvae_col)
        
        plot_multi_gen((res_ident[0], res_fd[0], res_bvae[0]), class_ax,
                       colors=colors)
        gpl.add_hlines(.5, class_ax)
        class_ax.set_ylim([.5, 1])
        plot_multi_gen((res_ident[1], res_fd[1], res_bvae[1]), regr_ax,
                       colors=colors)
        gpl.add_hlines(0, regr_ax)
        regr_ax.set_ylim([0, 1])

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
        
