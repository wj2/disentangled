
import numpy as np
import scipy.stats as sts
import functools as ft

import general.plotting as gpl
import general.plotting_styles as gps
import general.paper_utilities as pu
import general.utility as u
import disentangled.data_generation as dg
import disentangled.disentanglers as dd
import disentangled.characterization as dc
import disentangled.aux as da

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
    gpl.violinplot(results.T, xs, ax=ax, color=(color, color))
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    gpl.clean_plot(ax, 0)
    gpl.clean_plot_bottom(ax, keeplabels=True)
    return ax

def plot_multi_gen(res_list, ax, xs=None, labels=('standard', 'gen'),
                   sep=.2):
    if xs is None:
        xs = np.array([0, 1])
    start_xs = xs - len(res_list)/4
    for i, rs in enumerate(res_list):
        plot_single_gen(rs, ax, xs=start_xs + i*sep)
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
                                   n_train_samps=n_train_eg,
                                   use_mp=True, n_reps=1,
                                   batch_size=batch_size,
                                   hide_print=hide_print)
    return out

def characterize_generalization(dg, model, c_reps, train_samples=500,
                                test_samples=500, bootstrap_regr=True,
                                n_boots=1000, norm=True, cut_zero=True):
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
            n_iters=1)[0]
        results_class[i, 1] = dc.classifier_generalization(
            dg, model, train_distrib=train_distr,
            test_distrib=test_distr, n_train_samples=train_samples,
            n_test_samples=test_samples, n_iters=1)[0]
        results_regr[i, 0] = dc.find_linear_mapping_single(
            dg, model, half=False, n_samps=train_samples)[1]
        results_regr[i, 1] = dc.find_linear_mapping_single(
            dg, model, n_samps=train_samples)[1]
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

class Figure2(pu.Figure):
    
    def __init__(self, fig_key='figure2', colors=colors, **kwargs):
        fsize = (6, 5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.panel_keys = ('order_disorder', 'training_rep', 'rep_summary')
        super().__init__(fsize, params, colors=colors, **kwargs)
    
    def make_gss(self):
        gss = {}

        ordered_rep_grid = self.gs[:25, :30]
        class_perf_grid = self.gs[75:, :15]
        regr_perf_grid = self.gs[75:, 30:45]
        gss[self.panel_keys[0]] = self.get_axs((ordered_rep_grid,
                                                class_perf_grid,
                                                regr_perf_grid))
        
        train_grid = self.gs[:15, 35:55]
        train_ax = self.get_axs((train_grid,))[0]
        n_parts = len(self.params.getlist('n_parts'))
        rep_grids = pu.make_mxn_gridspec(self.gs, n_parts, 2,
                                         0, 65, 70, 100,
                                         5, 5)
        rep_axs = self.get_axs(rep_grids, sharex='vertical',
                               sharey='vertical')
        gss[self.panel_keys[1]] = train_ax, rep_axs
        
        rep_classifier_grid = self.gs[75:, 60:75]
        rep_regression_grid = self.gs[75:, 85:]
        gss[self.panel_keys[2]] = self.get_axs((rep_classifier_grid,
                                                rep_regression_grid))
        self.gss = gss
    
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
                                             l2_weight=dg_regweight,
                                             source_distribution=source_distr)
            fdg.fit(source_distribution=source_distr, epochs=dg_epochs,
                    train_samples=dg_train_egs)
            self.fdg = fdg 
        return fdg

    def panel_order_disorder(self):
        key = self.panel_keys[0]
        latent_ax, class_ax, regr_ax = self.gss[key]
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
        dc.plot_diagnostics(fdg, pass_model, rs, n_arcs, plot_source=True,
                            dim_red=False,
                            scale_mag=.2, markers=False, ax=latent_ax)
        dg_color = self.params.getcolor('dg_color')
        plot_single_gen(gen_perf[0], class_ax, color=dg_color)
        plot_single_gen(gen_perf[1], regr_ax, color=dg_color)
        class_ax.set_ylabel('classifier\ngeneralization')
        regr_ax.set_ylabel('regression\ngeneralization')
        gpl.add_hlines(.5, class_ax)
        gpl.add_hlines(0, regr_ax)

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

            model_kinds = list(ft.partial(dd.FlexibleDisentanglerAE,
                                          true_inp_dim=fdg.input_dim, 
                                          n_partitions=num_p,
                                          no_autoenc=True) 
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
                                    rep_scale_mag=5,
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

class Figure3(pu.Figure):
    
    def __init__(self, fig_key='figure3', colors=colors, **kwargs):
        fsize = (5.5, 3.5)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.panel_keys = ('unbalanced_partitions', 'contextual_partitions',
                           'bvae_performance')
        super().__init__(fsize, params, colors=colors, **kwargs)        

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

    def panel_bvae_performance(self):
        key = self.panel_keys[2]
        axs = self.gss[key]
        if not key in self.data.keys():
            fdg = self.make_fdg()

            out = train_eg_bvae(fdg, self.params)
            
            self.data[key] = (fdg, out)
        fdg, out = self.data[key]
        m, _ = out

        run_inds = (self.params.get('beta_eg_ind'),)
        f_pattern = self.params.get('beta_f_pattern')
        folder = self.params.get('beta_simulations_path')
        labels = (r'$\beta$VAE',)

        bvae_color = self.params.getcolor('bvae_color')
        colors = (bvae_color,)
        
        m[0, 0].p_vectors = []
        m[0, 0].p_offsets = []
        pv_mask = np.array([False, True, False])
        self._standard_panel(fdg, m, run_inds, f_pattern, folder, axs,
                             labels=labels, pv_mask=pv_mask,
                             xlab=r'$\beta$', colors=colors)
        for ax in axs:
            ax.set_ylabel('')
            
class Figure4(pu.Figure):

    def __init__(self, fig_key='figure4', colors=colors, **kwargs):
        fsize = (6, 4)
        cf = u.ConfigParserColor()
        cf.read(config_path)
        
        params = cf[fig_key]
        self.panel_keys = ('rf_input', 'disentangling_comparison')
        super().__init__(fsize, params, colors=colors, **kwargs)        

    def make_rfdg(self, retrain=False):
        try:
            assert not retrain
            rfdg = self.rfdg
        except:
            inp_dim = self.params.getint('inp_dim')
            dg_dim = self.params.getint('dg_dim')
            in_noise = self.params.getfloat('in_noise')
            out_noise = self.params.getfloat('out_noise')
            width_scaling = self.params.getfloat('width_scaling')
            dg_source_var = self.params.getfloat('dg_source_var')
            
            source_distr = sts.multivariate_normal(np.zeros(inp_dim),
                                                   dg_source_var)
            rfdg = dg.RFDataGenerator(inp_dim, dg_dim, total_out=True, 
                                      input_noise=in_noise, noise=out_noise,
                                      width_scaling=width_scaling,
                                      source_distribution=source_distr)
            self.rfdg = rfdg 
        return rfdg
        
    def make_gss(self):
        gss = {}

        rf_schematic_grid = self.gs[:50, :25]
        rf_projection_grid = self.gs[:50, 25:50]
        rf_decoding_grid = self.gs[50:, 30:45]
        gss[self.panel_keys[0]] = self.get_axs((rf_schematic_grid,
                                                rf_projection_grid,
                                                rf_decoding_grid))

        rep_grids = pu.make_mxn_gridspec(self.gs, 2, 2, 0, 100,
                                         55, 100, 5, 5)
        gss[self.panel_keys[1]] = self.get_axs(rep_grids)
        
        self.gss = gss

    def panel_rf_input(self):
        key = self.panel_keys[0]
        schem_ax, proj_ax, dec_ax = self.gss[key]
        rfdg = self.make_rfdg()

        rf_eg_color = self.params.getcolor('rf_eg_color')
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
        plot_single_gen(results_class, dec_ax, color=color)

    def panel_disentangling_comparison(self):
        key = self.panel_keys[1]
        axs = self.gss[key]
        rfdg = self.make_rfdg()

        if not key in self.data.keys():
            out_fd = train_eg_fd(rfdg, self.params)
            out_bvae = train_eg_bvae(rfdg, self.params)
            self.data[key] = (out_fd, out_bvae)
        out_fd, out_bvae = self.data[key]
        m_fd = out_fd[0][0, 0]
        m_bvae = out_bvae[0][0, 0]

        rs = self.params.getlist('manifold_radii', typefunc=float)
        n_arcs = self.params.getint('manifold_arcs')

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
        
        dc.plot_recon_gen_summary(run_ind_fd, f_pattern, log_x=False, 
                                  collapse_plots=False, folder=folder,
                                  axs=res_axs, legend='partition',
                                  print_args=False, pv_mask=pv_mask,
                                  set_title=False)
        dc.plot_recon_gen_summary(run_ind_beta, beta_f_pattern, log_x=False, 
                                  collapse_plots=False, folder=beta_folder,
                                  axs=res_axs, legend=r'$\beta$VAE',
                                  print_args=False, pv_mask=pv_mask,
                                  set_title=False)

class Figure5(pu.Figure):

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
                                         0, 30, 3, 1)        
        gss[self.panel_keys[0]] = self.get_axs(img_grids)

        rep_geom_fd = self.gs[30:65, :14]
        rep_geom_bvae = self.gs[30:55, 16:30]
        rep_geom_class_perf = self.gs[60:, :20]
        rep_geom_regr_perf = self.gs[60:, 20:40]
        gss[self.panel_keys[1]] = self.get_axs((rep_geom_fd, rep_geom_bvae,
                                                rep_geom_class_perf,
                                                rep_geom_regr_perf))

        recon_grids = pu.make_mxn_gridspec(self.gs, 5, 7, 0, 100,
                                           40, 100, 3, 1)        
        gss[self.panel_keys[2]] = self.get_axs(recon_grids)
        
        self.gss = gss

    def panel_img_egs(self):
        key = self.panel_keys[0]
        axs = self.gss[key]
        shape_dg = self.make_shape_dg()

        out = shape_dg.sample_reps(sample_size=np.product(axs.shape))
        _, sample_imgs = out
        for i, ind in enumerate(u.make_array_ind_iterator(axs.shape)):
            axs[ind].imshow(sample_imgs[i])
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
            res_fd = characterize_generalization(shape_dg, m_fd,
                                                 c_reps, norm=False)
            res_bvae = characterize_generalization(shape_dg, m_bvae,
                                                   c_reps, norm=False)
            self.data[key]['gen'] = (res_fd, res_bvae)

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
        res_fd, res_bvae = self.data[key]['gen']
        plot_multi_gen((res_fd[0], res_bvae[0]), class_ax)
        plot_multi_gen((res_fd[1], res_bvae[1]), regr_ax)

    def panel_traversal_comparison(self):
        key = self.panel_keys[2]
        axs = self.gss[key]
        shape_dg = self.make_shape_dg()
        m_fd, m_bvae = self._get_eg_models()

        traverse_dim = self.params.getint('traverse_dim')
        learn_dim = self.params.getint('learn_dim')
        n_pts = axs.shape[1]
        
        fd_perturb = self.params.getfloat('fd_perturb')
        bvae_perturb = self.params.getfloat('bvae_perturb')
        eps_d = self.params.getfloat('eps_d')
        
        out = dc.plot_traversal_plot(shape_dg, m_fd, full_perturb=fd_perturb,
                                     trav_dim=traverse_dim, n_pts=n_pts,
                                     eps_d=eps_d, learn_dim=learn_dim,
                                     n_dense_pts=n_pts)
        recs, di, dl, dr, lr = out
        dc.plot_img_series(di, title='', axs=axs[0])
        dc.plot_img_series(dr, title='', axs=axs[1])
        dc.plot_img_series(recs, title='', axs=axs[2])

        out = dc.plot_traversal_plot(shape_dg, m_bvae,
                                     full_perturb=bvae_perturb,
                                     trav_dim=traverse_dim, n_pts=n_pts,
                                     eps_d=eps_d, n_dense_pts=n_pts,
                                     learn_dim=learn_dim)
        recs, di, dl, dr, lr = out
        dc.plot_img_series(dr, title='', axs=axs[3])
        dc.plot_img_series(recs, title='', axs=axs[4])
        
