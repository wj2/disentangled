import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np
import scipy.stats as sts
import sklearn.decomposition as skd
import sklearn.kernel_approximation as skka
import functools as ft
import matplotlib.pyplot as plt

import general.rf_models as rfm
import general.utility as u
import general.plotting as gpl
import disentangled.aux as da
import disentangled.regularizer as dr
import disentangled.disentanglers as dd

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

class DataGenerator(da.TFModel):
    
    def fit(self, train_x=None, train_y=None, eval_x=None, eval_y=None,
            source_distribution=None, epochs=15, train_samples=10**5,
            eval_samples=10**3, batch_size=32, **kwargs):
        if not self.compiled:
            self._compile()

        if source_distribution is not None:
            train_x = source_distribution.rvs(train_samples)
            train_y = train_x
            eval_x = source_distribution.rvs(eval_samples)
            eval_y = eval_x
            self.source_distribution = source_distribution
            
        if train_y is None:
            train_y = train_x
            
        if eval_x is not None and eval_y is None:
            eval_y = eval_x

        out = self.model.fit(train_x, train_y, validation_data=(eval_x, eval_y),
                             epochs=epochs, batch_size=batch_size, **kwargs)
        return out

    def representation_dimensionality(self, source_distribution=None,
                                      participation_ratio=False,
                                      sample_size=10**4, **pca_args):
        if source_distribution is None and self.source_distribution is not None:
            source_distribution = self.source_distribution
        elif source_distribution is None:
            raise Exception('no source distribution provided')

        samples = source_distribution.rvs(sample_size)
        rep = self.get_representation(samples)
        p = skd.PCA(**pca_args)
        p.fit(rep)
        if participation_ratio:
            pd = p.explained_variance_ratio_
            out = np.sum(pd)**2/np.sum(pd**2)
        else:
            out = (p.explained_variance_ratio_, p.components_)
        return out

    def sample_reps(self, sample_size=10**4, source_distribution=None,
                    repl_mean=None):
        if source_distribution is None and self.source_distribution is not None:
            source_distribution = self.source_distribution
        elif source_distribution is None:
            raise Exception('no source distribution provided')
        inp_samps = source_distribution.rvs(sample_size)
        if repl_mean is not None:
            inp_samps[:, repl_mean] = source_distribution.mean[repl_mean]
        rep_samps = self.get_representation(inp_samps)
        return inp_samps, rep_samps
    
class VariationalDataGenerator(DataGenerator):

    def __init__(self, inp_dim, transform_widths, out_dim,
                 beta=1):
        
        self.generator = self.make_generator(inp_dim, transform_widths,
                                             out_dim, beta=beta)
        self.degenerator = self.make_degenerator(inp_dim,
                                                 transform_widths[::-1],
                                                 out_dim)
        self.model = tfk.Model(inputs=self.generator.inputs,
                               outputs=self.degenerator(self.generator.outputs[0]))
        self.input_dim = inp_dim
        self.output_dim = out_dim
        self.compiled = False

    def make_generator(self, inp_dim, hidden_dims, out_dim,
                       layer_type=tfkl.Dense, act_func=tf.nn.relu,
                       expo_mean=100, beta=1):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=inp_dim))

        for hd in hidden_dims:
            l_i = layer_type(hd, activation=act_func)
            layer_list.append(l_i)

        last_layer = layer_type(out_dim, activation=act_func)
        layer_list.append(last_layer)

        p_size = tfpl.IndependentNormal.params_size(out_dim)
        layer_list.append(tfkl.Dense(p_size, activation=None))
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(out_dim), scale=1),
                                reinterpreted_batch_ndims=1)
        if beta > 0:
            rep_reg = tfpl.KLDivergenceRegularizer(prior, weight=beta)
        else:
            rep_reg = None
        rep_layer = tfpl.IndependentNormal(out_dim,
                                           activity_regularizer=rep_reg)

        layer_list.append(rep_layer)
        gen = tfk.Sequential(layer_list)
        return gen

    def make_degenerator(self, inp_dim, hidden_dims, out_dim,
                         layer_type=tfkl.Dense, act_func=tf.nn.relu,
                         out_eps=.01):
        layer_list = []
        
        layer_list.append(tfkl.InputLayer(input_shape=out_dim))
        for hd in hidden_dims:
            l_i = layer_type(hd, activation=act_func)
            layer_list.append(l_i)

        out_layer = layer_type(inp_dim, activation=None)
        layer_list.append(out_layer)

        fixed_std = lambda x: tfd.Normal(x, out_eps)
        layer_list.append(tfpl.DistributionLambda(
            make_distribution_fn=fixed_std))

        degen = tfk.Sequential(layer_list)
        return degen
    
    def _compile(self, optimizer=None,
                 loss=da.negloglik):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer, loss)
        self.compiled = True

    def get_representation(self, x):
        return self.generator(x).mean()

class ImageSourceDistrib(object):

    def __init__(self, datalist, set_partition=None, position_distr=None,
                 use_partition=False, partition_vec=None, offset=None):
        self.partition_func = None
        self.data_list = np.array(datalist)
        self.position_distr = position_distr
        self.dim = self.data_list.shape[1] 
        if position_distr is not None:
            self.dim = self.dim + position_distr.dim
        if use_partition:
            if partition_vec is not None:
                self.partition = u.make_unit_vector(np.array(partition_vec))
                self.offset = offset
                set_partition = ft.partial(da._binary_classification,
                                           plane=self.partition,
                                           off=self.offset)
            elif set_partition is None:
                out = da.generate_partition_functions(self.dim, n_funcs=1)
                pfs, vecs, offs = out
                set_partition = pfs[0]
                self.partition = vecs[0]
                self.offset = offs[0]
            else:
                self.partition = None
                self.offset = None
            self.partition_func = set_partition

    @property
    def mean(self):
        p_meds = np.percentile(self.data_list, 50, axis=0,
                               interpolation='nearest')
        return p_meds
            
    def rvs(self, rvs_shape):
        rvs = self._candidate_rvs(rvs_shape)
        while (self.partition_func is not None
               and not np.all(self.partition_func(rvs))):
            mask = np.logical_not(self.partition_func(rvs))
            sample = self._candidate_rvs(rvs_shape)
            rvs[mask] = sample[mask]
        return rvs

    def _candidate_rvs(self, n):
        inds = np.random.choice(self.data_list.shape[0], n)
        out = self.data_list[inds]
        if self.position_distr is not None:
            ps = self.position_distr.rvs(n)
            out = np.concatenate((out, ps), axis=1)
        return out

    def make_partition(self, set_partition=None, partition_vec=None,
                       offset=None):
        if partition_vec is not None and offset is None:
            offset = 0
        return ImageSourceDistrib(self.data_list,
                                  position_distr=self.position_distr,
                                  set_partition=set_partition,
                                  use_partition=True, offset=offset,
                                  partition_vec=partition_vec)

    def flip(self):
        if self.partition_func is not None:
            set_part = lambda x: np.logical_not(self.partition_func(x))
            new = ImageSourceDistrib(self.data_list, set_partition=set_part,
                                     position_distr=self.position_distr,
                                     use_partition=True, offset=self.offset)
            new.partition = -self.partition
            new.offset = self.offset
        else:
            print('no partition to flip')
            new = ImageSourceDistrib(self.data_list)
        return new        

class ChairSourceDistrib(ImageSourceDistrib):

    def __init__(self, *args, img_identifier='chair_id', **kwargs):
        super().__init__(*args, **kwargs)
        self.img_identifier = img_identifier
    
class ImageDatasetGenerator(DataGenerator):

    def __init__(self, data, img_params, include_position=False,
                 position_distr=None, max_move=4, **kwargs):
        if include_position and position_distr is None:
            position_distr = sts.multivariate_normal((0, 0), (.5*max_move)**2)
        self.include_position = include_position
        self.position_distr = position_distr
        self.data_table = data
        self.img_params = img_params
        self.n_img_params = len(self.img_params)
        self.img_out_label = ['images'][0]
        self.source_distribution = ImageSourceDistrib(
            data[self.img_params],
            position_distr=position_distr)
        self.img_size = data[self.img_out_label][0].shape
        self.params = self.img_params
        self.data_dict = None
        if include_position:
            self.params = self.params + ['horiz_offset', 'vert_offset']
        else:
            self._make_dict()
        self.output_dim = self.img_size
        self.input_dim = len(self.params)
        self.max_move = max_move
        self.x_uniques = None

    def _make_dict(self):
        data_dict = {}
        for i, row in self.data_table.iterrows():
            ident = row[self.img_params]
            img = row['images']
            data_dict[tuple(ident)] = img
        self.data_dict = data_dict
        
    def fit(*args, **kwargs):
        return tf.keras.callbacks.History()
    
    def generator(self, x):
        return self.get_representation(x)

    def _move_img(self, img, coords):
        coords[coords > self.max_move] = self.max_move
        coords[coords < -self.max_move] = -self.max_move
        img_dim = np.array(self.output_dim[:-1])
        shifts = np.round(img_dim*coords/2).astype(int)
        s_img = np.roll(img, shifts, axis=(0,1))
        return s_img

    def ppf(self, perc, dim):
        if dim < self.n_img_params:
            vals = self.data_table[self.img_params[dim]]
            p_val = np.percentile(vals, perc*100, interpolation='nearest')
        else:
            p_val = sts.norm(0, .5*self.max_move).ppf(perc)
            if p_val > self.max_move:
                p_val = self.max_move
            elif p_val < -self.max_move:
                p_val = -self.max_move
        return p_val

    def get_center(self):
        p_meds = np.percentile(self.data_table[self.img_params], 50, axis=0,
                               interpolation='nearest')
        if self.include_position:
            m_meds = self.position_distr.mean
            p_meds = np.concatenate((p_meds, m_meds))
        return p_meds

    def _get_uniques(self):
        if self.x_uniques is None:
            img_params = np.array(self.data_table[self.img_params])
            self.x_uniques = list(np.unique(img_params[:, i])
                                  for i in range(self.n_img_params))
        return self.x_uniques
    
    def get_representation(self, x, flat=False, same_img=False, nearest=True):
        x = np.array(x)
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        out = np.zeros(x.shape[0], dtype=object)
        img_params = np.array(self.data_table[self.img_params])
        if nearest:
            new_x = np.zeros_like(x, dtype=float)
            x_uniques = self._get_uniques()
            for i, xi in enumerate(x):
                for j, xu in enumerate(x_uniques):
                    xind = np.argmin(np.abs(xu - xi[j]))
                    new_x[i, j] = xu[xind]
            new_x[:, self.n_img_params:] = x[:, self.n_img_params:]
            x = new_x
        if same_img and self.img_identifier is not None:
            img_ids = self.data_table[self.img_identifier]
            chosen_id = np.random.choice(img_ids, 1)[0]
            id_mask = img_ids == chosen_id
        # this is a bit slow
        for i, xi in enumerate(x):
            xi_img = xi[:self.n_img_params]
            if self.data_dict is None:
                mask = np.product(img_params == xi_img,
                                  axis=1, dtype=bool)
                if same_img and self.img_identifier is not None:
                    mask = mask*id_mask
                s = np.array(self.data_table[self.img_out_label])[mask]
                out_ind = np.random.choice(range(s.shape[0]))
                samp = s[out_ind]
                if self.position_distr is not None:
                    samp = self._move_img(samp, xi[self.n_img_params:])
                if flat:
                    samp = samp.flatten()
                out[i] = samp
            else:
                out[i] = self.data_dict[tuple(xi_img)]
        return np.stack(out)

    def representation_dimensionality(self, source_distribution=None,
                                      sample_size=10**1, participation_ratio=False,
                                      **pca_args):
        if source_distribution is None and self.source_distribution is not None:
            source_distribution = self.source_distribution
        elif source_distribution is None:
            raise Exception('no source distribution provided')

        samples = source_distribution.rvs(sample_size)
        rep = self.get_representation(samples, flat=True)
        p = skd.PCA(**pca_args)
        p.fit(rep)
        if participation_ratio:
            pd = p.explained_variance_ratio_
            out = np.sum(pd)**2/np.sum(pd**2)
        else:
            out = (p.explained_variance_ratio_, p.components_)
        return out        
    
class ChairGenerator(ImageDatasetGenerator):

    def __init__(self, folder, norm_params=True, img_size=(128, 128),
                 include_position=False, position_distr=None, max_move=4,
                 max_load=np.inf, filter_edges=None,
                 param_keys=['rotation', 'pitch'], **kwargs):
        data = da.load_chair_images(folder, img_size=img_size, norm_params=True,
                                    max_load=max_load, filter_edges=filter_edges,
                                    **kwargs)
        super().__init__(data, param_keys,
                         include_position=include_position,
                         position_distr=position_distr,
                         max_move=max_move)    

class TwoDShapeGenerator(ImageDatasetGenerator):

    default_pks = ['shape', 'scale','orientation', 'x_pos', 'y_pos']
    def __init__(self, folder, img_size=(64, 64), norm_params=True,
                 max_load=np.inf, param_keys=default_pks, convert_color=False,
                 **kwargs):
        self.img_identifier = None
        data = da.load_2d_shapes(folder, img_size=img_size,
                                 convert_color=convert_color,
                                 norm_params=norm_params,
                                 max_load=max_load)
        super().__init__(data, param_keys, **kwargs)

class ThreeDShapeGenerator(ImageDatasetGenerator):

    default_pks = ['floor_hue', 'wall_hue', 'object_hue','scale', 'shape',
                   'orientation']
    def __init__(self, folder, img_size=(64, 64), norm_params=True,
                 max_load=np.inf, param_keys=default_pks, **kwargs):
        self.img_identifier = None
        data = da.load_3d_shapes(folder, img_size=img_size,
                                 norm_params=norm_params,
                                 max_load=max_load)
        super().__init__(data, param_keys, **kwargs)

class RFDataGenerator(DataGenerator):

    def __init__(self, inp_dim, out_dim, source_distribution=None, noise=0.001,
                 distrib_variance=1, setup_distribution=None, total_out=False,
                 width_scaling=4, input_noise=0, low_thr=None):
        if source_distribution is None:
            source_distribution = sts.multivariate_normal(np.zeros(inp_dim),
                                                          distrib_variance)
        if setup_distribution is None:
            setup_distribution = source_distribution
        try:
            sd_list = setup_distribution.get_indiv_distributions()
        except:
            sd_list = [sts.norm(m, np.sqrt(setup_distribution.cov[i, i]))
                       for i, m in enumerate(setup_distribution.mean)]
        if total_out:
            out_dim = int(np.round(.5*out_dim**(1/inp_dim))*2)
        self.input_dim = inp_dim
        out = self.make_generator(out_dim, sd_list, input_noise_var=input_noise,
                                  noise=noise, width_scaling=width_scaling)
        self.generator, self.rf_cents, self.rf_wids = out
        self.output_dim = len(self.rf_cents)
        self.low_thr = low_thr
        self.compiled = True
        self.source_distribution = source_distribution

    def plot_rfs(self, ax=None, plot_dots=False, color=None, make_scales=True,
                 thin=1):
        if ax is None:
            f, ax = plt.subplots(1, 1)
        cps = da.get_circle_pts(100, 2)
        for i, rfc in enumerate(self.rf_cents[::thin]):
            rfw = np.sqrt(self.rf_wids[i])
            l = ax.plot(cps[:, 0]*rfw[0] + rfc[0],
                        cps[:, 1]*rfw[1] + rfc[1],
                        color=color)
            if plot_dots:
                ax.plot(rfc[0], rfc[1], 'o',
                        color=l[0].get_color())
        if make_scales:
            x_scale = self.source_distribution.cov[0, 0]
            y_scale = self.source_distribution.cov[1, 1]
            gpl.make_xaxis_scale_bar(ax, x_scale, label='dimension 1')
            gpl.make_yaxis_scale_bar(ax, y_scale, label='dimension 2')
            gpl.clean_plot(ax, 0)
        
    def make_generator(self, out_dim, source_distribution, noise=.01,
                       scale=1, baseline=0, width_scaling=1, input_noise_var=0):
        out = rfm.get_distribution_gaussian_resp_func(out_dim,
                                                      source_distribution,
                                                      scale=scale,
                                                      baseline=baseline,
                                                      wid_scaling=width_scaling)
        rfs, _, ms, ws = out
        noise_distr = sts.multivariate_normal(np.zeros(len(ms)), noise,
                                              allow_singular=True)
        in_distr = sts.multivariate_normal(np.zeros(self.input_dim),
                                           input_noise_var,
                                           allow_singular=True)
        def gen(x, input_noise=True, output_noise=True):
            if input_noise:
                in_noise = in_distr.rvs(x.shape[0])
            else:
                in_noise = 0
            if output_noise:
                out_noise = noise_distr.rvs(x.shape[0])
            else:
                out_noise = 0
            samps = rfs(x + in_noise) + out_noise
            if self.low_thr is not None:
                samps[samps < self.low_thr] = 0 
            return samps
        return gen, ms, ws

    def get_representation(self, x, **kwargs):
        return self.generator(x, **kwargs)

    def fit(*args, **kwargs):
        return tf.keras.callbacks.History()

class KernelDataGenerator(DataGenerator):

    def __init__(self, inp_dim, transform_widths, out_dim,
                 kernel_func=skka.RBFSampler, l2_weight=(0, .1),
                 layer=None, distrib_variance=1, low_thr=None,
                 source_distribution=None, noise=.01, **kernel_kwargs):
        self.kernel = kernel_func(n_components=out_dim, **kernel_kwargs)
        self.input_dim = inp_dim
        self.output_dim = out_dim
        self.compiled = False
        self.kernel_init = False
        if source_distribution is None:
            source_distribution = sts.multivariate_normal(np.zeros(inp_dim),
                                                          distrib_variance)
        self.source_distribution = source_distribution
        if layer is not None:
            self.layer = dd.SingleLayer(out_dim, layer)
        else:
            self.layer = dd.IdentityModel()
        self.low_thr = low_thr

    def fit(self, train_x=None, train_y=None, eval_x=None, eval_y=None,
            source_distribution=None, epochs=15, train_samples=10**5,
            eval_samples=10**3, batch_size=32, **kwargs):
        if source_distribution is not None:
            train_x = source_distribution.rvs(train_samples)
            train_y = train_x
            self.source_distribution = source_distribution
        self.kernel.fit(train_x)
        self.kernel_init = True
        
    def _compile(self, *args, **kwargs):
        self.compiled = True

    def generator(self, x):
        if not self.kernel_init:
            self.kernel.fit(x)
            self.kernel_init = True
        out = self.layer.get_representation(self.kernel.transform(x))
        if self.low_thr is not None:
            out[out < self.low_thr] = 0
        return out
        
    def get_representation(self, x):
        if not self.kernel_init:
            self.kernel.fit(x)
            self.kernel_init = True
        return self.generator(x)

class ShiftMapDataGenerator(DataGenerator):

    def __init__(self, inp_dim, transform_widths, out_dim,
                 shift_base=2, layer=None, distrib_support=(-.5, .5),
                 source_distribution=None):
        if source_distribution is None:
            source_distribution = da.MultivariateUniform(inp_dim,
                                                         distrib_support)
        self.source_distribution = source_distribution
        units_per_dim = int(np.floor(out_dim/inp_dim))
        shift_map = np.ones((units_per_dim, inp_dim))
        for i in range(units_per_dim):
            shift_map[i] = shift_base**(i + 1)
        self.shift_map = shift_map
        self.input_dim = inp_dim
        self.output_dim = units_per_dim*inp_dim

    def generator(self, x):
        x_exp = np.expand_dims(x, 1)
        x_lin = self.shift_map*x_exp
        x_code = x_lin - np.round(x_lin)
        
        x_out = np.reshape(x_code, (x_code.shape[0], -1))
        return x_out

    def get_representation(self, x):
        return self.generator(x)
        
class FunctionalDataGenerator(DataGenerator):

    def __init__(self, inp_dim, transform_widths, out_dim, l2_weight=(0, .1),
                 source_distribution=None, noise=.01, **kwargs):
        
        self.generator = self.make_generator(inp_dim, transform_widths,
                                             out_dim, l2_weight=l2_weight,
                                             noise=noise, **kwargs)
        self.degenerator = self.make_degenerator(inp_dim,
                                                 transform_widths[::-1],
                                                 out_dim, **kwargs)
        self.model = tfk.Model(inputs=self.generator.inputs,
                               outputs=self.degenerator(self.generator.outputs[0]))
        self.input_dim = inp_dim
        self.output_dim = out_dim
        self.compiled = False
        self.source_distribution = source_distribution

    def make_generator(self, inp_dim, hidden_dims, out_dim,
                       layer_type=tfkl.Dense, act_func=tf.nn.relu,
                       expo_mean=5, l2_weight=1, noise=None):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=inp_dim))

        regularizer = dr.VarianceRegularizer(l2_weight)

        for hd in hidden_dims:
            l_i = layer_type(hd, activation=act_func,
                             activity_regularizer=regularizer)
            layer_list.append(l_i)
            if noise is not None:
                layer_list.append(tfkl.GaussianNoise(noise))

        last_layer = layer_type(out_dim, activation=act_func,
                                activity_regularizer=regularizer)
        layer_list.append(last_layer)

        gen = tfk.Sequential(layer_list)
        return gen

    def make_degenerator(self, inp_dim, hidden_dims, out_dim,
                         layer_type=tfkl.Dense, act_func=tf.nn.relu):
        layer_list = []
        
        layer_list.append(tfkl.InputLayer(input_shape=out_dim))
        for hd in hidden_dims:
            l_i = layer_type(hd, activation=act_func)
            layer_list.append(l_i)

        out_layer = layer_type(inp_dim, activation=None)
        layer_list.append(out_layer)

        degen = tfk.Sequential(layer_list)
        return degen

    def save(self, path):
        tf_entries = ('generator', 'degenerator', 'model')
        self._save_wtf(path, tf_entries)

    @classmethod
    def load(cls, path):
        dummy = FunctionalDataGenerator(0, (10,), 0)
        return cls._load_model(dummy, path)
    
    def _compile(self, optimizer=None,
                 loss=tf.losses.MeanSquaredError()):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer, loss)
        self.compiled = True
        
    def get_representation(self, x):
        return self.generator(x)
