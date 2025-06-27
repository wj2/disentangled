import tensorflow as tf
import tensorflow_probability as tfp

import collections as c
import numpy as np
import scipy as sp
import scipy.stats as sts
import scipy.special as ss
import sklearn.decomposition as skd
import sklearn.kernel_approximation as skka
import sklearn.gaussian_process as skgp
import sklearn.preprocessing as skp
import functools as ft
import matplotlib.pyplot as plt
import sklearn.pipeline as sklpipe

import general.rf_models as rfm
import general.utility as u
import general.plotting as gpl
import disentangled.auxiliary as da
import disentangled.regularizer as dr
import disentangled.disentanglers as dd
import modularity.auxiliary as maux
import composite_tangling.code_creation as cc

import superposition_codes.codes as spc

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class DataGenerator(da.TFModel):
    def fit(
        self,
        train_x=None,
        train_y=None,
        eval_x=None,
        eval_y=None,
        source_distribution=None,
        epochs=15,
        train_samples=10**5,
        eval_samples=10**3,
        batch_size=32,
        **kwargs
    ):
        if not self.compiled:
            self._compile()
        if source_distribution is None and self.source_distribution is not None:
            source_distribution = self.source_distribution
        if source_distribution is not None and train_x is None:
            train_x = source_distribution.rvs(train_samples)
            train_y = train_x
            eval_x = source_distribution.rvs(eval_samples)
            eval_y = eval_x
            self.source_distribution = source_distribution
        if train_y is None:
            train_y = train_x

        if eval_x is not None and eval_y is None:
            eval_y = eval_x

        out = self.model.fit(
            train_x,
            train_y,
            validation_data=(eval_x, eval_y),
            epochs=epochs,
            batch_size=batch_size,
            **kwargs
        )
        return out

    def representation_dimensionality(
        self, participation_ratio=False, sample_size=10**4, **pca_args
    ):
        samples, rep = self.sample_reps(sample_size)
        if participation_ratio:
            out = u.participation_ratio(rep)
        else:
            p = skd.PCA(**pca_args)
            p.fit(rep)
            out = (p.explained_variance_ratio_, p.components_)
        return out

    def sample_reps(
        self, sample_size=10**4, source_distribution=None, repl_mean=None
    ):
        if source_distribution is None and self.source_distribution is not None:
            source_distribution = self.source_distribution
        elif source_distribution is None:
            raise Exception("no source distribution provided")
        inp_samps = source_distribution.rvs(sample_size)
        if len(inp_samps.shape) < 2 and sample_size == 1:
            inp_samps = np.expand_dims(inp_samps, 0)
        if len(inp_samps.shape) < 2 and self.input_dim == 1:
            inp_samps = np.expand_dims(inp_samps, 1)

        if repl_mean is not None:
            inp_samps[:, repl_mean] = source_distribution.mean[repl_mean]
        rep_samps = self.get_representation(inp_samps)
        return inp_samps, rep_samps


class IdentityDG(DataGenerator):
    def __init__(self, distr):
        self.model = dd.IdentityModel()
        self.source_distribution = distr
        self.input_dim = distr.dim

    def generator(self, x):
        return self.get_representation(x)

    def get_representation(self, x):
        return self.model.get_representation(x)


def visualize_gp(
    length_scale,
    inp_dim=2,
    dim=0,
    domain=(-2, 2),
    n_samples=1000,
    func_samps=1,
    plot_dim=1,
    ax=None,
    fwid=5,
    **kwargs
):
    if ax is None:
        if plot_dim == 3:
            f = plt.figure(figsize=(fwid, fwid))
            ax = f.add_subplot(1, 1, 1, projection="3d")
        else:
            f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
    gp = GaussianProcessDataGenerator(inp_dim, 0, 1, length_scale=length_scale)
    inp = np.zeros((n_samples, gp.input_dim))
    inp_val = np.sort(gp.source_distribution.rvs(n_samples)[:, dim])
    inp_val = np.linspace(*domain, n_samples)
    inp[:, dim] = inp_val
    out = gp.model.sample_y(inp, n_samples=func_samps * plot_dim, random_state=None)
    if plot_dim == 1:
        ax.plot(inp_val, out, **kwargs)
    elif plot_dim == 2:
        out = out.reshape((out.shape[0], 2, -1))
        ax.plot(out[:, 0], out[:, 1], **kwargs)
    elif plot_dim == 3:
        out = out.reshape((out.shape[0], 3, -1))
        for i in range(out.shape[2]):
            ax.plot(out[:, 0, i], out[:, 1, i], out[:, 2, i], **kwargs)
    return ax


class RollDataGenerator(DataGenerator):
    def __init__(
        self,
        inp_dim,
        transform_width,
        out_dim,
        period_distribution=None,
        source_distribution=None,
        single_period=None,
        use_unit_directions=True,
        distrib_variance=1,
    ):
        if source_distribution is None:
            source_distribution = sts.multivariate_normal(
                np.zeros(inp_dim), distrib_variance
            )
        self.source_distribution = source_distribution

        if period_distribution is None:
            period_distribution = sts.uniform(0.1, 0.5)
        self.period_distribution = period_distribution

        self.input_dim = inp_dim
        self.output_dim = out_dim
        self.use_unit_directions = use_unit_directions
        self.single_period = single_period

    def fit(
        self,
        train_x=None,
        train_y=None,
        eval_x=None,
        eval_y=None,
        source_distribution=None,
        epochs=15,
        train_samples=1000,
        eval_samples=10**3,
        batch_size=32,
        **kwargs
    ):
        if self.single_period is None:
            periods = self.period_distribution.rvs((self.output_dim, 1))
        else:
            periods = np.ones((self.output_dim, 1)) * self.single_period

        self.periods = periods

        directions = sts.norm(0, 1).rvs((self.output_dim, self.input_dim))
        intercepts = sts.norm(0, 1).rvs((self.output_dim, 1))
        if self.use_unit_directions:
            directions = u.make_unit_vector(directions, squeeze=False)
        self.directions = directions
        self.intercepts = intercepts

    def _compile(self, *args, **kwargs):
        self.compiled = True

    def generator(self, x):
        if len(x.shape) == 1:
            x = np.expand_dims(x, 0)
        raw = x @ self.directions.T + self.intercepts.T
        out = np.sin(raw / self.periods.T)
        return out

    def get_representation(self, x):
        return self.generator(x)


class GaussianProcessDataGenerator(DataGenerator):
    def __init__(
        self,
        inp_dim,
        transform_width,
        out_dim,
        kernel_func=skgp.kernels.RBF,
        l2_weight=(0, 0.1),
        layer=None,
        distrib_variance=1,
        low_thr=None,
        source_distribution=None,
        noise=0.01,
        rand_state=None,
        alpha=1e-10,
        **kernel_kwargs
    ):
        self.input_dim = inp_dim
        self.output_dim = out_dim
        self.compiled = False
        self.kernel_init = True
        kernel = kernel_func(**kernel_kwargs)
        self.model = skgp.GaussianProcessRegressor(kernel, alpha=alpha)
        if source_distribution is None:
            source_distribution = sts.multivariate_normal(
                np.zeros(inp_dim), distrib_variance
            )
        self.source_distribution = source_distribution
        if layer is not None:
            self.layer = dd.SingleLayer(out_dim, layer)
        else:
            self.layer = dd.IdentityModel()
        self.low_thr = low_thr
        self.rand_state = rand_state

    def fit(
        self,
        train_x=None,
        train_y=None,
        eval_x=None,
        eval_y=None,
        source_distribution=None,
        epochs=15,
        train_samples=1000,
        eval_samples=10**3,
        batch_size=32,
        **kwargs
    ):
        in_samp = self.source_distribution.rvs(train_samples)
        if self.input_dim == 1:
            in_samp = in_samp.reshape((-1, 1))
        samp_proc = self.model.sample_y(
            in_samp, n_samples=self.output_dim, random_state=self.rand_state
        )
        self.model.fit(in_samp, samp_proc)

    def _compile(self, *args, **kwargs):
        self.compiled = True

    def generator(self, x):
        if len(x.shape) == 1:
            x = np.reshape(x, (1, -1))
        out = self.layer.get_representation(self.model.predict(x))
        if self.low_thr is not None:
            out[out < self.low_thr] = 0
        return out

    def get_representation(self, x):
        return self.generator(x)


class VariationalDataGenerator(DataGenerator):
    def __init__(self, inp_dim, transform_widths, out_dim, beta=1):
        self.generator = self.make_generator(
            inp_dim, transform_widths, out_dim, beta=beta
        )
        self.degenerator = self.make_degenerator(
            inp_dim, transform_widths[::-1], out_dim
        )
        self.model = tfk.Model(
            inputs=self.generator.inputs,
            outputs=self.degenerator(self.generator.outputs[0]),
        )
        self.input_dim = inp_dim
        self.output_dim = out_dim
        self.compiled = False

    def make_generator(
        self,
        inp_dim,
        hidden_dims,
        out_dim,
        layer_type=tfkl.Dense,
        act_func=tf.nn.relu,
        expo_mean=100,
        beta=1,
    ):
        layer_list = []
        layer_list.append(tfkl.InputLayer(shape=(inp_dim,)))

        for hd in hidden_dims:
            l_i = layer_type(hd, activation=act_func)
            layer_list.append(l_i)

        last_layer = layer_type(out_dim, activation=act_func)
        layer_list.append(last_layer)

        p_size = tfpl.IndependentNormal.params_size(out_dim)
        layer_list.append(tfkl.Dense(p_size, activation=None))
        prior = tfd.Independent(
            tfd.Normal(loc=tf.zeros(out_dim), scale=1), reinterpreted_batch_ndims=1
        )
        if beta > 0:
            rep_reg = tfpl.KLDivergenceRegularizer(prior, weight=beta)
        else:
            rep_reg = None
        rep_layer = tfpl.IndependentNormal(out_dim, activity_regularizer=rep_reg)

        layer_list.append(rep_layer)
        gen = tfk.Sequential(layer_list)
        return gen

    def make_degenerator(
        self,
        inp_dim,
        hidden_dims,
        out_dim,
        layer_type=tfkl.Dense,
        act_func=tf.nn.relu,
        out_eps=0.01,
    ):
        layer_list = []

        layer_list.append(tfkl.InputLayer(shape=(out_dim,)))
        for hd in hidden_dims:
            l_i = layer_type(hd, activation=act_func)
            layer_list.append(l_i)

        out_layer = layer_type(inp_dim, activation=None)
        layer_list.append(out_layer)

        fixed_std = lambda x: tfd.Normal(x, out_eps)
        layer_list.append(tfpl.DistributionLambda(make_distribution_fn=fixed_std))

        degen = tfk.Sequential(layer_list)
        return degen

    def _compile(self, optimizer=None, loss=da.negloglik):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer, loss)
        self.compiled = True

    def get_representation(self, x):
        return self.generator(x).mean()


class ImageSourceDistrib(object):
    def __init__(
        self,
        datalist,
        set_partition=None,
        position_distr=None,
        use_partition=False,
        partition_vec=None,
        offset=None,
        categorical_variables=None,
        cat_mask=None,
    ):
        self.partition_func = None
        self.data_list = np.array(datalist)
        self.position_distr = position_distr
        self.dim = self.data_list.shape[1]
        self.categorical_variables = categorical_variables

        if position_distr is not None:
            self.dim = self.dim + position_distr.dim
            self.pos_dim = position_distr.dim
        else:
            self.pos_dim = 0
        if self.categorical_variables is None:
            self.not_cat = np.ones(self.dim, dtype=bool)
        else:
            self.not_cat = np.logical_not(self.categorical_variables)
        self.partition = partition_vec
        self.offset = offset
        if use_partition:
            if partition_vec is not None:
                self.partition = u.make_unit_vector(np.array(partition_vec))
                self.offset = offset
                set_partition = ft.partial(
                    da._binary_classification, plane=self.partition, off=self.offset
                )
            elif set_partition is None:
                out = da.generate_partition_functions(sum(self.not_cat), n_funcs=1)
                pfs, vecs, offs = out
                set_partition = pfs[0]
                p_vec = np.zeros(self.dim)
                p_vec[self.not_cat] = vecs[0]
                self.partition = p_vec
                self.offset = offs[0]
            self.partition_func = set_partition
        self.cat_mask = cat_mask

    @property
    def mean(self):
        p_meds = np.percentile(self.data_list, 50, axis=0, interpolation="nearest")
        return p_meds

    def rvs(self, rvs_shape, max_iters=10000):
        rvs = self._candidate_rvs(rvs_shape)
        iters = 0
        while self.partition_func is not None and not np.all(
            self.partition_func(rvs[:, self.not_cat])
        ):
            mask = np.logical_not(self.partition_func(rvs[:, self.not_cat]))
            sample = self._candidate_rvs(rvs_shape)
            rvs[mask] = sample[mask]
            iters += 1
            if iters > max_iters:
                raise IOError(
                    "could not find enough samples in {} "
                    "iterations".format(max_iters)
                )
        return rvs

    def _candidate_rvs(self, n):
        if self.cat_mask is not None:
            data_list = self.data_list[self.cat_mask]
        else:
            data_list = self.data_list
        inds = np.random.choice(data_list.shape[0], n)
        out = data_list[inds]
        if self.position_distr is not None:
            ps = self.position_distr.rvs(n)
            out = np.concatenate((out, ps), axis=1)
        return out

    def make_cat_partition(self, cat_mask=None, part_frac=0.5):
        if cat_mask is None and self.categorical_variables is not None:
            if self.pos_dim > 0:
                ext = -self.pos_dim
            else:
                ext = None
            rel_mask = self.categorical_variables[:ext]
            cat_vals = self.data_list[:, rel_mask]
            u_vals = np.unique(cat_vals)
            part_num = int(np.round(len(u_vals) * part_frac))
            half_vals = np.random.choice(u_vals, part_num, replace=False)
            cat_mask = np.isin(cat_vals, half_vals)
        else:
            cat_mask = cat_mask
        return ImageSourceDistrib(
            self.data_list,
            position_distr=self.position_distr,
            categorical_variables=self.categorical_variables,
            cat_mask=np.squeeze(cat_mask),
        )

    def make_partition(self, set_partition=None, partition_vec=None, offset=None):
        if partition_vec is not None and offset is None:
            offset = 0
        return ImageSourceDistrib(
            self.data_list,
            position_distr=self.position_distr,
            set_partition=set_partition,
            use_partition=True,
            offset=offset,
            partition_vec=partition_vec,
            categorical_variables=self.categorical_variables,
            cat_mask=self.cat_mask,
        )

    def flip_cat_partition(self):
        if self.cat_mask is None:
            print("no cat mask")
            cat_mask = None
        else:
            cat_mask = np.logical_not(self.cat_mask)
        new = ImageSourceDistrib(
            self.data_list,
            cat_mask=cat_mask,
            position_distr=self.position_distr,
            set_partition=self.partition_func,
            use_partition=self.partition_func is not None,
            categorical_variables=self.categorical_variables,
        )
        new.partition = self.partition
        offset = self.offset
        return new

    def flip(self):
        if self.partition_func is not None:
            set_part = lambda x: np.logical_not(self.partition_func(x))
            new = ImageSourceDistrib(
                self.data_list,
                set_partition=set_part,
                position_distr=self.position_distr,
                use_partition=True,
                offset=self.offset,
                categorical_variables=self.categorical_variables,
                cat_mask=self.cat_mask,
            )
            new.partition = -self.partition
            new.offset = self.offset
        else:
            print("no partition to flip")
            new = ImageSourceDistrib(
                self.data_list, categorical_variables=self.categorical_variables
            )
        return new


class ChairSourceDistrib(ImageSourceDistrib):
    def __init__(self, *args, img_identifier="chair_id", **kwargs):
        super().__init__(*args, **kwargs)
        self.img_identifier = img_identifier


class ImageDatasetGenerator(DataGenerator):
    def __init__(
        self,
        data,
        img_params,
        include_position=False,
        position_distr=None,
        max_move=4,
        pre_model=None,
        img_out_label="images",
        categorical_variables=None,
    ):
        if include_position and position_distr is None:
            position_distr = sts.multivariate_normal((0, 0), (0.5 * max_move) ** 2)
        self.include_position = include_position
        self.position_distr = position_distr
        self.data_table = data
        self.img_params = img_params
        self.n_img_params = len(self.img_params)
        self.img_out_label = img_out_label
        self.source_distribution = ImageSourceDistrib(
            data[self.img_params],
            position_distr=position_distr,
            categorical_variables=categorical_variables,
        )
        self.img_size = data[self.img_out_label].iloc[0].shape
        if len(self.img_size) == 1:
            self.img_size = self.img_size[0]

        self.params = self.img_params
        self.data_dict = None
        if include_position:
            self.params = self.params + ["horiz_offset", "vert_offset"]
        else:
            self._make_dict()
        self.output_dim = self.img_size
        self.input_dim = len(self.params)
        self.max_move = max_move
        self.x_uniques = None
        if pre_model is not None:
            self.pm = pre_model
            self.output_dim = self.pm.output_size
        else:
            self.pm = dd.IdentityModel()
            self.output_dim = self.img_size
        if categorical_variables is None:
            self.categorical_variables = np.zeros(len(self.params), dtype=bool)
        else:
            self.categorical_variables = categorical_variables

    def get_category_partitions(self, vec=None, bound=0):
        assert np.any(self.categorical_variables)
        if vec is None:
            vec = u.make_unit_vector(self.categorical_variables.astype(float))
        sd1 = self.source_distribution.make_partition(partition_vec=vec, offset=bound)
        sd2 = sd1.flip()
        return sd1, sd2

    def save(self, path):
        pass

    def _make_dict(self):
        data_dict = {}
        for i, row in self.data_table.iterrows():
            ident = row[self.img_params]
            img = row["images"]
            data_dict[tuple(ident)] = img
        self.data_dict = data_dict

    def fit(*args, **kwargs):
        return tf.keras.callbacks.History()

    def generator(self, x):
        return self.get_representation(x)

    def _move_img(self, img, coords):
        coords[coords > self.max_move] = self.max_move
        coords[coords < -self.max_move] = -self.max_move
        img_dim = np.array(self.img_size[:-1])
        shifts = np.round(img_dim * coords / 2).astype(int)
        s_img = np.roll(img, shifts, axis=(0, 1))
        return s_img

    def ppf(self, perc, dim):
        if dim < self.n_img_params:
            vals = self.data_table[self.img_params[dim]]
            p_val = np.percentile(vals, perc * 100, interpolation="nearest")
        else:
            p_val = sts.norm(0, 0.5 * self.max_move).ppf(perc)
            if p_val > self.max_move:
                p_val = self.max_move
            elif p_val < -self.max_move:
                p_val = -self.max_move
        return p_val

    def get_center(self):
        p_meds = np.percentile(
            self.data_table[self.img_params], 50, axis=0, interpolation="nearest"
        )
        if self.include_position:
            m_meds = self.position_distr.mean
            p_meds = np.concatenate((p_meds, m_meds))
        return p_meds

    def _get_uniques(self):
        if self.x_uniques is None:
            img_params = np.array(self.data_table[self.img_params])
            self.x_uniques = list(
                np.unique(img_params[:, i]) for i in range(self.n_img_params)
            )
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
            new_x[:, self.n_img_params :] = x[:, self.n_img_params :]
            x = new_x
        if same_img and self.img_identifier is not None:
            img_ids = self.data_table[self.img_identifier]
            chosen_id = np.random.choice(img_ids, 1)[0]
            id_mask = img_ids == chosen_id
        # this is a bit slow
        for i, xi in enumerate(x):
            samp = self._params_to_samp(xi, img_params, flat=flat, same_img=same_img)
            out[i] = samp
        return np.stack(out)

    def _params_to_samp(self, xi, img_params, flat=False, same_img=False):
        xi_img = xi[: self.n_img_params]
        if self.data_dict is None:
            mask = np.product(img_params == xi_img, axis=1, dtype=bool)
            if same_img and self.img_identifier is not None:
                mask = mask * id_mask
            s = np.array(self.data_table[self.img_out_label])[mask]
            out_ind = np.random.choice(range(s.shape[0]))
            samp = s[out_ind]
            if self.position_distr is not None:
                samp = self._move_img(samp, xi[self.n_img_params :])
        else:
            samp = self.data_dict[tuple(xi_img)]
        samp = self.pm.get_representation(np.expand_dims(samp, 0))[0]
        if flat:
            samp = np.asarray(samp).flatten()
        return samp

    def representation_dimensionality(
        self,
        source_distribution=None,
        sample_size=10**3,
        participation_ratio=False,
        **pca_args
    ):
        if source_distribution is None and self.source_distribution is not None:
            source_distribution = self.source_distribution
        elif source_distribution is None:
            raise Exception("no source distribution provided")

        samples = source_distribution.rvs(sample_size)
        rep = self.get_representation(samples, flat=True)
        print("rep shape", rep.shape)
        if participation_ratio:
            out = u.participation_ratio(rep)
        else:
            p = skd.PCA(**pca_args)
            p.fit(rep)
            out = (p.explained_variance_ratio_, p.components_)
        return out


def make_split_chair_generators(
    folder,
    norm_params=True,
    img_size=(128, 128),
    include_position=False,
    position_distr=None,
    max_move=4,
    max_load=np.inf,
    filter_edges=None,
    pre_model=None,
    percent_split=0.5,
    id_key="chair_id_num",
    **kwargs
):
    categorical_variables = np.array([True, False, False, False, False])
    data = da.load_chair_images(
        folder,
        img_size=img_size,
        norm_params=True,
        max_load=max_load,
        filter_edges=filter_edges,
        **kwargs
    )

    param_keys = ChairGenerator.default_param_keys
    id_cut = np.percentile(data[id_key], percent_split * 100)
    data1 = data[data[id_key] <= id_cut]
    chairs1 = ImageDatasetGenerator(
        data1,
        param_keys,
        include_position=True,
        position_distr=position_distr,
        max_move=max_move,
        pre_model=pre_model,
        categorical_variables=categorical_variables,
    )

    data2 = data[data[id_key] > id_cut]
    chairs2 = ImageDatasetGenerator(
        data2,
        param_keys,
        include_position=True,
        position_distr=position_distr,
        max_move=max_move,
        pre_model=pre_model,
        categorical_variables=categorical_variables,
    )
    return chairs1, chairs2


class ChairGenerator(ImageDatasetGenerator):
    default_param_keys = ["chair_id_num", "rotation", "pitch"]

    def __init__(
        self,
        folder,
        norm_params=True,
        img_size=(128, 128),
        include_position=False,
        position_distr=None,
        max_move=4,
        max_load=np.inf,
        filter_edges=None,
        pre_model=None,
        param_keys=["chair_id_num", "rotation", "pitch"],
        **kwargs
    ):
        categorical_variables = np.array([True, False, False, False, False])
        data = da.load_chair_images(
            folder,
            img_size=img_size,
            norm_params=True,
            max_load=max_load,
            filter_edges=filter_edges,
            **kwargs
        )
        if pre_model is not None:
            pre_model = dd.PretrainedModel(img_size, pre_model, trainable=False)

        super().__init__(
            data,
            param_keys,
            include_position=include_position,
            position_distr=position_distr,
            categorical_variables=categorical_variables,
            max_move=max_move,
            pre_model=pre_model,
        )


class TwoDShapeGenerator(ImageDatasetGenerator):
    default_pks = ["shape", "scale", "orientation", "x_pos", "y_pos"]

    def __init__(
        self,
        folder,
        img_size=(64, 64),
        norm_params=True,
        max_load=np.inf,
        param_keys=default_pks,
        convert_color=False,
        pre_model=None,
        cached_data_table=None,
        **kwargs
    ):
        self.img_identifier = None
        categorical_variables = np.array([True, False, False, False, False])
        if pre_model is not None:
            pre_model = dd.PretrainedModel(img_size, pre_model, trainable=False)

        if cached_data_table is None:
            data = da.load_2d_shapes(
                folder,
                img_size=img_size,
                convert_color=convert_color,
                norm_params=norm_params,
                max_load=max_load,
                pre_model=pre_model,
            )
        else:
            data = cached_data_table
        super().__init__(
            data, param_keys, categorical_variables=categorical_variables, **kwargs
        )


class ThreeDShapeGenerator(ImageDatasetGenerator):
    default_pks = [
        "floor_hue",
        "wall_hue",
        "object_hue",
        "scale",
        "shape",
        "orientation",
    ]

    def __init__(
        self,
        folder,
        img_size=(64, 64),
        norm_params=True,
        max_load=np.inf,
        param_keys=default_pks,
        pre_model=None,
        **kwargs
    ):
        self.img_identifier = None
        if pre_model is not None:
            pre_model = dd.PretrainedModel(img_size, pre_model, trainable=False)

        data = da.load_3d_shapes(
            folder,
            img_size=img_size,
            norm_params=norm_params,
            max_load=max_load,
            pre_model=pre_model,
        )
        super().__init__(data, param_keys, **kwargs)


class RFDataGenerator(DataGenerator):
    def __init__(
        self,
        inp_dim,
        out_dim,
        source_distribution=None,
        noise=0.001,
        distrib_variance=1,
        setup_distribution=None,
        total_out=True,
        width_scaling=1,
        input_noise=0,
        low_thr=0.01,
        random_widths=False,
        use_random_rfs=False,
    ):
        if source_distribution is None:
            source_distribution = sts.multivariate_normal(
                np.zeros(inp_dim), distrib_variance
            )
        if setup_distribution is None:
            setup_distribution = source_distribution
        try:
            sd_list = setup_distribution.get_indiv_distributions()
        except:
            sd_list = [
                sts.norm(m, np.sqrt(setup_distribution.cov[i, i]))
                for i, m in enumerate(setup_distribution.mean)
            ]
        if total_out:
            out_dim = int(np.round(0.5 * out_dim ** (1 / inp_dim)) * 2)
        self.input_dim = inp_dim
        out = self.make_generator(
            out_dim,
            sd_list,
            input_noise_var=input_noise,
            noise=noise,
            width_scaling=width_scaling,
            random_widths=random_widths,
            use_random_rfs=use_random_rfs,
        )
        self.generator, self.rf_cents, self.rf_wids = out
        self.output_dim = len(self.rf_cents)
        self.low_thr = low_thr
        self.compiled = True
        self.random_rfs = use_random_rfs
        self.source_distribution = source_distribution

    def plot_rfs(
        self,
        ax=None,
        plot_dots=False,
        color=None,
        make_scales=True,
        thin=1,
        x_scale=None,
        y_scale=None,
        **kwargs
    ):
        if ax is None:
            f, ax = plt.subplots(1, 1)
        cps = da.get_circle_pts(100, 2)
        if self.random_rfs:
            rf_wids = self.rf_wids * np.ones((self.output_dim, self.input_dim))
            rf_cents = (self.rf_cents - 0.5) * 2
        else:
            rf_wids = np.sqrt(self.rf_wids)
            rf_cents = self.rf_cents
        for i, rfc in enumerate(rf_cents[::thin]):
            rfw = rf_wids[i]
            l = ax.plot(
                cps[:, 0] * rfw[0] + rfc[0],
                cps[:, 1] * rfw[1] + rfc[1],
                color=color,
                **kwargs
            )
            if plot_dots:
                ax.plot(rfc[0], rfc[1], "o", color=l[0].get_color())
        if make_scales:
            if x_scale is None:
                x_scale = self.source_distribution.cov[0, 0]
            if y_scale is None:
                y_scale = self.source_distribution.cov[1, 1]
            gpl.make_xaxis_scale_bar(ax, x_scale, label="dimension 1")
            gpl.make_yaxis_scale_bar(ax, y_scale, label="dimension 2")
            gpl.clean_plot(ax, 0)

    def make_generator(
        self,
        out_dim,
        source_distribution,
        noise=0.01,
        scale=1,
        baseline=0,
        width_scaling=1,
        input_noise_var=0,
        random_widths=False,
        use_random_rfs=False,
        inp_pwr=100,
    ):
        if use_random_rfs:
            rf_code = spc.Code(
                inp_pwr, out_dim, dims=len(source_distribution), sigma_n=1
            )
            rfs = rf_code.rf
            ms = rf_code.rf_cents
            ws = rf_code.wid
        else:
            out = rfm.get_distribution_gaussian_resp_func(
                out_dim,
                source_distribution,
                scale=scale,
                baseline=baseline,
                wid_scaling=width_scaling,
                random_widths=random_widths,
            )
            rfs, _, ms, ws = out

        noise_distr = sts.multivariate_normal(
            np.zeros(len(ms)), noise, allow_singular=True
        )
        in_distr = sts.multivariate_normal(
            np.zeros(self.input_dim), input_noise_var, allow_singular=True
        )

        def gen(x, input_noise=True, output_noise=True):
            if use_random_rfs:
                x = (x + 1) / 2
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


class GaussianKernel:
    def __init__(self, n_components=100, min_wid=0.01, max_wid=1.5):
        self.n_components = n_components
        self.min_wid = min_wid
        self.max_wid = max_wid

    def fit(self, x):
        self.ss = skp.StandardScaler()
        x = self.ss.fit_transform(x)
        rand = np.random.default_rng()
        cents = rand.standard_normal(size=(self.n_components, x.shape[1]))
        wid = x.shape[1]
        self.cents = np.expand_dims(cents, 0)
        dists = np.sqrt(np.sum(cents**2, axis=1))
        md = np.max(dists)
        self.wids = self.min_wid + self.max_wid ** (dists / md) - 1
        self.dists = dists
        return self

    def transform(self, x):
        x = self.ss.transform(x)
        x = np.expand_dims(x, 1)
        arg = np.sum((x - self.cents) ** 2, axis=2) / (2 * self.wids)
        rs = np.exp(-arg)
        return rs


class LinearDataGenerator(DataGenerator):
    def __init__(self, inp_dim, out_dim, source_distribution=None, distrib_variance=1):
        self.w = sts.norm(0, 1).rvs((out_dim, inp_dim))
        self.input_dim = inp_dim
        self.output_dim = out_dim
        if source_distribution is None:
            source_distribution = sts.multivariate_normal(
                np.zeros(inp_dim), distrib_variance
            )
        self.source_distribution = source_distribution

    def fit(self, *args, **kwargs):
        pass

    def generator(self, x):
        out = np.dot(x, self.w.T)
        return out

    def get_representation(self, x):
        return self.generator(x)


class KernelDataGenerator(DataGenerator):
    def __init__(
        self,
        inp_dim,
        transform_widths,
        out_dim,
        kernel_func=skka.RBFSampler,
        l2_weight=(0, 0.1),
        layer=None,
        distrib_variance=1,
        low_thr=None,
        source_distribution=None,
        noise=0.01,
        **kernel_kwargs
    ):
        self.kernel = kernel_func(n_components=out_dim, **kernel_kwargs)
        self.input_dim = inp_dim
        self.output_dim = out_dim
        self.compiled = False
        self.kernel_init = False
        if source_distribution is None:
            source_distribution = sts.multivariate_normal(
                np.zeros(inp_dim), distrib_variance
            )
        self.source_distribution = source_distribution
        if layer is not None:
            self.layer = dd.SingleLayer(out_dim, layer)
        else:
            self.layer = dd.IdentityModel()
        self.low_thr = low_thr

    def fit(
        self,
        train_x=None,
        train_y=None,
        eval_x=None,
        eval_y=None,
        source_distribution=None,
        epochs=15,
        train_samples=10**5,
        eval_samples=10**3,
        batch_size=32,
        **kwargs
    ):
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
    def __init__(
        self,
        inp_dim,
        transform_widths,
        out_dim,
        shift_base=2,
        layer=None,
        distrib_support=(-0.5, 0.5),
        source_distribution=None,
    ):
        if source_distribution is None:
            source_distribution = da.MultivariateUniform(inp_dim, distrib_support)
        self.source_distribution = source_distribution
        units_per_dim = int(np.floor(out_dim / inp_dim))
        shift_map = np.ones((units_per_dim, inp_dim))
        for i in range(units_per_dim):
            shift_map[i] = shift_base ** (i + 1)
        self.shift_map = shift_map
        self.input_dim = inp_dim
        self.output_dim = units_per_dim * inp_dim

    def generator(self, x):
        x_exp = np.expand_dims(x, 1)
        x_lin = self.shift_map * x_exp
        x_code = x_lin - np.round(x_lin)

        x_out = np.reshape(x_code, (x_code.shape[0], -1))
        return x_out

    def get_representation(self, x):
        return self.generator(x)


def svd_loss_encourage(prod, stim, weight=0.1, eps=1e-5):
    svd_sum = 0
    if len(prod.shape) < 3:
        prod = tf.expand_dims(prod, 2)
    for i in range(prod.shape[1]):
        mat = (tf.transpose(prod[:, i]) @ stim) / stim.shape[0]
        # add noise to avoid degenerate svd
        mat = mat + eps * tf.random.normal(mat.shape)
        s = tf.linalg.svd(mat, compute_uv=False)
        nan_mask = tf.math.logical_not(tf.math.is_nan(s))
        s_use = tf.boolean_mask(s, nan_mask)
        svd_sum = tf.add(tf.reduce_sum(s_use), svd_sum) / prod.shape[1]
    return -svd_sum * weight


def svd_loss_discourage(prod, stim, weight=0.1, eps=1e-8):
    svd_sum = 0
    if len(prod.shape) < 3:
        prod = tf.expand_dims(prod, 2)
    for i in range(prod.shape[1]):
        mat = (tf.transpose(prod[:, i]) @ stim) / stim.shape[0]
        # add noise to avoid degenerate svd
        mat = mat + eps * tf.random.normal(mat.shape)
        s = tf.linalg.svd(mat, compute_uv=False)
        nan_mask = tf.math.logical_not(tf.math.is_nan(s))
        s_use = tf.boolean_mask(s, nan_mask)
        svd_sum = tf.add(tf.reduce_mean(s_use), svd_sum) / prod.shape[1]
    return svd_sum * weight


def _make_svd_rep_funcs(samps, targs, m, func_list):
    """
    compute the pre-gated targets in a <samples> X n_funcs X n_dims array
    """
    rel_stim = maux.get_relevant_dims(samps, m)

    out_arr = np.zeros((samps.shape[0], len(func_list), targs.shape[1]))
    for i, f in enumerate(func_list):
        out_arr[:, i] = np.expand_dims(f(rel_stim), 1) * targs
    return out_arr


class InputAugmentor:
    def __init__(self, fdg, weight=.5, augment=True, n_samples=1000):
        self.fdg = fdg
        self.input_dim = fdg.input_dim
        try:
            _, reps = fdg.get_all_stim()
        except AttributeError:
            _, reps = fdg.sample_reps(n_samples)
        if augment:
            self.output_dim = fdg.output_dim + reps.shape[0]
        else:
            self.output_dim = reps.shape[0]
        self.reference_reps = reps
        self.weight = weight
        self.augment = augment

    def sample_reps(self, *args, **kwargs):
        samps, reps = self.fdg.sample_reps(*args, **kwargs)
        reps = self.augment_reps(reps)
        return samps, reps

    def get_representation(self, *args, **kwargs):
        reps = self.fdg.get_representation(*args, **kwargs)
        aug_rep = self.augment_reps(reps)
        return aug_rep

    def augment_reps(self, reps):
        related_reps = self.augment_func(reps)
        if self.augment:
            o_rep = (1 - self.weight)*reps
            a_rep = self.weight*related_reps
            out = np.concatenate((o_rep, a_rep), axis=1)
        else:
            out = related_reps
        return out


class RelationalAugmentor(InputAugmentor):
    def __init__(self, *args, demean_reps=True, standardize_reps=False, **kwargs):
        super().__init__(*args, **kwargs)
        trs = skp.StandardScaler(with_std=standardize_reps, with_mean=demean_reps)
        self.reference_reps = trs.fit_transform(self.reference_reps)
        self.trs = trs
        
    def augment_func(self, reps):
        reps = self.trs.transform(reps)
        related_reps = reps @ self.reference_reps.T
        return related_reps


class NonexhaustiveMixedDiscreteDataGenerator(DataGenerator):
    def __init__(
        self,
        inp_dim,
        n_vals=2,
        mix_strength=0,
        out_dim=None,
        total_power=1,
        n_units=None,
        mixing_order=None,
        seed=None,
        n_stim_set=10000,
        corr_groups=None,
    ):
        self.input_dim = inp_dim
        self.rng = np.random.default_rng(seed)
        self.n_vals = n_vals
        self.total_power = total_power
        if mixing_order is None:
            mixing_order = self.input_dim
        self.m_units_nz = ss.comb(inp_dim, mixing_order) * n_vals**mixing_order
        categories = [np.arange(n_vals)] * inp_dim
        ohe = skp.OneHotEncoder(categories=categories, sparse_output=False)
        if mixing_order is None:
            mixing_order = inp_dim
        pf = skp.PolynomialFeatures(
            degree=(mixing_order, mixing_order),
            include_bias=False,
            interaction_only=True
        )
        scaler = skp.StandardScaler()
        make_mix_feats = sklpipe.make_pipeline(ohe, pf, scaler)
        fit_stim = self.sample_stim(n_stim_set)
        self.scaler = None
        if mix_strength > 0 and mixing_order > 1:
            self.mix_pipe = make_mix_feats.fit(fit_stim)
            m_len = self.mix_pipe.transform(fit_stim).shape[1]
        else:
            self.mix_pipe = None
            m_len = 0
        p_len = fit_stim.shape[1]
        self.pure_pipe = skp.StandardScaler().fit(fit_stim)
        if n_units is None:
            n_units = p_len + m_len
        self.output_dim = n_units
        self.p_units = p_len
        self.m_units = m_len
        self.trs_mat = u.make_trs_matrix(n_units, p_len + m_len, corr_groups=corr_groups)
        self.mix_strength = mix_strength
        

    def sample_stim(self, n_samps=1000):
        stim = self.rng.choice(np.arange(self.n_vals), size=(n_samps, self.input_dim))
        return stim

    def get_representation(self, stim, noise=0):
        if self.mix_pipe is not None:
            mix_rep = (
                self.mix_pipe.transform(stim)
                * np.sqrt(self.total_power * self.mix_strength)
                / np.sqrt(self.m_units_nz)
            )
        else:
            mix_rep = np.zeros((stim.shape[0], 0))

        pure_rep = (
            self.pure_pipe.transform(stim)
            * np.sqrt(self.total_power * (1 - self.mix_strength))
            / np.sqrt(self.p_units)
        )
        pre_rep = np.concatenate((pure_rep, mix_rep), axis=1)
        rep = pre_rep @ self.trs_mat.T
        if noise > 0:
            rep = rep + rng.normal(0, noise, size=rep.shape)
        return rep

    def sample_reps(self, n_samps, add_noise=False):
        stim = self.sample_stim(n_samps)
        reps = self.get_representation(stim, noise=add_noise)
        return stim, reps


class MixedDiscreteDataGenerator(DataGenerator):
    def __init__(
        self,
        inp_dim,
        n_vals=2,
        mix_strength=0,
        out_dim=None,
        total_power=None,
        n_units=None,
        mixing_order=None,
        sigma=1,
    ):
        if n_units is None:
            if mixing_order is None:
                order = inp_dim
            else:
                order = mixing_order
            n_units = int(inp_dim + ss.comb(inp_dim, order) * n_vals**order)
        code = cc.make_code(
            1 - mix_strength,
            total_power,
            inp_dim,
            n_vals,
            n_units,
            mixing_order=mixing_order,
            noise_cov=sigma ** 2,
        )
        self.code = code
        self.generator = self.code.get_representation
        self.output_dim = n_units
        self.input_dim = inp_dim

    def get_predicted_ccgp(self, order=2):
        sig = self.code.noise_cov
        dls = list(code_i.get_minimum_distance_theor() for code_i in self.code.codes
                   if c.order <= order)
        dns = list(code_i.get_minimum_distance_theor() for code_i in self.code.codes
                   if c.order > order)
        dl = np.sqrt(np.sum(dls**2))
        dn = np.sqrt(np.sum(dns**2))
        f3 = (dl**2 - .5*dl**2)/(sig*np.sqrt(dl**2 + dn**2))
        out = sts.norm(0, 1).cdf(-f3)
        return out

    def get_predicted_classification(self):
        sig = self.code.noise_cov
        ds = list(code_i.get_minimum_distance_theor() for code_i in self.code.codes)
        d = np.sqrt(np.sum(ds**2))
        out = sts.norm(0, 1).cdf(-d/np.sqrt(sig))
        return out

    def sample_stim(self, n_samps=1000):
        return self.code.sample_stim(n_samps)

    def sample_reps(self, n_samps=1000, add_noise=False):
        stim = self.code.sample_stim(n_samps)
        reps = self.code.get_representation(stim, noise=add_noise)
        return stim, reps

    def get_all_stim(self, con_dims=None, **kwargs):
        stim = np.array(self.code.stim)
        if con_dims is not None:
            mask = np.sum(stim[:, con_dims], axis=1) == 1
            stim = stim[mask]
        reps = self.get_representation(stim)
        return stim, reps

    def get_representation(self, stim, **kwargs):
        return self.generator(stim, **kwargs)


class FunctionalDataGenerator(DataGenerator):
    def __init__(
        self,
        inp_dim,
        transform_widths,
        out_dim,
        l2_weight=(0, 0.1),
        source_distribution=None,
        noise=0.01,
        use_pr_reg=False,
        distrib_variance=1,
        auto_encoder=True,
        rescale=False,
        r_eps=.01,
        **kwargs
    ):
        if use_pr_reg:
            reg = dr.L2PRRegularizerInv
        else:
            reg = dr.VarianceRegularizer
        self.generator = self.make_generator(
            inp_dim,
            transform_widths,
            out_dim,
            l2_weight=l2_weight,
            noise=noise,
            reg=reg,
            **kwargs
        )
        self.degenerator = self.make_degenerator(
            inp_dim, transform_widths[::-1], out_dim, **kwargs
        )
        self.model = tfk.Model(
            inputs=self.generator.inputs,
            outputs=self.degenerator(self.generator.outputs[0]),
        )
        self.input_dim = inp_dim
        self.output_dim = out_dim
        self.compiled = False
        self.include_ae = auto_encoder
        if source_distribution is None:
            source_distribution = sts.multivariate_normal(
                np.zeros(inp_dim), distrib_variance
            )
        self.source_distribution = source_distribution
        self.rescale = rescale
        self.rescale_weights = np.ones((1, out_dim))
        self.r_eps = r_eps

    def fit(self, *args, **kwargs):
        out = super().fit(*args, **kwargs)
        if self.rescale:
            self.rescale_weights = np.max(
                self.sample_reps()[1], axis=0, keepdims=True
            )
            self.rescale_weights[self.rescale_weights < self.r_eps] = self.r_eps
        return out

    def make_generator(
        self,
        inp_dim,
        hidden_dims,
        out_dim,
        layer_type=tfkl.Dense,
        act_func=tf.nn.relu,
        expo_mean=5,
        l2_weight=1,
        noise=None,
        reg=dr.VarianceRegularizer,
        kernel_init=None,
    ):
        if kernel_init is None:
            kernel_init = tfk.initializers.GlorotUniform
        layer_list = []
        layer_list.append(tfkl.InputLayer(shape=(inp_dim,)))

        regularizer = reg(l2_weight)

        for hd in hidden_dims:
            l_i = layer_type(
                hd,
                activation=act_func,
                activity_regularizer=regularizer,
                kernel_initializer=kernel_init(),
            )
            layer_list.append(l_i)
            if noise is not None:
                layer_list.append(tfkl.GaussianNoise(noise))

        last_layer = layer_type(
            out_dim,
            activation=act_func,
            activity_regularizer=regularizer,
            kernel_initializer=kernel_init(),
        )
        layer_list.append(last_layer)

        gen = tfk.Sequential(layer_list)
        return gen

    def make_degenerator(
        self,
        inp_dim,
        hidden_dims,
        out_dim,
        layer_type=tfkl.Dense,
        act_func=tf.nn.relu,
        kernel_init=None,
    ):
        if kernel_init is None:
            kernel_init = tfk.initializers.GlorotUniform
        layer_list = []

        layer_list.append(tfkl.InputLayer(shape=(out_dim,)))
        for hd in hidden_dims:
            l_i = layer_type(hd, activation=act_func, kernel_initializer=kernel_init())
            layer_list.append(l_i)

        out_layer = layer_type(inp_dim, activation=None)
        layer_list.append(out_layer)

        degen = tfk.Sequential(layer_list)
        return degen

    def save(self, path):
        tf_entries = ("generator", "degenerator", "model")
        self._save_wtf(path, tf_entries)

    @classmethod
    def load(cls, path):
        dummy = FunctionalDataGenerator(0, (10,), 0)
        return cls._load_model(dummy, path)

    def _compile(self, optimizer=None, loss=tf.losses.MeanSquaredError()):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        if not self.include_ae:
            loss = lambda x, y: 0
        self.model.compile(optimizer, loss)
        self.compiled = True

    def get_representation(self, x):
        out = self.generator(x) / self.rescale_weights
        return out


class SVDFunctionalDataGenerator(FunctionalDataGenerator):
    def __init__(
        self,
        inp_dim,
        transform_widths,
        out_dim,
        target_model,
        l2_weight=(0, 0.1),
        source_distribution=None,
        noise=0.01,
        use_pr_reg=True,
        distrib_variance=1,
        auto_encoder=True,
        **kwargs
    ):
        self.target_model = target_model
        self.branch_names = (
            "rep_branch_encourage",
            "rep_branch_discourage",
            "autoencoder_branch",
        )
        if use_pr_reg:
            reg = dr.L2PRRegularizerInv
        else:
            reg = dr.VarianceRegularizer
        out = self.make_model(
            inp_dim,
            transform_widths,
            out_dim,
            l2_weight=l2_weight,
            branch_names=self.branch_names,
            noise=noise,
            reg=reg,
            **kwargs
        )
        self.model, self.generator, self.auto_model = out

        self.input_dim = inp_dim
        self.output_dim = out_dim
        self.compiled = False
        self.include_ae = auto_encoder
        if source_distribution is None:
            source_distribution = sts.multivariate_normal(
                np.zeros(inp_dim), distrib_variance
            )
        self.source_distribution = source_distribution

    def _compile(self, optimizer=None, loss_dict=None, loss_ratio=10):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        if loss_dict is None:
            loss_dict = {
                self.branch_names[2]: tf.losses.MeanSquaredError(),
                self.branch_names[0]: svd_loss_encourage,
                self.branch_names[1]: svd_loss_discourage,
            }
        if not self.include_ae:
            lw_dict = {
                self.branch_names[0]: 1,
                self.branch_names[1]: 1,
                self.branch_names[2]: 0,
            }
        else:
            lw_dict = {
                self.branch_names[0]: 1,
                self.branch_names[1]: 1,
                self.branch_names[2]: loss_ratio,
            }
        if not self.include_ae:
            loss_dict[self.branch_names[2]] = lambda x, y: 0

        self.model.compile(optimizer, loss=loss_dict, loss_weights=lw_dict)
        self.compiled = True

    def make_model(
        self,
        input_shape,
        layer_shapes,
        encoded_size,
        act_func=tf.nn.relu,
        regularizer_weight=0.1,
        layer_type=tfkl.Dense,
        branch_names=(
            "rep_branch_encourage",
            "rep_branch_discourage",
            "autoencoder_branch",
        ),
        reg=dr.L2PRRegularizerInv,
        l2_weight=0.1,
        dropout_rate=0,
        noise=0,
        weight_reg_weight=0,
        weight_reg_type=tfk.regularizers.l2,
        readout_bias_reg_str=0,
        readout_bias_reg_type=tfk.regularizers.l2,
        kernel_init=None,
        full_reg=True,
        **layer_params
    ):
        if kernel_init is None:
            kernel_init = tfk.initializers.GlorotUniform()

        act_reg = reg(l2_weight)

        inputs = tfk.Input(shape=input_shape)
        x = inputs
        if weight_reg_weight > 0:
            kernel_reg = weight_reg_type(weight_reg_weight)
        else:
            kernel_reg = None
        if readout_bias_reg_str > 0:
            readout_bias_reg = readout_bias_reg_type(readout_bias_reg_str)
        else:
            readout_bias_reg = None

        # representation layer
        if full_reg:
            layer_act_reg = act_reg
        else:
            layer_act_reg = None

        for lp in layer_shapes:
            x = layer_type(
                *lp,
                activation=act_func,
                kernel_regularizer=kernel_reg,
                activity_regularizer=layer_act_reg,
                **layer_params
            )(x)

        if dropout_rate > 0:
            x = tfkl.Dropout(dropout_rate)(x)

        rep = tfkl.Dense(
            encoded_size,
            activation=None,
            activity_regularizer=act_reg,
            kernel_regularizer=kernel_reg,
            bias_regularizer=readout_bias_reg,
        )(x)
        if noise > 0:
            rep = tfkl.GaussianNoise(noise)(rep)
        rep_model = tfk.Model(inputs=inputs, outputs=rep, name=branch_names[0])
        rep_model_discourage = tfk.Model(
            inputs=inputs, outputs=rep, name=branch_names[1]
        )
        rep_inp = tfk.Input(shape=encoded_size)

        # decoder branch
        z = rep_inp
        for lp in layer_shapes[::-1]:
            z = tfkl.Dense(*lp, activation=act_func, **layer_params)(z)

        autoenc_branch = layer_type(input_shape, activation=act_func, **layer_params)(z)
        autoenc_model = tfk.Model(
            inputs=rep_inp, outputs=autoenc_branch, name=branch_names[2]
        )

        outs = [
            rep_model(inputs),
            rep_model_discourage(inputs),
            autoenc_model(rep_model(inputs)),
        ]
        full_model = tfk.Model(inputs=inputs, outputs=outs)

        return full_model, rep_model, autoenc_model

    def fit(
        self,
        encourage_funcs=(),
        discourage_funcs=(),
        train_x=None,
        train_y=None,
        eval_x=None,
        eval_y=None,
        source_distribution=None,
        epochs=15,
        train_samples=10**5,
        eval_samples=10**3,
        batch_size=200,
        norm_targs=True,
        **kwargs
    ):
        if not self.compiled:
            self._compile()

        _, train_x, targs = self.target_model.get_x_true(n_train=train_samples)
        if norm_targs:
            targs = skp.StandardScaler().fit_transform(targs)

        encourage_reps = _make_svd_rep_funcs(
            train_x, targs, self.target_model, encourage_funcs
        )
        discourage_reps = _make_svd_rep_funcs(
            train_x, targs, self.target_model, discourage_funcs
        )

        train_y_dict = {
            self.branch_names[0]: encourage_reps,
            self.branch_names[1]: discourage_reps,
            self.branch_names[2]: train_x,
        }
        out = self.model.fit(
            train_x, train_y_dict, epochs=epochs, batch_size=batch_size, **kwargs
        )
        return out
