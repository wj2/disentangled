import numpy as np
import scipy.stats as sts

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

import sklearn.decomposition as skd
import sklearn.svm as skc

import general.utility as u
import general.rf_models as rfm

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

float_cast_center = lambda x: tf.cast(x, tf.float32) - .5
negloglik = lambda x, rv_x: -rv_x.log_prob(x)
dummy = lambda x, rv_x: tf.zeros_like(x)

class DataGenerator(object):

    def fit(self, train_x=None, train_y=None, eval_x=None, eval_y=None,
            source_distribution=None, epochs=15, train_samples=10**5,
            eval_samples=10**3, batch_size=32):
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
                             epochs=epochs, batch_size=batch_size)
        return out

    def representation_dimensionality(self, source_distribution=None,
                                      sample_size=10**4, **pca_args):
        if source_distribution is None and self.source_distribution is not None:
            source_distribution = self.source_distribution
        elif source_distribution is None:
            raise Exception('no source distribution provided')

        samples = source_distribution.rvs(sample_size)
        rep = self.get_representation(samples)
        p = skd.PCA(**pca_args)
        p.fit(rep)
        out = (p.explained_variance_ratio_, p.components_)
        return out

    def sample_reps(self, source_distribution=None, sample_size=10**4):
        if source_distribution is None and self.source_distribution is not None:
            source_distribution = self.source_distribution
        elif source_distribution is None:
            raise Exception('no source distribution provided')
        inp_samps = source_distribution.rvs(sample_size)
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
    
    def _compile(self, optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                 loss=negloglik):
        self.model.compile(optimizer, loss)
        self.compiled = True

    def get_representation(self, x):
        return self.generator(x).mean()
        
class VarianceRegularizer(tf.keras.regularizers.Regularizer):

    def __init__(self, weights, **kwargs):
        if kwargs:
            raise TypeError('Argument(s) not recognized: %s' % (kwargs,))

        self.l2_weight = float(weights[0])
        self.var_weight = float(weights[1])

    def __call__(self, x):
        l2_r = tf.math.reduce_sum(tf.math.square(x))
        var_r = tf.math.reduce_variance(tf.math.reduce_mean(x**2, axis=0))
        return self.l2_weight*l2_r + self.var_weight*var_r

class RFDataGenerator(DataGenerator):

    def __init__(self, inp_dim, out_dim, source_distribution=None, noise=.01,
                 distrib_variance=1, setup_distribution=None):
        if source_distribution is None:
            source_distribution = sts.multivariate_normal(np.zeros(inp_dim),
                                                          distrib_variance)
        if setup_distribution is None:
            setup_distribution = source_distribution

        sd_list = [sts.norm(m, np.sqrt(setup_distribution.cov[i, i]))
                   for i, m in enumerate(setup_distribution.mean)]
        out = self.make_generator(out_dim, sd_list,
                                  noise=noise)
        self.generator, self.rf_cents, self.rf_wids = out
        self.input_dim = inp_dim
        self.output_dim = len(self.rf_cents)
        self.compiled = True
        self.source_distribution = source_distribution

    def make_generator(self, out_dim, source_distribution, noise=.01,
                       scale=1, baseline=0):
        out = rfm.get_distribution_gaussian_resp_func(out_dim,
                                                      source_distribution,
                                                      scale=scale,
                                                      baseline=baseline)
        rfs, _, ms, ws = out
        noise_distr = sts.multivariate_normal(np.zeros(len(ms)), noise)
        gen = lambda x: rfs(x) + noise_distr.rvs(x.shape[0])
        return gen, ms, ws

    def get_representation(self, x):
        return self.generator(x)

    def fit(*args, **kwargs):
        return tf.keras.callbacks.History()
    
class FunctionalDataGenerator(DataGenerator):

    def __init__(self, inp_dim, transform_widths, out_dim, l2_weight=.01,
                 source_distribution=None, noise=.01):
        
        self.generator = self.make_generator(inp_dim, transform_widths,
                                             out_dim, l2_weight=l2_weight,
                                             noise=noise)
        self.degenerator = self.make_degenerator(inp_dim,
                                                 transform_widths[::-1],
                                                 out_dim)
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

        regularizer = VarianceRegularizer(l2_weight)

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
    
    def _compile(self, optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                 loss=tf.losses.MeanSquaredError()):
        self.model.compile(optimizer, loss)
        self.compiled = True
        
    def get_representation(self, x):
        return self.generator(x)

class SupervisedDisentangler(object):

    def __init__(self, input_shape, layer_shapes, encoded_size,
                 act_func=tf.nn.relu, **layer_params):
        enc = self.make_encoder(input_shape, layer_shapes, encoded_size,
                                act_func=act_func, **layer_params)
        self.encoder = enc
        self.model = tfk.Model(inputs=self.encoder.inputs,
                               outputs=self.encoder.outputs[0])
        self.input_shape = input_shape
        self.encoded_size = encoded_size
        self.compiled = False

    def make_encoder(self, input_shape, layer_shapes, encoded_size,
                     act_func=tf.nn.relu, layer_type=tfkl.Dense,
                     **layer_params):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=input_shape))
            
        for lp in layer_shapes:
            l_i = layer_type(*lp, activation=act_func, **layer_params)
            layer_list.append(l_i)

        layer_list.append(tfkl.Dense(encoded_size, activation=None))

        enc = tfk.Sequential(layer_list)
        return enc

    def _compile(self, optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                 loss=tf.losses.MeanSquaredError()):
        self.model.compile(optimizer, loss)
        self.compiled = True

    def fit_sets(self, train_set, eval_set=None, **kwargs):
        train_y, train_x = train_set
        if eval_set is not None:
            eval_y, eval_x = eval_set
        else:
            eval_y, eval_x = None, None
        return self.fit(train_x, train_y, eval_x=eval_x, eval_y=eval_y, **kwargs)
        
    def fit(self, train_x, train_y, eval_x=None, eval_y=None, epochs=15,
            data_generator=None, batch_size=32):
        if not self.compiled:
            self._compile()

        if train_y.shape[1] < self.encoded_size:
            train_y = pad_zeros(train_y, self.encoded_size)
        if eval_y is not None and eval_y.shape[1] < self.encoded_size:
            eval_y = pad_zeros(eval_y, self.encoded_size)
            
        if eval_x is not None and eval_y is not None:
            eval_set = (eval_x, eval_y)
        else:
            eval_set = None

        out = self.model.fit(x=train_x, y=train_y, epochs=epochs,
                           validation_data=eval_set, batch_size=batch_size)
        return out   

    def get_representation(self, samples):
        rep = self.encoder(samples)
        return rep

def pad_zeros(x, dim):
    to_pad = dim - x.shape[1]
    add = np.zeros((x.shape[0], to_pad))
    x_new = np.concatenate((x, add), axis=1)
    return x_new
    
class StandardAE(object):

    def __init__(self, input_shape, layer_shapes, encoded_size,
                 act_func=tf.nn.relu, **layer_params):
        enc = self.make_encoder(input_shape, layer_shapes, encoded_size,
                                act_func=act_func, **layer_params)
        self.encoder = enc
        dec = self.make_decoder(encoded_size, layer_shapes, input_shape,
                                act_func=act_func, **layer_params)
        self.decoder = dec
        self.model = tfk.Model(inputs=self.encoder.inputs,
                               outputs=self.decoder(self.encoder.outputs[0]))
        self.input_shape = input_shape
        self.encoded_size = encoded_size
        self.compiled = False

    def make_encoder(self, input_shape, layer_shapes, encoded_size,
                     act_func=tf.nn.relu, layer_type=tfkl.Dense,
                     **layer_params):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=input_shape))
            
        for lp in layer_shapes:
            l_i = layer_type(*lp, activation=act_func, **layer_params)
            layer_list.append(l_i)

        layer_list.append(tfkl.Dense(encoded_size, activation=None))

        enc = tfk.Sequential(layer_list)
        return enc

    def make_decoder(self, encoded_size, layer_shapes, input_shape,
                     act_func=tf.nn.relu, layer_type=tfkl.Dense,
                     **layer_params):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=[encoded_size]))

        for lp in layer_shapes:
            l_i = layer_type(*lp, activation=act_func, **layer_params)
            layer_list.append(l_i)

        layer_list.append(layer_type(input_shape, activation=act_func))
        dec = tfk.Sequential(layer_list)
        return dec
    
    def _compile(self, optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                 loss=tf.losses.MeanSquaredError()):
        self.model.compile(optimizer, loss)
        self.compiled = True

    def fit_sets(self, train_set, eval_set=None, **kwargs):
        train_x = train_set[1]
        if eval_set is not None:
            eval_x = eval_set[1]
        else:
            eval_x = None
        return self.fit(train_x, eval_x=eval_x, **kwargs)
        
    def fit(self, train_x, eval_x=None, epochs=15,
            data_generator=None, batch_size=32):
        if not self.compiled:
            self._compile()
            
        if eval_x is not None:
            eval_set = (eval_x, eval_x)
        else:
            eval_set = None

        out = self.model.fit(x=train_x, y=train_x, epochs=epochs,
                           validation_data=eval_set, batch_size=batch_size)
        return out   

    def get_representation(self, samples):
        rep = self.encoder(samples)
        return rep
    
    
class BetaVAE(object):

    def __init__(self, input_shape, layer_shapes, encoded_size,
                 act_func=tf.nn.relu, beta=1, **layer_params):
        enc, prior = self.make_encoder(input_shape, layer_shapes, encoded_size,
                                       act_func=act_func, beta=beta,
                                       **layer_params)
        self.encoder = enc
        self.prior = prior
        self.decoder = self.make_decoder(input_shape, layer_shapes[::-1],
                                         encoded_size, act_func=act_func,
                                         **layer_params)
        self.vae = tfk.Model(inputs=self.encoder.inputs,
                             outputs=self.decoder(self.encoder.outputs[0]))
        self.input_shape = input_shape
        self.encoded_size = encoded_size
        self.compiled = False

    def make_encoder(self, input_shape, layer_shapes, encoded_size,
                     act_func=tf.nn.relu, strides=1,
                     transform_layer=None, layer_type=tfkl.Dense,
                     conv=False, beta=1, **layer_params):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=input_shape))
        if transform_layer is not None:
            layer_list.append(tfkl.Lambda(transform_layer))
            
        for lp in layer_shapes:
            l_i = layer_type(*lp, activation=act_func, **layer_params)
            layer_list.append(l_i)

        if conv:
            layer_list.append(tfkl.Flatten())
            p_size = tfpl.MultivariateNormalTriL.params_size(encoded_size)
        else:
            p_size = tfpl.IndependentNormal.params_size(encoded_size)
            
        layer_list.append(tfkl.Dense(p_size, activation=None))

        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(encoded_size), scale=1),
                                reinterpreted_batch_ndims=1)
        if beta > 0:
            rep_reg = tfpl.KLDivergenceRegularizer(prior, weight=beta)
        else:
            rep_reg = None
        if conv:
            rep_layer = tfpl.MultivariateNormTriL(encoded_size,
                                                  activity_regularizer=rep_reg)
        else:
            rep_layer = tfpl.IndependentNormal(encoded_size,
                                               activity_regularizer=rep_reg)
        layer_list.append(rep_layer)

        enc = tfk.Sequential(layer_list)
        return enc, prior

    def make_decoder(self, input_shape, layer_shapes, encoded_size,
                     act_func=tf.nn.relu, strides=1,
                     transform_layer=None, layer_type=tfkl.Dense,
                     conv=False, out_eps=.01, **layer_params):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=[encoded_size]))

        for lp in layer_shapes:
            l_i = layer_type(*lp, activation=act_func, **layer_params)
            layer_list.append(l_i)

        if conv:
            layer_list.append(tfkl.Conv2D(1, fs, strides=strides,
                                          padding='same', activation=act_func))
            layer_list.append(tfkl.Flatten())
        else:
            layer_list.append(layer_type(input_shape, activation=act_func))

        fixed_std = lambda x: tfd.Normal(x, out_eps)
        layer_list.append(tfpl.DistributionLambda(
            make_distribution_fn=fixed_std))
        dec = tfk.Sequential(layer_list)
        return dec

    def _compile(self, optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                 loss=negloglik):
        self.vae.compile(optimizer, negloglik)
        self.compiled = True
        
    def fit(self, train_x, train_y=None, eval_x=None, eval_y=None, epochs=15,
            data_generator=None, batch_size=32):
        if not self.compiled:
            self._compile()

        if train_y is None:
            train_y = train_x
            
        if eval_x is not None and eval_y is None:
            eval_y = eval_x
            eval_set = (eval_x, eval_y)
        else:
            eval_set = None

        if data_generator is not None:
            train_x = data_generator.gen
            train_y = None
            eval_data = data_generator.rvs(10*5)
            eval_set = (eval_data, eval_data)
            
        out = self.vae.fit(x=train_x, y=train_y, epochs=epochs,
                           validation_data=eval_set, batch_size=batch_size)
        return out

    def sample_latents(self, sample_size=10):
        samps = self.prior.sample(sample_size)
        outs = self.decoder(samps).mean()
        return outs

    def get_representation(self, samples):
        rep = self.encoder(samples).sample()
        return rep

class InputGenerator(object):

    def __init__(self, distribution, proc_func):
        self.source = distribution
        self.proc = proc_func

    def gen(self):
        while True:
            yield self.proc(self.source.rvs())

    def rvs(self, n):
        out = self.proc(self.source.rvs(n))
        return out
    
    
def classifier_generalization(gen, vae, train_func, train_distrib=None,
                              test_distrib=None, test_func=None,
                              n_train_samples=2*10**4, n_test_samples=10**4,
                              classifier=skc.SVC, kernel='linear',
                              **classifier_params):
    if train_distrib is None:
        train_distrib = gen.source_distribution
    if test_distrib is None:
        test_distrib = train_distrib
    if test_func is None:
        test_func = train_func

    train_samples = train_distrib.rvs(n_train_samples)
    train_labels = train_func(train_samples)
    train_rep = vae.get_representation(gen.generator(train_samples))
    c = classifier(kernel=kernel, **classifier_params)
    c.fit(train_rep, train_labels)

    test_samples = test_distrib.rvs(n_test_samples)
    test_labels = test_func(test_samples)
    test_rep = vae.get_representation(gen.generator(test_samples))
    score = c.score(test_rep, test_labels)
    chance = np.max(np.histogram(test_labels.astype(int))[0]/n_test_samples)
    return score, chance

def train_multiple_bvae(dg, betas, layer_spec, n_reps=10, batch_size=32,
                        n_train_samps=10**6, epochs=5, hide_print=False,
                        input_dim=None):
    training_history = np.zeros((len(betas), n_reps), dtype=object)
    models = np.zeros_like(training_history, dtype=object)
    if input_dim is None:
        input_dim = dg.input_dim
    for i, beta in enumerate(betas):
        for j in range(n_reps):
            inp_set, train_set = dg.sample_reps(sample_size=n_train_samps)
            inp_eval_set, eval_set = dg.sample_reps()
            bvae = BetaVAE(dg.output_dim, layer_spec, input_dim, beta=beta)
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

def train_multiple_models(dg, model_kinds, layer_spec, n_reps=10, batch_size=32,
                          n_train_samps=10**6, epochs=5, hide_print=False,
                          input_dim=None):
    training_history = np.zeros((len(model_kinds), n_reps), dtype=object)
    models = np.zeros_like(training_history, dtype=object)
    if input_dim is None:
        input_dim = dg.input_dim
    for i, mk in enumerate(model_kinds):
        for j in range(n_reps):
            train_set = dg.sample_reps(sample_size=n_train_samps)
            eval_set = dg.sample_reps()
            m = mk(dg.output_dim, layer_spec, input_dim)
            if hide_print:
                with u.HiddenPrints():
                    th = m.fit_sets(train_set, eval_set=eval_set, epochs=epochs,
                                    batch_size=batch_size)
            else:
                th = m.fit_sets(train_set, eval_set=eval_set, epochs=epochs,
                                batch_size=batch_size)
            training_history[i, j] = th
            models[i, j] = m
    return models, training_history


def evaluate_multiple_models(dg, models, train_func, test_distributions,
                             **classifier_args):
    performance = np.zeros(models.shape + (len(test_distributions),))
    chance = np.zeros_like(performance)
    for i in range(models.shape[0]):
        for j in range(models.shape[1]):
            bvae = models[i, j]
            for k, td in enumerate(test_distributions):
                out = classifier_generalization(dg, bvae, train_func,
                                                test_distrib=td,
                                                **classifier_args)
                performance[i, j, k] = out[0]
                chance[i, j, k]= out[1]
    return performance, chance

def train_and_evaluate_models(dg, betas, layer_spec, train_func,
                              test_distributions, n_reps=10, hide_print=False,
                              n_train_samps=10**6, epochs=5, input_dim=None,
                              models=None, batch_size=32, **classifier_args):
    if models is None:
        models, th = train_multiple_bvae(dg, betas, layer_spec, n_reps=n_reps,
                                         n_train_samps=n_train_samps,
                                         batch_size=batch_size, epochs=epochs,
                                         hide_print=hide_print,
                                         input_dim=input_dim)
        out2 = (models, th)
    else:
        out2 = (models, None)
    out = evaluate_multiple_models(dg, models, train_func, test_distributions,
                                   **classifier_args)
    return out, out2 
                              
def plot_diagnostics(dg, model, rs, n_arcs, ax=None, n=1000, dim_red=True,
                     n_dim_red=10*4, **pca_args):
    if ax is None:
        f, ax = plt.subplots(1, 1)

    angs = np.linspace(0, 2*np.pi, n)
    pts = np.stack((np.cos(angs), np.sin(angs),) +
                   (np.zeros_like(angs),)*(dg.input_dim - 2), axis=1)
    
    if dim_red:
        distrib_pts = dg.source_distribution.rvs(n_dim_red)
        distrib_reps = dg.generator(distrib_pts)
        mod_distrib_reps = model.get_representation(distrib_reps)
        p = skd.PCA(**pca_args)
        p.fit(mod_distrib_reps)
    
    for r in rs:
        s_reps = dg.generator(r*pts)
        mod_reps = model.get_representation(s_reps)
        if dim_red:
            mod_reps = p.transform(mod_reps)
        ax.plot(mod_reps[:, 0], mod_reps[:, 1], 'o')

    if n_arcs > 0:
        skips = int(np.round(n/n_arcs))
        sub_pts = rs[-1]*pts[::skips]
        y = np.expand_dims(np.linspace(0, 1, n), 1)
        for sp in sub_pts:
            sp = np.expand_dims(sp, 0)
            s_reps = dg.generator(sp*y)
            mod_reps = model.get_representation(s_reps)
            if dim_red:
                mod_reps = p.transform(mod_reps)
            ax.plot(mod_reps[:, 0], mod_reps[:, 1], 'o')
    return ax
