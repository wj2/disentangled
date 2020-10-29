import numpy as np
import general.utility as u

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

import sklearn.decomposition as skd
import sklearn.svm as skc

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

float_cast_center = lambda x: tf.cast(x, tf.float32) - .5
negloglik = lambda x, rv_x: -rv_x.log_prob(x)
dummy = lambda x, rv_x: tf.zeros_like(x)

class VariationalDataGenerator(object):

    def __init__(self, inp_dim, transform_widths, out_dim,
                 out_regularizer=None):
        
        self.generator = self.make_generator(inp_dim, transform_widths,
                                             out_dim)
        self.degenerator = self.make_degenerator(inp_dim,
                                                 transform_widths[::-1],
                                                 out_dim)
        self.model = tfk.Model(inputs=self.generator.inputs,
                               outputs=self.degenerator(self.generator.outputs[0]))
        self.input_dim = inp_dim
        self.output_dim = out_dim
        self.compiled = False

    def make_generator(self, inp_dim, hidden_dims, out_dim,
                       layer_type=tfkl.Dense, act_func=tf.nn.leaky_relu,
                       expo_mean=5, beta=1):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=inp_dim))

        for hd in hidden_dims:
            l_i = layer_type(hd, activation=act_func)
            layer_list.append(l_i)

        last_layer = layer_type(out_dim, activation=act_func)
        layer_list.append(last_layer)
        
        prior = tfd.Independent(tfd.Poisson(rate=tf.ones(out_dim)*expo_mean),
                                reinterpreted_batch_ndims=1)
        if beta > 0:
            rep_reg = tfpl.KLDivergenceRegularizer(prior, weight=beta)
        else:
            rep_reg = None
        rep_layer = tfpl.IndependentPoisson(out_dim,
                                            activity_regularizer=rep_reg)
        layer_list.append(rep_layer)
        gen = tfk.Sequential(layer_list)
        return gen

    def make_degenerator(self, inp_dim, hidden_dims, out_dim,
                         layer_type=tfkl.Dense, act_func=tf.nn.leaky_relu):
        layer_list = []
        
        layer_list.append(tfkl.InputLayer(input_shape=out_dim))
        for hd in hidden_dims:
            l_i = layer_type(hd, activation=act_func)
            layer_list.append(l_i)

        ll_sizes = tfpl.IndependentNormal.params_size(inp_dim)

        out_layer = layer_type(ll_sizes, activation=None)
        layer_list.append(out_layer)

        rep_layer = tfpl.IndependentNormal(inp_dim)
        layer_list.append(rep_layer)

        degen = tfk.Sequential(layer_list)
        return degen
    
    def _compile(self, optimizer=tf.optimizers.Adam(learning_rate=1e-3),
                 loss=negloglik):
        self.model.compile(optimizer, loss)
        self.compiled = True
        
    def fit(self, train_x, train_y=None, eval_x=None, eval_y=None,
            epochs=15):
        if not self.compiled:
            self._compile()

        if train_y is None:
            train_y = train_x
            
        if eval_x is not None and eval_y is None:
            eval_y = eval_x

        out = self.model.fit(train_x, train_y, validation_data=(eval_x, eval_y),
                             epochs=epochs)
        return out

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
    
class FunctionalDataGenerator(object):

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

        # regularizer = tf.keras.regularizers.L2(l2_weight)
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

        # if noise is not None:
        #     layer_list.append(tfkl.GaussianNoise(noise))

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
        
    def fit(self, train_x=None, train_y=None, eval_x=None, eval_y=None,
            source_distribution=None, epochs=15, train_samples=2*10**4,
            eval_samples=10**3):
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
                             epochs=epochs)
        return out

    def representation_dimensionality(self, source_distribution=None,
                                      sample_size=10**4, **pca_args):
        if source_distribution is None and self.source_distribution is not None:
            source_distribution = self.source_distribution
        elif source_distribution is None:
            raise Exception('no source distribution provided')

        samples = source_distribution.rvs(sample_size)
        rep = self.generator(samples)
        p = skd.PCA(**pca_args)
        p.fit(rep)
        out = (p.explained_variance_ratio_, p.components_)
        return out

    def sample_reps(self, source_distribution=None, sample_size=10**4):
        if source_distribution is not None:
            self.source_distribution = source_distribution
        inp_samps = self.source_distribution.rvs(sample_size)
        rep_samps = self.generator(inp_samps)
        return inp_samps, rep_samps
    
class BetaVAE(object):

    def __init__(self, input_shape, layer_shapes, encoded_size,
                 act_func=tf.nn.relu, beta=1, **layer_params):
        enc, prior = self.make_encoder(input_shape, layer_shapes, encoded_size,
                                       act_func=act_func, beta=beta,
                                       **layer_params)
        self.encoder = enc
        self.prior = prior
        self.decoder = self.make_decoder(input_shape, layer_shapes, encoded_size,
                                         act_func=act_func, **layer_params)
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
        
    def fit(self, train_x, train_y=None, eval_x=None, eval_y=None, epochs=15):
        if not self.compiled:
            self._compile()

        if train_y is None:
            train_y = train_x
            
        if eval_x is not None and eval_y is None:
            eval_y = eval_x
            eval_set = (eval_x, eval_y)
        else:
            eval_set = None
            
        out = self.vae.fit(train_x, train_y, epochs=epochs,
                           validation_data=eval_set)
        return out

    def sample_latents(self, sample_size=10):
        samps = self.prior.sample(sample_size)
        outs = self.decoder(samps).mean()
        return outs

def classifier_generalization(gen, vae, train_func, train_distrib=None,
                              test_distrib=None, test_func=None,
                              n_train_samples=10**4, n_test_samples=10**3,
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
    train_rep = vae.encoder(gen.generator(train_samples)).sample()
    c = classifier(kernel=kernel, **classifier_params)
    c.fit(train_rep, train_labels)

    test_samples = test_distrib.rvs(n_test_samples)
    test_labels = test_func(test_samples)
    test_rep = vae.encoder(gen.generator(test_samples)).sample()
    score = c.score(test_rep, test_labels)
    chance = np.max(np.histogram(test_labels.astype(int))[0]/n_test_samples)
    return score, chance

def train_multiple_bvae(dg, betas, layer_spec, n_reps=10,
                        n_train_samps=10**6, epochs=5, hide_print=True):
    training_history = np.zeros((len(betas), n_reps), dtype=object)
    models = np.zeros_like(training_history, dtype=object)
    for i, beta in enumerate(betas):
        for j in range(n_reps):
            inp_set, train_set = dg.sample_reps(sample_size=n_train_samps)
            inp_eval_set, eval_set = dg.sample_reps()

            bvae = BetaVAE(dg.output_dim, layer_spec, dg.input_dim, beta=beta)
            if hide_print:
                with u.HiddenPrints():
                    th = bvae.fit(train_set, eval_x=eval_set, epochs=epochs)
            else:
                th = bvae.fit(train_set, eval_x=eval_set, epochs=epochs)
            training_history[i, j] = th
            models[i, j] = bvae
    return models, training_history

def evaluate_multiple_models(dg, models, train_func, test_distributions,
                             **classifier_args):
    performance = np.zeros(models.shape + (len(test_distributions),))
    chance = np.zeros_like(performance)
    for i, beta in enumerate(betas):
        for j in range(n_reps):
            bvae = models[i, j]
            for k, td in enumerate(test_distributions):
                out = classifier_generalization(dg, bvae, train_func,
                                                test_distrib=td,
                                                **classifier_args)
                performance[i, j, k] = out[0]
                chance[i, j, k]= out[1]
    return performance, chance

def train_and_evaluate_models(dg, betas, layer_spec, train_func,
                              test_distributions, n_reps=10,
                              n_train_samps=10**6, epochs=5,
                              models=None, **classifier_args):
    if models is None:
        models, th = train_multiple_bvae(dg, betas, layer_spec, n_reps=n_reps,
                                         n_train_samps=n_train_samps,
                                         epochs=epochs)
        out2 = (models, th)
    else:
        out2 = (models, None)
    out = evaluate_multiple_models(dg, models, train_func, test_distributions,
                                   **classifier_args)
    return out, out2 
                              
