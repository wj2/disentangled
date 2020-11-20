import tensorflow as tf
import tensorflow_probability as tfp

import numpy as np

import disentangled.aux as da
import disentangled.regularizer as dr

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

class SupervisedDisentangler(da.TFModel):

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

    def save(self, path):
        tf_entries = ('encoder', 'model')
        self._save_wtf(path, tf_entries)

    @classmethod
    def load(cls, path):
        dummy = SupervisedDisentangler(0, (10,), 0)
        return cls._load_model(dummy, path)
        
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
            data_generator=None, batch_size=32, **kwargs):
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
                             validation_data=eval_set, batch_size=batch_size,
                             **kwargs)
        return out   

    def get_representation(self, samples):
        rep = self.encoder(samples)
        return rep

    
def pad_zeros(x, dim):
    to_pad = dim - x.shape[1]
    add = np.zeros((x.shape[0], to_pad))
    x_new = np.concatenate((x, add), axis=1)
    return x_new
    
class StandardAE(da.TFModel):

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

    def save(self, path):
        tf_entries = ('encoder', 'decoder', 'model')
        self._save_wtf(path, tf_entries)

    @classmethod
    def load(cls, path):
        dummy = StandardAE(0, (10,), 0)
        return cls._load_model(dummy, path)
        
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
            data_generator=None, batch_size=32, **kwargs):
        if not self.compiled:
            self._compile()
            
        if eval_x is not None:
            eval_set = (eval_x, eval_x)
        else:
            eval_set = None

        out = self.model.fit(x=train_x, y=train_x, epochs=epochs,
                             validation_data=eval_set, batch_size=batch_size,
                             **kwargs)
        return out   

    def get_representation(self, samples):
        rep = self.encoder(samples)
        return rep
    
    
class BetaVAE(da.TFModel):

    def __init__(self, input_shape, layer_shapes, encoded_size,
                 act_func=tf.nn.relu, beta=1, **layer_params):
        enc, prior = self.make_encoder(input_shape, layer_shapes, encoded_size,
                                       act_func=act_func, beta=beta,
                                       **layer_params)
        self.encoder = enc
        self.prior = prior
        self.beta = beta
        self.decoder = self.make_decoder(input_shape, layer_shapes[::-1],
                                         encoded_size, act_func=act_func,
                                         **layer_params)
        self.vae = tfk.Model(inputs=self.encoder.inputs,
                             outputs=self.decoder(self.encoder.outputs[0]))
        self.input_shape = input_shape
        self.encoded_size = encoded_size
        self.compiled = False

    def save(self, path):
        tf_entries = ('encoder', 'decoder', 'vae')
        self.encoder.layers[-1].activity_regularizer = None
        self._save_wtf(path, tf_entries)

    @classmethod
    def load(cls, path):
        dummy = BetaVAE(0, ((10,),), 0)
        model = cls._load_model(dummy, path)
        if model.beta > 0:
            prior = tfd.Independent(tfd.Normal(loc=tf.zeros(model.encoded_size),
                                               scale=1),
                                    reinterpreted_batch_ndims=1)
            rep_reg = tfpl.KLDivergenceRegularizer(prior, weight=model.beta)
            model.encoder.layers[-1].activity_regularizer = rep_reg
        return model

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
                 loss=da.negloglik):
        self.vae.compile(optimizer, loss)
        self.compiled = True

    def fit_sets(self, train_set, eval_set=None, **kwargs):
        _, train_x = train_set
        if eval_set is not None:
            _, eval_x = eval_set
        else:
            eval_x = None
        return self.fit(train_x, eval_x=eval_x, **kwargs)
        
    def fit(self, train_x, train_y=None, eval_x=None, eval_y=None, epochs=15,
            data_generator=None, batch_size=32, **kwargs):
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
                           validation_data=eval_set, batch_size=batch_size,
                           **kwargs)
        return out

    def sample_latents(self, sample_size=10):
        samps = self.prior.sample(sample_size)
        outs = self.decoder(samps).mean()
        return outs

    def get_representation(self, samples):
        rep = self.encoder(samples).sample()
        return rep
