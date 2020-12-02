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

    
class FlexibleDisentangler(da.TFModel):

    def __init__(self, input_shape, layer_shapes, encoded_size, true_inp_dim,
                 n_partitions=5, regularizer_weight=.01, act_func=tf.nn.relu,
                 **layer_params):
        enc = self.make_encoder(input_shape, layer_shapes, encoded_size,
                                n_partitions, act_func=act_func,
                                regularizer_weight=regularizer_weight,
                                **layer_params)
        self.encoder = enc
        self.regularizer_weight = regularizer_weight
        self.model = tfk.Model(inputs=self.encoder.inputs,
                               outputs=self.encoder.outputs[0])
        out = da.generate_partition_functions(true_inp_dim,
                                              n_funcs=n_partitions)
        self.p_funcs, self.p_vectors, self.p_offsets = out
        self.n_partitions = n_partitions
        self.rep_model = tfk.Model(inputs=self.encoder.inputs,
                                   outputs=self.encoder.layers[-2].output)
        self.input_shape = input_shape
        self.encoded_size = encoded_size
        self.compiled = False

    def generate_target(self, inps):
        cats = list(pf(inps) for pf in self.p_funcs)
        if len(cats) > 0:
            target = np.stack(cats, axis=1)
        else:
            target = np.zeros((len(inps), 0))
        return target
        
    def save(self, path):
        tf_entries = ('encoder', 'model')
        self._save_wtf(path, tf_entries)

    @classmethod
    def load(cls, path):
        dummy = FlexibleDisentangler(0, (10,), 0, 5)
        return cls._load_model(dummy, path)
        
    def make_encoder(self, input_shape, layer_shapes, encoded_size,
                     n_partitions, act_func=tf.nn.relu, regularizer_weight=.1,
                     layer_type=tfkl.Dense, **layer_params):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=input_shape))
            
        for lp in layer_shapes:
            l_i = layer_type(*lp, activation=act_func, **layer_params)
            layer_list.append(l_i)

        l2_reg = tfk.regularizers.L2(l2=regularizer_weight)
        layer_list.append(tfkl.Dense(encoded_size, activation=None,
                                     activity_regularizer=l2_reg))
        
        sig_act = tf.keras.activations.sigmoid
        layer_list.append(tfkl.Dense(n_partitions, activation=sig_act))
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

        train_y = self.generate_target(train_y)
        eval_y = self.generate_target(eval_y)
            
        if eval_x is not None and eval_y is not None:
            eval_set = (eval_x, eval_y)
        else:
            eval_set = None

        out = self.model.fit(x=train_x, y=train_y, epochs=epochs,
                             validation_data=eval_set, batch_size=batch_size,
                             **kwargs)
        return out   

    def get_representation(self, samples):
        rep = self.rep_model(samples)
        return rep

    def get_categorizations(self, samples, hard=False):
        cats = self.encoder(samples)
        if hard:
            cats[cats > .5] = 1
            cats[cats <= .5] = 0
        return cats
    
def pad_zeros(x, dim):
    to_pad = dim - x.shape[1]
    add = np.zeros((x.shape[0], to_pad))
    x_new = np.concatenate((x, add), axis=1)
    return x_new

class FlexibleDisentanglerAE(FlexibleDisentangler):

    def __init__(self, input_shape, layer_shapes, encoded_size,
                 true_inp_dim=None, n_partitions=5, regularizer_weight=0,
                 act_func=tf.nn.relu,
                 branch_names=('class_branch', 'autoenc_branch'),
                 **layer_params):
        if true_inp_dim is None:
            true_inp_dim = encoded_size
        self.regularizer_weight = regularizer_weight
        out = self.make_encoder(input_shape, layer_shapes, encoded_size,
                                n_partitions, act_func=act_func,
                                regularizer_weight=regularizer_weight,
                                branch_names=branch_names,
                                **layer_params)
        inputs, rep, class_branch, autoenc_branch = out

        self.branch_names = branch_names
        # self.branches = dict(zip(branch_names,
        #                          (class_branch, autoenc_branch)))
        # self.class_branch = class_branch
        # self.autoenc_branch = autoenc_branch
        outputs = [class_branch, autoenc_branch]
        self.model = tfk.Model(inputs=inputs,
                               outputs=outputs)
        out = da.generate_partition_functions(true_inp_dim,
                                              n_funcs=n_partitions)
        self.n_partitions = n_partitions
        self.p_funcs, self.p_vectors, self.p_offsets = out
        self.rep_model = tfk.Model(inputs=inputs, outputs=rep)
        self.input_shape = input_shape
        self.encoded_size = encoded_size
        self.compiled = False

    def save(self, path):
        tf_entries = ('model', 'rep_model')
        self._save_wtf(path, tf_entries)

    @classmethod
    def load(cls, path):
        dummy = FlexibleDisentanglerAE(0, ((10,),), 0, 5)
        return cls._load_model(dummy, path)
        
    def make_encoder(self, input_shape, layer_shapes, encoded_size,
                     n_partitions, act_func=tf.nn.relu, regularizer_weight=.1,
                     layer_type=tfkl.Dense, branch_names=('a', 'b'),
                     **layer_params):
        inputs = tfk.Input(shape=input_shape)
        x = inputs
        for lp in layer_shapes:
            x = layer_type(*lp, activation=act_func, **layer_params)(x)

        # representation layer
        l2_reg = tfk.regularizers.L2(l2=regularizer_weight)
        rep = tfkl.Dense(encoded_size, activation=None,
                         activity_regularizer=l2_reg)(x)

        # partition branch
        sig_act = tf.keras.activations.sigmoid
        class_branch = tfkl.Dense(n_partitions, activation=sig_act,
                                  name=branch_names[0])(x)

        # decoder branch
        z = rep
        for lp in layer_shapes[::-1]:
            z = tfkl.Dense(*lp, activation=act_func, **layer_params)(z)

        autoenc_branch = layer_type(input_shape, activation=act_func,
                                    name=branch_names[1], **layer_params)(z)
        return inputs, rep, class_branch, autoenc_branch

    def _compile(self, *args, loss=tf.losses.MeanSquaredError(), **kwargs):
        loss_dict = {self.branch_names[0]:loss,
                     self.branch_names[1]:loss}
        if self.n_partitions == 0:
            loss_dict[self.branch_names[0]] = lambda x, y: 0.
        super()._compile(*args, loss=loss_dict, **kwargs)
    
    def fit(self, train_x, train_y, eval_x=None, eval_y=None, epochs=15,
            data_generator=None, batch_size=32, **kwargs):
        if not self.compiled:
            self._compile()

        train_y = self.generate_target(train_y)
        train_y_dict = {self.branch_names[0]:train_y,
                        self.branch_names[1]:train_x}
        
        eval_y = self.generate_target(eval_y)
            
        if eval_x is not None and eval_y is not None:
            eval_y_dict = {self.branch_names[0]:eval_y,
                           self.branch_names[1]:eval_x}
            eval_set = (eval_x, eval_y_dict)
        else:
            eval_set = None

        out = self.model.fit(x=train_x, y=train_y_dict, epochs=epochs,
                             validation_data=eval_set, batch_size=batch_size,
                             **kwargs)
        return out   

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
        self.loaded = False

    def save(self, path):
        tf_entries = ('encoder', 'decoder', 'vae')
        self.encoder.layers[-1].activity_regularizer = None
        self._save_wtf(path, tf_entries)

    @classmethod
    def load(cls, path):
        dummy = BetaVAE(0, ((10,),), 0)
        model = cls._load_model(dummy, path, skip=('vae',))
        if model.beta > 0:
            prior = tfd.Independent(tfd.Normal(loc=tf.zeros(model.encoded_size),
                                               scale=1),
                                    reinterpreted_batch_ndims=1)
            rep_reg = tfpl.KLDivergenceRegularizer(prior, weight=model.beta)
            model.encoder.layers[-1].activity_regularizer = rep_reg
        model.var = tfk.Model(inputs=model.encoder.inputs,
                              outputs=model.decoder(model.encoder.outputs[0]))
        model.loaded = True
        model._compile()
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
        if self.loaded:
            rep = self.encoder(samples)
        else:
            rep = self.encoder(samples).sample()
        return rep
