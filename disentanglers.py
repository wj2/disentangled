import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_hub as tfhub
import functools as ft

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

    def _compile(self, optimizer=None,
                 loss=tf.losses.MeanSquaredError(), loss_weights=None):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.model.compile(optimizer, loss, loss_weights=loss_weights)
        self.compiled = True

    def fit_sets(self, train_set, eval_set=None, **kwargs):
        train_y, train_x = train_set
        if eval_set is not None:
            eval_y, eval_x = eval_set
        else:
            eval_y, eval_x = None, None
        return self.fit(train_x, train_y, eval_x=eval_x, eval_y=eval_y, **kwargs)

    def fit(self, train_x, train_y, eval_x=None, eval_y=None, epochs=15,
            data_generator=None, batch_size=32, standard_loss=True, **kwargs):
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

    def __init__(self, input_shape, layer_shapes, encoded_size,
                 true_inp_dim=None, n_partitions=5, regularizer_weight=.01,
                 act_func=tf.nn.relu, offset_distr=None, **layer_params):
        if true_inp_dim is None:
            true_inp_dim = encoded_size
        enc = self.make_encoder(input_shape, layer_shapes, encoded_size,
                                n_partitions, act_func=act_func,
                                regularizer_weight=regularizer_weight,
                                **layer_params)
        self.encoder = enc
        self.regularizer_weight = regularizer_weight
        self.model = tfk.Model(inputs=self.encoder.inputs,
                               outputs=self.encoder.outputs[0])
        out = da.generate_partition_functions(true_inp_dim,
                                              n_funcs=n_partitions,
                                              offset_distribution=offset_distr)
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

        l2_reg = tfk.regularizers.l2(regularizer_weight)
        layer_list.append(tfkl.Dense(encoded_size, activation=None,
                                     activity_regularizer=l2_reg))
        
        sig_act = tf.keras.activations.sigmoid
        layer_list.append(tfkl.Dense(n_partitions, activation=sig_act))
        enc = tfk.Sequential(layer_list)
        return enc

    def _compile(self, optimizer=None,
                 loss=tf.losses.MeanSquaredError(), loss_weights=None):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-4)
        self.model.compile(optimizer, loss, loss_weights=loss_weights)
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
        if eval_y is not None:
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

def _mse_nanloss(label, prediction):
    nan_mask = tf.math.logical_not(tf.math.is_nan(label))
    label = tf.boolean_mask(label, nan_mask)
    prediction = tf.boolean_mask(prediction, nan_mask)
    mult = tf.square(prediction - label)
    mse_nanloss = tf.reduce_mean(mult)
    return mse_nanloss

def _binary_crossentropy_nan(label, prediction):
    nan_mask = tf.math.logical_not(tf.math.is_nan(label))
    label = tf.boolean_mask(label, nan_mask)
    prediction = tf.boolean_mask(prediction, nan_mask)
    bcel = tf.keras.losses.binary_crossentropy(label, prediction)
    return bcel

class IdentityModel(da.TFModel):

    def __init__(self, flatten=False):
        self.p_vectors = []
        self.p_offsets = []
        self.flatten = flatten
        
    def get_representation(self, x):
        if self.flatten:
            x = np.reshape(x, (x.shape[0], -1))
        return x

class SingleLayer(da.TFModel):

    def __init__(self, input_dim, output_dim):
        self.p_vectors = []
        self.p_offsets = []
        x = tfkl.Input(input_dim)
        rep = tfkl.Dense(output_dim)(x)
        self.generator = tfk.Model(inputs=x, outputs=rep)
        
    def get_representation(self, x):
        return self.generator(x)

class IntermediateLayers():

    def __init__(self, model):
        self.model = model
        inp = tfk.Input(self.model.layers[0].input_shape[0][1])
        x = inp
        i_models = []
        for i in range(0, len(self.model.layers)):
            x = self.model.layers[i](x)
            i_models.append(tfk.Model(inputs=inp, outputs=x))
        self.i_models = i_models
        self.use_i = 0

    def get_representation(self, x, layer_ind=None):
        if layer_ind is None:
            layer_ind = self.use_i
        return self.i_models[layer_ind](x)
    
class FlexibleDisentanglerAE(FlexibleDisentangler):

    def __init__(self, input_shape, layer_shapes, encoded_size,
                 true_inp_dim=None, n_partitions=5, regularizer_weight=0,
                 act_func=tf.nn.relu, orthog_partitions=False,
                 branch_names=('class_branch', 'autoenc_branch'),
                 offset_distr=None, contextual_partitions=False,
                 no_autoenc=False, loss_ratio=10, dropout_rate=0,
                 regularizer_type=tfk.regularizers.l2,
                 noise=0, context_offset=False, nan_salt=None,
                 grid_coloring=False, n_granules=2, granule_sparseness=.5,
                 n_grids=0, **layer_params):
        if true_inp_dim is None:
            true_inp_dim = encoded_size
        self.regularizer_weight = regularizer_weight
        self.nan_salt = nan_salt
        out = self.make_encoder(input_shape, layer_shapes, encoded_size,
                                n_partitions, act_func=act_func,
                                regularizer_weight=regularizer_weight,
                                branch_names=branch_names,
                                dropout_rate=dropout_rate,
                                regularizer_type=regularizer_type,
                                noise=noise, **layer_params)
        model, rep_model, autoenc_model, class_model = out
        self.branch_names = branch_names
        self.model = model
        
        out = da.generate_partition_functions(
            true_inp_dim,
            n_funcs=n_partitions,
            orth_basis=orthog_partitions,
            offset_distribution=offset_distr,
            contextual=contextual_partitions)
        self.p_funcs, self.p_vectors, self.p_offsets = out
        if grid_coloring or n_grids > 0:
            if grid_coloring:
                n_g = n_partitions
            else:
                n_g = n_grids
            out = da.generate_grid_functions(true_inp_dim,
                                             n_funcs=n_g,
                                             n_granules=n_granules,
                                             sparseness=granule_sparseness)
            p_fs_g, p_vs_g, p_os_g = out, (None,)*n_g, (None,)*n_g
        if grid_coloring and n_grids > 0:
            self.p_funcs = p_fs_g
            self.p_vectors = None
            self.p_offsets = None
        elif n_grids > 0:
            self.p_funcs = np.concatenate((self.p_funcs, p_fs_g))
        self.contextual_partitions = contextual_partitions
        self.n_partitions = n_partitions
        self.rep_model = rep_model
        self.input_shape = input_shape
        self.encoded_size = encoded_size
        self.compiled = False
        self.no_autoencoder = no_autoenc
        self.loss_ratio = loss_ratio
        self.recon_model = autoenc_model
        self.layer_shapes = layer_shapes

    def save(self, path):
        tf_entries = ('model', 'rep_model', 'recon_model')
        self._save_wtf(path, tf_entries)

    @classmethod
    def load(cls, path):
        dummy = FlexibleDisentanglerAE(0, ((10,),), 0, 5)
        return cls._load_model(dummy, path)
        
    def make_encoder(self, input_shape, layer_shapes, encoded_size,
                     n_partitions, act_func=tf.nn.relu, regularizer_weight=.1,
                     layer_type=tfkl.Dense, branch_names=('a', 'b'),
                     regularizer_type=tfk.regularizers.l2,
                     dropout_rate=0, noise=0, **layer_params):
        inputs = tfk.Input(shape=input_shape)
        x = inputs
        for lp in layer_shapes:
            x = layer_type(*lp, activation=act_func, **layer_params)(x)

        if dropout_rate > 0:
            x = tfkl.Dropout(dropout_rate)(x)
        
        # representation layer
        act_reg = regularizer_type(regularizer_weight)
        rep = tfkl.Dense(encoded_size, activation=None,
                         activity_regularizer=act_reg)(x)
        if noise > 0:
            rep = tfkl.GaussianNoise(noise)(rep)
        rep_model = tfk.Model(inputs=inputs, outputs=rep)
        rep_inp = tfk.Input(shape=encoded_size)
        
        # partition branch
        class_inp = rep_inp
        sig_act = tf.keras.activations.sigmoid
        class_branch = tfkl.Dense(n_partitions, activation=sig_act,
                                  name=branch_names[0])(class_inp)
        class_model = tfk.Model(inputs=rep_inp, outputs=class_branch,
                                name=branch_names[0])

        # decoder branch
        z = rep_inp
        for lp in layer_shapes[::-1]:
            z = tfkl.Dense(*lp, activation=act_func, **layer_params)(z)

        autoenc_branch = layer_type(input_shape, activation=act_func,
                                    name=branch_names[1], **layer_params)(z)
        autoenc_model = tfk.Model(inputs=rep_inp, outputs=autoenc_branch,
                                  name=branch_names[1])

        outs = [class_model(rep), autoenc_model(rep)]
        full_model = tfk.Model(inputs=inputs, outputs=outs)
        
        return full_model, rep_model, autoenc_model, class_model

    def _compile(self, *args, categ_loss=None,
                 autoenc_loss=None, standard_loss=True,
                 loss_ratio=None, **kwargs):
        if categ_loss is None:
            categ_loss = tfk.losses.BinaryCrossentropy()
        if autoenc_loss is None:
            autoenc_loss = tfk.losses.MeanSquaredError()
        if (not standard_loss or self.contextual_partitions
            or not self.nan_salt is None):
            categ_loss = _binary_crossentropy_nan,
        loss_dict = {self.branch_names[0]:categ_loss,
                     self.branch_names[1]:autoenc_loss}
        if loss_ratio is None:
            loss_ratio = self.loss_ratio
        if self.no_autoencoder:
            loss_weights = {self.branch_names[0]:1, self.branch_names[1]:0}
        else:
            loss_weights = {self.branch_names[0]:1,
                            self.branch_names[1]:loss_ratio}
        if self.n_partitions == 0:
            loss_dict[self.branch_names[0]] = lambda x, y: 0.
        super()._compile(*args, loss=loss_dict, loss_weights=loss_weights,
                         **kwargs)
    
    def fit(self, train_x, train_y, eval_x=None, eval_y=None, epochs=15,
            data_generator=None, batch_size=32, standard_loss=False,
            nan_salt=None, **kwargs): 
        if nan_salt is None:
            nan_salt = self.nan_salt
        if standard_loss or self.contextual_partitions or nan_salt is None:
            comp_kwargs = {'standard_loss':True}
        else:
            comp_kwargs = {'standard_loss':False}
        if not self.compiled:
            self._compile(**comp_kwargs)

        train_y = self.generate_target(train_y)
        if nan_salt is not None and nan_salt != 'single':
            mask = np.random.random_sample(train_y.shape) < nan_salt
            train_y[mask] = np.nan
        elif nan_salt == 'single':
            mask = np.ones_like(train_y, dtype=bool)
            inds = np.random.choice(train_y.shape[1], train_y.shape[0])
            full_inds = list(zip(range(train_y.shape[0]), inds))
            for ind in full_inds:
                mask[ind] = False
            train_y[mask] = np.nan
        train_y_dict = {self.branch_names[0]:train_y,
                        self.branch_names[1]:train_x}

        if eval_y is not None:
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

    def get_reconstruction(self, reps):
        recon = self.recon_model(reps)
        return recon

    def get_reconstruction_mse(self, samples):
        reps = self.get_representation(samples)
        recon = self.get_reconstruction(reps)
        return np.mean((samples - recon)**2)


class FlexibleDisentanglerAEConv(FlexibleDisentanglerAE):

    @classmethod
    def load(cls, path):
        dummy_layers = ((10, 10, 3), (10,))
        dummy = FlexibleDisentanglerAEConv((32, 32, 3), dummy_layers, 10, 5)
        return cls._load_model(dummy, path)

    def make_encoder(self, input_shape, layer_shapes, encoded_size,
                     n_partitions, act_func=tf.nn.relu, regularizer_weight=1,
                     layer_types_enc=None, dropout_rate=0,
                     layer_types_dec=None,
                     regularizer_type=tfk.regularizers.l2,
                     noise=0, 
                     branch_names=('a', 'b'),
                     **layer_params):
        inputs = tfk.Input(shape=input_shape)
        x = inputs
        strides = []
        ll = len(input_shape)
        for i, lp in enumerate(layer_shapes):
            if ll != len(lp):
                transition_shape = x.shape[1:]
                x = tfkl.Flatten()(x)
            ll = len(lp)
            if layer_types_enc is None:
                if len(lp) == 3:
                    layer_type = ft.partial(tfkl.Conv2D, padding='same')
                    strides.append(lp[2])
                elif len(lp) == 1:
                    layer_type = tfkl.Dense
                    strides.append(1)
            else:
                layer_type = layer_types_enc[i]
            x = layer_type(*lp, activation=act_func,
                           **layer_params)(x)
        if ll == 3:
            x = tfkl.Flatten()(x)
            
        if dropout_rate > 0:
            x = tfkl.Dropout(dropout_rate)(x)
                        
        # representation layer
        reg = regularizer_type(regularizer_weight)
        rep = tfkl.Dense(encoded_size, activation=None,
                         activity_regularizer=reg)(x)
        if noise > 0:
            rep = tfkl.GaussianNoise(noise)(rep)
        rep_model = tfk.Model(inputs=inputs, outputs=rep)
        rep_inp = tfk.Input(shape=encoded_size)
        
        # partition branch
        class_inp = rep_inp
        sig_act = tf.keras.activations.sigmoid
        class_branch = tfkl.Dense(n_partitions, activation=sig_act,
                                  name=branch_names[0])(class_inp)
        class_model = tfk.Model(inputs=rep_inp, outputs=class_branch,
                                name=branch_names[0])

        # decoder branch
        z = rep_inp
        ll = 1
        for i, lp in enumerate(layer_shapes[::-1]):
            if ll != len(lp):
                z = tfkl.Dense(np.product(transition_shape), activation=None)(z)
                z = tfkl.Reshape(target_shape=transition_shape)(z)
            ll = len(lp)
            if layer_types_dec is None:
                if len(lp) == 3:
                    layer_type = ft.partial(tfkl.Conv2DTranspose,
                                            padding='same')
                elif len(lp) == 1:
                    layer_type = tfkl.Dense
            else:
                layer_type = layer_types_dec[i]
            z = layer_type(*lp, activation=act_func,
                               **layer_params)(z)

        z = tfkl.Conv2DTranspose(3, 1, strides=1,
                                 activation=None,
                                 padding='same', **layer_params)(z)

        autoenc_branch = tfkl.Conv2DTranspose(3, 1, strides=1,
                                              activation=tf.nn.sigmoid,
                                              name=branch_names[1],
                                              padding='same', **layer_params)(z)
        autoenc_model = tfk.Model(inputs=rep_inp, outputs=autoenc_branch,
                                  name=branch_names[1])

        outs = [class_model(rep), autoenc_model(rep)]
        full_model = tfk.Model(inputs=inputs, outputs=outs)
        return full_model, rep_model, autoenc_model, class_model

    
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
    
    def _compile(self, optimizer=None,
                 loss=tf.losses.MeanSquaredError(), loss_weights=None):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)

        self.model.compile(optimizer, loss, loss_weights=loss_weights)
        self.compiled = True

    def fit_sets(self, train_set, eval_set=None, **kwargs):
        train_x = train_set[1]
        if eval_set is not None:
            eval_x = eval_set[1]
        else:
            eval_x = None
        return self.fit(train_x, eval_x=eval_x, **kwargs)
        
    def fit(self, train_x, eval_x=None, epochs=15,
            data_generator=None, batch_size=32, standard_loss=True,
            **kwargs):
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

    def get_reconstruction(self, reps):
        recon = self.model(reps)
        return recon

    def get_reconstruction_mse(self, samples):
        reps = self.get_representation(samples)
        recon = self.get_reconstruction(reps)
        return np.mean((samples - recon)**2)

class PretrainedModel(da.TFModel):

    def __init__(self, img_shape, model_path, trainable=False):
        full_img_shape = img_shape + (3,)
        layer_list = [tfkl.InputLayer(input_shape=full_img_shape),
                      tfhub.KerasLayer(model_path, trainable=trainable)]
        model = tfk.Sequential(layer_list)
        model.build((None,) + full_img_shape)
        self.encoder = model

    def get_representation(self, samples):
        rep = self.encoder(samples)
        return rep
    
class BetaVAE(da.TFModel):

    def __init__(self, input_shape, layer_shapes, encoded_size,
                 act_func=tf.nn.relu, beta=1, dropout_rate=0,
                 full_cov=False, rescaler=True, **layer_params):
        enc, prior = self.make_encoder(input_shape, layer_shapes, encoded_size,
                                       act_func=act_func, beta=beta,
                                       full_cov=full_cov, **layer_params)
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
        self.rescaler = rescaler
        self.rescale_factor = None

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
                     conv=False, beta=1, full_cov=True, **layer_params):
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=input_shape))
        if transform_layer is not None:
            layer_list.append(tfkl.Lambda(transform_layer))
            
        for lp in layer_shapes:
            l_i = layer_type(*lp, activation=act_func, **layer_params)
            layer_list.append(l_i)

        if conv:
            layer_list.append(tfkl.Flatten())
        if full_cov:
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
            
        if full_cov:
            rep_layer = tfpl.MultivariateNormalTriL(encoded_size,
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
                     conv=False, out_eps=.0001, **layer_params):
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

    def _compile(self, optimizer=None,
                 loss=da.negloglik, loss_weights=None):
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=1e-3)
        self.vae.compile(optimizer, loss, loss_weights=loss_weights)
        self.compiled = True

    def fit_sets(self, train_set, eval_set=None, **kwargs):
        _, train_x = train_set
        if eval_set is not None:
            _, eval_x = eval_set
        else:
            eval_x = None
        return self.fit(train_x, eval_x=eval_x, **kwargs)
        
    def fit(self, train_x, train_y=None, eval_x=None, eval_y=None, epochs=15,
            data_generator=None, batch_size=32, standard_loss=True, **kwargs):
        if not self.compiled:
            self._compile()

        if train_y is None:
            train_y = train_x
            
        if self.rescaler and self.rescale_factor is None:
            avg_max = np.median(np.max(train_x, axis=0))
            self.rescale_factor = 1/avg_max
        if self.rescaler:
            train_x = self.rescale_factor*train_x
            train_y = self.rescale_factor*train_y
            
        if eval_x is not None and eval_y is None:
            eval_y = eval_x
            if self.rescaler:
                eval_x = self.rescale_factor*eval_x
                eval_y = self.rescale_factor*eval_y
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

    def sample_latents(self, sample_size=10, use_mean=True):
        samps = self.prior.sample(sample_size)
        distr = self.decoder(samps)
        if use_mean:
            outs = distr.mean()
        else:
            outs = distr.sample()
        return outs

    def get_representation(self, samples, use_loaded=False, use_mean=True):
        if self.rescaler:
            samples = self.rescale_factor*samples
        if self.loaded and use_loaded:
            rep = self.encoder(samples)
        else:
            if use_mean:
                rep = self.encoder(samples).mean()
            else:
                rep = self.encoder(samples).sample()
        return rep

    def get_reconstruction_mse(self, samples, use_mean=True):
        reps = self.get_representation(samples, use_mean=use_mean)
        recon = self.get_reconstruction(reps, use_mean=use_mean)
        return np.mean((samples - recon)**2)
    
    def get_reconstruction(self, reps, use_mean=True):
        if self.loaded:
            recon = self.decoder(reps)
        else:
            if use_mean:
                recon = self.decoder(reps).mean()
            else:
                recon = self.decoder(reps).sample()
        if self.rescaler:
            recon = recon/self.rescale_factor
        return recon

class BetaVAEConv(BetaVAE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, rescaler=False, **kwargs)
    
    @classmethod
    def load(cls, path):
        dummy = BetaVAEConv((32, 32, 1), ((10, 1, 1), (10,)), 2)
        model = cls._load_model(dummy, path, skip=('vae',))
        if model.beta > 0:
            prior = tfd.Independent(tfd.Normal(loc=tf.zeros(model.encoded_size),
                                               scale=1),
                                    reinterpreted_batch_ndims=1)
            rep_reg = tfpl.KLDivergenceRegularizer(prior, weight=model.beta)
            model.encoder.layers[-1].activity_regularizer = rep_reg
        model.var = tfk.Model(inputs=model.encoder.inputs,
                              outputs=model.decoder(model.encoder.outputs[0]))
        model.rescale_factor = 1
        model.loaded = True
        model._compile()
        return model
    
    def make_encoder(self, input_shape, layer_shapes, encoded_size,
                     act_func=tf.nn.relu, strides=1,
                     transform_layer=None, layer_type=None,
                     conv=False, beta=1, full_cov=True, dropout_rate=0,
                     output_distrib=None, **layer_params):
        inputs = tfkl.InputLayer(input_shape=input_shape)
        layer_list = [inputs]
        strides = []
        ll = len(input_shape)
        shape = (None,) + input_shape
        for i, lp in enumerate(layer_shapes):
            if ll != len(lp):
                self.transition_shape = shape
                layer_list.append(tfkl.Flatten())
                shape = layer_list[-1].compute_output_shape(shape)
            ll = len(lp)
            if layer_type is None:
                if len(lp) == 3:
                    layer_type_i = ft.partial(tfkl.Conv2D, padding='same')
                    strides.append(lp[2])
                elif len(lp) == 1:
                    layer_type_i = tfkl.Dense
                    strides.append(1)
            else:
                layer_type_i = layer_type[i]
            layer_list.append(layer_type_i(*lp, activation=act_func,
                                           **layer_params))
            shape = layer_list[-1].compute_output_shape(shape)
        if ll == 3:
            layer_list.append(tfkl.Flatten())
            
        if dropout_rate > 0:
            layer_list.append(tfkl.Dropout(dropout_rate))
                        
        # representation layer
        if full_cov:
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

        if full_cov:
            layer_list.append(tfpl.MultivariateNormalTriL(
                encoded_size, activity_regularizer=rep_reg))
        else:
            layer_list.append(tfpl.IndependentNormal(
                encoded_size,
                activity_regularizer=rep_reg))
        rep_model = tfk.Sequential(layer_list)
        return rep_model, prior

    def make_decoder(self, input_shape, layer_shapes, encoded_size,
                     act_func=tf.nn.relu, strides=1,
                     transform_layer=None, layer_type=None,
                     conv=False, out_eps=.01, output_distrib=None,
                     **layer_params):
        z = tfkl.InputLayer(input_shape=encoded_size)
        ll = 1
        layer_list = [z]
        for i, lp in enumerate(layer_shapes):
            if ll != len(lp):
                layer_list.append(tfkl.Dense(
                    np.product(self.transition_shape[1:]),
                    activation=None))
                layer_list.append(
                    tfkl.Reshape(target_shape=self.transition_shape[1:]))
            ll = len(lp)
            if layer_type is None:
                if len(lp) == 3:
                    layer_type_i = ft.partial(tfkl.Conv2DTranspose,
                                            padding='same')
                elif len(lp) == 1:
                    layer_type_i = tfkl.Dense
            else:
                layer_type_i = layer_type[i]
            layer_list.append(layer_type_i(*lp, activation=act_func,
                                           **layer_params))

        col_dim = input_shape[-1]
        layer_list.append(tfkl.Conv2DTranspose(col_dim, 1, strides=1,
                                               activation=None,
                                               padding='same', **layer_params))

        if output_distrib is None:
            fixed_std = lambda x: tfd.Normal(x, out_eps)
            out_distr = tfpl.DistributionLambda(
                                                make_distribution_fn=fixed_std)
        elif output_distrib == 'binary':
            layer_list.append(tfkl.Flatten())
            out_distr = tfpl.IndependentBernoulli(input_shape,
                                                  tfd.Bernoulli.logits)
        else:
            layer_list.append(tfkl.Flatten())
            out_distr = output_distrib(event_shape=input_shape)
        layer_list.append(out_distr)

        dec = tfk.Sequential(layer_list)
        return dec

    def get_reconstruction(self, reps, use_mean=True):
        recon = super().get_reconstruction(reps, use_mean)
        if self.loaded:
            recon = tfd.Bernoulli(logits=recon)
            if use_mean:
                recon = recon.mean()
            else:
                recon = recon.sample()
        return recon
