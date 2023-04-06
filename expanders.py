
import numpy as np
import scipy.stats as sts

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_similarity as tfsim
import tensorflow_addons as tfa

import disentangled.regularizer as dr

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions

def sample_pairs(fdg, n_samps=1000, distance=.1):
    lvs, reps1 = fdg.sample_reps(n_samps)
    perturbs = sts.norm(0, distance).rvs(lvs.shape)
    reps2 = fdg.get_representation(lvs + perturbs)
    return reps1, reps2

class InputExpander:

    def __init__(self, inp_dim, layers, rep_dim, pred_layers, algorithm='barlow',
                 **kwargs):

        self.inp_dim = inp_dim
        self.rep_dim = rep_dim

        out = self.make_model(inp_dim, layers, rep_dim, pred_layers, **kwargs)
        self.rep_model, self.pred_model = out
        self.algorithm = algorithm

    def make_model(self, inp_dim, rep_layers, rep_dim,
                   pred_layers, 
                   act_reg_weight=(0, .1),
                   act_func=tf.nn.relu,
                   kernel_init=None,
                   layer_type=tfkl.Dense,
                   noise=.01,
                   reg=dr.L2PRRegularizerInv):
        if kernel_init is None:
            kernel_init = tfk.initializers.GlorotUniform
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=inp_dim))

        regularizer = reg(act_reg_weight)

        for hd in rep_layers:
            l_i = layer_type(hd, activation=act_func,
                             activity_regularizer=regularizer,
                             kernel_initializer=kernel_init())
            layer_list.append(l_i)
            if noise is not None:
                layer_list.append(tfkl.GaussianNoise(noise))

        last_layer = layer_type(rep_dim, activation=act_func,
                                activity_regularizer=regularizer,
                                kernel_initializer=kernel_init)
        layer_list.append(last_layer)

        rep_model = tfk.Sequential(layer_list)
        for pred in pred_layers:
            l_i = layer_type(pred, activation=act_func,
                             kernel_initializer=kernel_init())
            layer_list.append(l_i)
            
        pred_model = tfk.Sequential(layer_list)
        return rep_model, pred_model

    def compile(self, **kwargs):

        
        loss, opt = self_super_algorithms[self.algorithm]
        self.model.compile(loss=loss, optimizer=opt)
        self.compiled = True

    def fit(self, input_model, n_train=10000, n_val=1000,
            view_distance=.1, batch_size=100, pre_train_epochs=800,
            pre_train_steps_per_epoch=None, val_steps_per_epoch=20,
            weight_decay=5e-4, warmup_lr=0, warmup_steps=0,
            temperature=None, momentum=.9, **kwargs):
        if pre_train_steps_per_epoch is None:
            pre_train_steps_per_epoch = n_train // batch_size

        if self.algorithm == "simsiam":
            init_lr = 3e-2 * int(batch_size / 256)
            loss = tfsim.losses.SimSiamLoss(projection_type="cosine_distance",
                                            name=self.algorithm)
            lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=init_lr,
                decay_steps=pre_train_epochs * pre_train_steps_per_epoch,
            )
            wd_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=weight_decay,
                decay_steps=pre_train_epochs * pre_train_steps_per_epoch,
            )
            optimizer = tfa.optimizers.SGDW(
                learning_rate=lr_decayed_fn,
                weight_decay=wd_decayed_fn,
                momentum=momentum,
            )

        elif self.algorithm == "barlow":
            init_lr = 1e-3 
            warmup_steps = 1000 
            loss = tfsim.losses.Barlow(name=self.algorithm)
            optimizer = tfa.optimizers.LAMB(learning_rate=init_lr)
            # optimizer = tf.optimizers.Adam(learning_rate=init_lr)

        elif self.algorithm == "simclr":
            init_lr = 1e-3  
            temperature = 0.5
            loss = tfsim.losses.SimCLRLoss(name=self.algorithm,
                                           temperature=temperature)
            optimizer = tfa.optimizers.LAMB(learning_rate=init_lr)
            # optimizer = tf.optimizers.Adam(learning_rate=init_lr)
        elif self.algorithm == "vicreg":
            init_lr = 1e-3
            loss = tfsim.losses.VicReg(name=self.algorithm)
            optimizer = tfa.optimizers.LAMB(learning_rate=init_lr)
        else:
            raise ValueError(f"{ALGORITHM} is not supported.")
        
        contrastive_model = tfsim.models.contrastive_model.ContrastiveModel(
            backbone=self.rep_model,
            projector=self.pred_model,
            algorithm=self.algorithm,
            name=self.algorithm,
        )
        contrastive_model.compile(loss=loss, optimizer=optimizer)
        
        train12 = sample_pairs(input_model, n_samps=n_train,
                               distance=view_distance)
        train12 = tf.data.Dataset.from_tensor_slices(train12)
        train12 = train12.shuffle(1024)
        train12 = train12.batch(batch_size)
        
        val12 = sample_pairs(input_model, n_samps=n_val,
                             distance=view_distance)
        val12 = tf.data.Dataset.from_tensor_slices(val12)
        val12 = val12.shuffle(1024)
        val12 = val12.batch(batch_size)
        
        h = contrastive_model.fit(train12, validation_data=val12,
                                  epochs=pre_train_epochs,
                                  steps_per_epoch=pre_train_steps_per_epoch,
                                  validation_steps=val_steps_per_epoch,
                                  verbose=1,
                                  **kwargs)
        return contrastive_model, h
