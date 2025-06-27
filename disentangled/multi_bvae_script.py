import argparse
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft
import tensorflow as tf
import tensorflow.keras as tfk

import general.utility as u

import disentangled.characterization as dc
import disentangled.auxiliary as da
import disentangled.disentanglers as dd
import disentangled.data_generation as dg

def create_parser():
    parser = argparse.ArgumentParser(description='fit several autoencoders')
    parser.add_argument('-o', '--output_folder', default='results-n', type=str,
                        help='folder to save the output in')
    parser.add_argument('-s', '--no_multiprocessing', default=False,
                        action='store_true',
                        help='path to trained data generator')
    parser.add_argument('-d', '--data_generator', type=str, default=None,
                        help='file with saved data generator to use')
    parser.add_argument('-l', '--latent_dims', default=None, type=int,
                        help='number of dimensions to use in latent layer')
    parser.add_argument('-i', '--input_dims', default=2, type=int,
                        help='true number of dimensions in input')
    parser.add_argument('-b', '--betas', nargs='*', type=float,
                        help='betas with which to train models')
    parser.add_argument('--beta_mult', default=1, type=float,
                        help='multiply submitted beta value by this, '
                        'useful for running as an array job')
    parser.add_argument('-c', '--save_datagenerator', default=None,
                        type=str, help='path to save data generator at')
    parser.add_argument('-n', '--n_reps', default=5,
                        type=int, help='number of times to train each model')
    parser.add_argument('-t', '--n_train_diffs', default=6, type=int,
                        help='number of different training data sample sizes '
                        'to use')
    parser.add_argument('--n_train_bounds', nargs=2, default=(2, 6.5),
                        type=float, help='order of magnitudes for range to '
                        'sample training differences from within (using '
                        'logspace)')
    parser.add_argument('--no_models', default=False, action='store_true',
                        help='do not store tensorflow models')
    parser.add_argument('--test', default=False, action='store_true',
                        help='run minimal code to test that this script works')
    parser.add_argument('--show_prints', default=False,
                        action='store_true',
                        help='print training information for disentangler '
                        'models')
    parser.add_argument('--use_rf_dg', default=False,
                        action='store_true',
                        help='use an RF-based data generator')
    parser.add_argument('--dg_layer_spec', default=None, type=int, nargs='*',
                        help='the layer sizes to use')
    parser.add_argument('--dg_dim', default=200, type=int,
                        help='dimensionality of the data generator')
    parser.add_argument('--batch_size', default=30, type=int,
                        help='batch size to use for training model')
    parser.add_argument('--dropout', default=0, type=float,
                        help='amount of dropout to include during model '
                        'training')
    parser.add_argument('--model_epochs', default=200, type=int,
                        help='the number of epochs to train each model for')
    parser.add_argument('--full_cov', default=False, action='store_true',
                        help='fit the full covariance matrix')
    parser.add_argument('--config_path', default=None, type=str,
                        help='path to config file to use, will override other '
                        'params')
    parser.add_argument('--use_tanh', default=False, action='store_true',
                        help='use tanh instead of relu transfer function')
    parser.add_argument('--layer_spec', default=None, type=int, nargs='*',
                        help='the layer sizes to use')
    parser.add_argument('--train_dg', default=False, action='store_true',
                        help='train data generator')
    parser.add_argument('--source_distr', default='normal', type=str,
                        help='distribution to sample from (normal or uniform)')
    parser.add_argument('--no_data', default=False, action='store_true',
                        help='do not save representation samples')
    parser.add_argument('--use_prf_dg', default=False, action='store_true',
                        help='use participation ratio-optimized data generator')
    parser.add_argument('--rf_width', default=4, type=float,
                        help='scaling of RFs for RF data generator')
    parser.add_argument('--rf_input_noise', default=0, type=float,
                        help='noise applied to latent variables before '
                        'they are passed to the RF functions (default 0)')
    parser.add_argument('--rf_output_noise', default=0, type=float,
                        help='noise applied to the output of the RF '
                        'functions (default 0)')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    if args.config_path is not None:
        config_dict = pickle.load(open(args.config_path, 'rb'))
        args = u.merge_params_dict(args, config_dict)

    true_inp_dim = args.input_dims
    est_inp_dim = args.latent_dims
    if args.test:
        n_reps = 1
        n_train_diffs = 1
        dg_train_epochs = 10
        args.model_epochs = 1
    else:
        n_reps = args.n_reps
        n_train_diffs = args.n_train_diffs
        dg_train_epochs = 25

    if not args.train_dg:
        dg_train_epochs = 0

    if args.dg_layer_spec is None and dg_train_epochs > 0:
        dg_layers = (100, 200)
    elif args.dg_layer_spec is None:
        dg_layers = (100, 200, 300, 100)
    else:
        dg_layers = args.dg_layer_spec
        
    save_tf_models = not args.no_models
    if args.source_distr == 'uniform':
        sd = da.MultivariateUniform(true_inp_dim, (-1, 1))
    else:
        sd = None

    if args.data_generator is not None:
        dg_use = dg.FunctionalDataGenerator.load(args.data_generator)
        inp_dim = dg_use.input_dim
    elif args.use_rf_dg:
        dg_use = dg.RFDataGenerator(true_inp_dim, args.dg_dim, total_out=True,
                                    width_scaling=args.rf_width,
                                    noise=args.rf_output_noise,
                                    input_noise=args.rf_input_noise,
                                    source_distribution=sd)
    elif args.use_prf_dg:
        prf_train_epochs = 5
        dg_layers = (100, 200)
        dg_use = dg.FunctionalDataGenerator(true_inp_dim, dg_layers,
                                            args.dg_dim,
                                            source_distribution=sd,
                                            noise=.01, use_pr_reg=True)
        dg_use.fit(epochs=prf_train_epochs,
                   batch_size=args.batch_size)
    else:
        dg_use = None

    hide_print = not args.show_prints

    if args.use_tanh:
        act_func = tf.nn.tanh
    else:
        act_func = tf.nn.relu

    if args.layer_spec is None:
        layer_spec = ((50,), (50,), (50,))
    else:
        layer_spec = tuple((i,) for i in args.layer_spec)

    betas = args.betas
    model_kinds = list(ft.partial(dd.BetaVAE, beta=b*args.beta_mult,
                                  dropout_rate=args.dropout,
                                  full_cov=args.full_cov, act_func=act_func)
                       for b in betas)
    
    use_mp = not args.no_multiprocessing
    out = dc.test_generalization_new(dg_use=dg_use, est_inp_dim=est_inp_dim,
                                     inp_dim=true_inp_dim,
                                     layer_spec=layer_spec,
                                     hide_print=hide_print,
                                     dg_train_epochs=dg_train_epochs,
                                     n_reps=n_reps, model_kinds=model_kinds,
                                     use_mp=use_mp, models_n_diffs=n_train_diffs,
                                     models_n_bounds=args.n_train_bounds,
                                     dg_dim=args.dg_dim,
                                     model_batch_size=args.batch_size,
                                     model_n_epochs=args.model_epochs,
                                     distr_type=args.source_distr,
                                     generate_data=not args.no_data)
    dg, (models, th), (p, c), (lrs, scrs, sims), gd = out
    
    da.save_generalization_output(args.output_folder, dg, models, th, p, c,
                                  lrs, (scrs, sims), gd, save_args=args,
                                  save_tf_models=save_tf_models)
