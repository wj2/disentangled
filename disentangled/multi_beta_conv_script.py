import argparse
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft
import tensorflow as tf
import tensorflow.keras as tfk

import disentangled.characterization as dc
import disentangled.auxiliary as da
import disentangled.disentanglers as dd
import disentangled.data_generation as dg

def create_parser():
    parser = argparse.ArgumentParser(description='fit several autoencoders')
    parser.add_argument('-o', '--output_folder', default='results-n', type=str,
                        help='folder to save the output in')
    parser.add_argument('--img_folder',
                        default='../data/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz',
                        type=str, help='folder with input images')
    parser.add_argument('--img_resize', default=(32, 32), nargs=2,
                        type=tuple, help='size to resize images to')    
    parser.add_argument('--max_imgs', default=np.inf, 
                        type=int, help='number of images to load')    
    parser.add_argument('--max_shift', default=.6, 
                        type=float, help='max amount to shift chair position')
    parser.add_argument('--no_position', default=False, action='store_true',
                        help='do not manipulate chair position')    
    parser.add_argument('-s', '--no_multiprocessing', default=False,
                        action='store_true',
                        help='path to trained data generator')
    parser.add_argument('-d', '--data_generator', type=str, default=None,
                        help='file with saved data generator to use')
    parser.add_argument('-l', '--latent_dims', default=None, type=int,
                        help='number of dimensions to use in latent layer')
    parser.add_argument('-i', '--input_dims', default=2, type=int,
                        help='true number of dimensions in input')
    parser.add_argument('-b', '--betas', nargs='*', type=int,
                        help='betas to run models with')
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
    parser.add_argument('-m', '--no_models', default=False, action='store_true',
                        help='do not store tensorflow models')
    parser.add_argument('--test', default=False, action='store_true',
                        help='run minimal code to test that this script works')
    parser.add_argument('--show_prints', default=False,
                        action='store_true',
                        help='print training information for disentangler '
                        'models')
    parser.add_argument('--dg_dim', default=200, type=int,
                        help='dimensionality of the data generator')
    parser.add_argument('--batch_size', default=30, type=int,
                        help='batch size to use for training model')
    parser.add_argument('--dropout', default=0, type=float,
                        help='amount of dropout to include during model '
                        'training')
    parser.add_argument('--model_epochs', default=200, type=int,
                        help='the number of epochs to train each model for')
    parser.add_argument('--dg_train_epochs', default=25, type=int,
                        help='the number of epochs to train the data generator '
                        'for')
    parser.add_argument('--rep_noise', default=0, type=float,
                        help='std of noise to use in representation layer '
                        'during training')
    parser.add_argument('--no_data', default=False, action='store_true',
                        help='do not save representation samples')
    parser.add_argument('--use_tanh', default=False, action='store_true',
                        help='use tanh instead of relu transfer function')
    parser.add_argument('--layer_spec', default=None, type=int, nargs='*',
                        help='the layer sizes to use')
    parser.add_argument('--source_distr', default='normal', type=str,
                        help='distribution to sample from (normal or uniform)')
    parser.add_argument('--config_path', default=None, type=str,
                        help='path to config file to use, will override other '
                        'params')
    parser.add_argument('--full_cov', default=False, action='store_true',
                        help='fit the full covariance matrix')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    if args.config_path is not None:
        config_dict = pickle.load(open(args.config_path, 'rb'))
        args = u.merge_params_dict(args, config_dict)

    betas = args.betas
    true_inp_dim = args.input_dims
    est_inp_dim = args.latent_dims
    if args.test:
        n_reps = 1
        n_train_diffs = 1
        dg_train_epochs = 1
        args.model_epochs = 1
    else:
        n_reps = args.n_reps
        n_train_diffs = args.n_train_diffs
        dg_train_epochs = args.dg_train_epochs

    save_tf_models = not args.no_models
    if args.source_distr == 'uniform':
        sd = da.MultivariateUniform(true_inp_dim, (-1, 1))
    else:
        sd = None

    dg_use = dg.TwoDShapeGenerator(args.img_folder, img_size=args.img_resize,
                                   max_load=args.max_imgs)
    inp_dim = dg_use.input_dim
        
    if args.use_tanh:
        act_func = tf.nn.tanh
    else:
        act_func = tf.nn.relu

    if args.layer_spec is None:
        layer_spec = ((128, 2, 2), (128, 2, 2), (256,),
                      (128,), (128,))
        layer_spec = ((128, 2, 2), (128, 2, 2), (512,),
                      (256,), (128,), (128,))
    else:
        layer_spec = tuple((i,) for i in args.layer_spec)
        
    hide_print = not args.show_prints

    model_kinds = list(ft.partial(dd.BetaVAEConv, beta=b*args.beta_mult,
                                  dropout_rate=args.dropout,
                                  output_distrib='binary',
                                  full_cov=args.full_cov,
                                  act_func=act_func)
                       for b in betas)
            
    use_mp = not args.no_multiprocessing
    out = dc.test_generalization_new(dg_use=dg_use, est_inp_dim=est_inp_dim,
                                     inp_dim=true_inp_dim, 
                                     hide_print=hide_print,
                                     dg_train_epochs=dg_train_epochs,
                                     n_reps=n_reps, model_kinds=model_kinds,
                                     use_mp=use_mp, models_n_diffs=n_train_diffs,
                                     models_n_bounds=args.n_train_bounds,
                                     layer_spec=layer_spec,
                                     model_batch_size=args.batch_size,
                                     model_n_epochs=args.model_epochs,
                                     plot=False, gpu_samples=True)
    dg, (models, th), (p, c), (lrs, scrs, sims), gd = out

    da.save_generalization_output(args.output_folder, dg, models, th, p, c,
                                  lrs, (scrs, sims), gd, save_args=args,
                                  save_tf_models=save_tf_models)
