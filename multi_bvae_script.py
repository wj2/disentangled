import argparse
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft

import disentangled.characterization as dc
import disentangled.aux as da
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
    parser.add_argument('-m', '--no_models', default=False, action='store_true',
                        help='do not store tensorflow models')
    parser.add_argument('--test', default=False, action='store_true',
                        help='run minimal code to test that this script works')
    parser.add_argument('--use_orthog_partitions', default=False,
                        action='store_true',
                        help='use only mutually orthogonal partition functions '
                        '(if the number of partitions exceeds the number of '
                        'dimensions, they will just be resampled')
    parser.add_argument('--offset_distr_var', default=0, type=float,
                        help='variance of the binary partition offset '
                        'distribution (will be Gaussian, default 0)')
    parser.add_argument('--show_prints', default=False,
                        action='store_true',
                        help='print training information for disentangler '
                        'models')
    parser.add_argument('--contextual_partitions', default=False,
                        action='store_true',
                        help='use contextual partitions')
    parser.add_argument('--use_rf_dg', default=False,
                        action='store_true',
                        help='use an RF-based data generator')
    parser.add_argument('--dg_dim', default=200, type=int,
                        help='dimensionality of the data generator')
    parser.add_argument('--batch_size', default=30, type=int,
                        help='batch size to use for training model')
    parser.add_argument('--loss_ratio', default=10, type=float,
                        help='the ratio between autoencoder loss/classifier '
                        'loss')
    parser.add_argument('--no_autoencoder', default=False, action='store_true',
                        help='construct models with no autoencoder component')
    parser.add_argument('--dropout', default=0, type=float,
                        help='amount of dropout to include during model '
                        'training')
    parser.add_argument('--model_epochs', default=60, type=int,
                        help='the number of epochs to train each model for')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    partitions = args.partitions
    true_inp_dim = args.input_dims
    est_inp_dim = args.latent_dims
    if args.test:
        n_reps = 1
        n_train_diffs = 1
        dg_train_epochs = 10
    else:
        n_reps = args.n_reps
        n_train_diffs = args.n_train_diffs
        dg_train_epochs = 25

    save_tf_models = not args.no_models
    if args.data_generator is not None:
        dg_use = dg.FunctionalDataGenerator.load(args.data_generator)
        inp_dim = dg_use.input_dim
    elif args.use_rf_dg:
        dg_use = dg.RFDataGenerator(true_inp_dim, args.dg_dim, total_out=True)
    else:
        dg_use = None

    hide_print = not args.show_prints
        
    betas = args.betas
    model_kinds = list(ft.partial(dd.BetaVAE, beta=b*args.beta_mult,
                                  dropout_rate=args.dropout)
                       for b in betas)

    use_mp = not args.no_multiprocessing
    out = dc.test_generalization_new(dg_use=dg_use, est_inp_dim=est_inp_dim,
                                     inp_dim=true_inp_dim,
                                     hide_print=hide_print,
                                     dg_train_epochs=dg_train_epochs,
                                     n_reps=n_reps, model_kinds=model_kinds,
                                     use_mp=use_mp, models_n_diffs=n_train_diffs,
                                     models_n_bounds=args.n_train_bounds,
                                     dg_dim=args.dg_dim,
                                     model_batch_size=args.batch_size,
                                     model_n_epochs=args.model_epochs)
    dg, (models, th), (p, c), (lrs, scrs, sims), gd = out

    da.save_generalization_output(args.output_folder, dg, models, th, p, c,
                                  lrs, (scrs, sims), gd, save_args=args,
                                  save_tf_models=save_tf_models)


    use_mp = not args.no_multiprocessing
    out = dc.test_generalization_new(dg=dg_use, est_inp_dim=est_inp_dim,
                                     inp_dim=true_inp_dim,
                                     dg_train_epochs=dg_train_epochs,
                                     n_reps=n_reps, model_kinds=model_kinds,
                                     use_mp=use_mp, models_n_diffs=n_train_diffs,
                                     models_n_bounds=args.n_train_bounds)
    dg, (models, th), (p, c), (lrs, scrs, sims) = out

    da.save_generalization_output(args.output_folder, dg, models, th, p, c,
                                  lrs, (scrs, sims),
                                  save_tf_models=save_tf_models)
