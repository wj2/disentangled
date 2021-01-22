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
    parser.add_argument('-p', '--partitions', nargs='*', type=int,
                        help='numbers of partitions to models with')
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
        dg_train_epochs = 1
    else:
        n_reps = args.n_reps
        n_train_diffs = args.n_train_diffs
        dg_train_epochs = 25

    save_tf_models = not args.no_models
    if args.data_generator is not None:
        dg_use = dg.FunctionalDataGenerator.load(args.data_generator)
        inp_dim = dg_use.input_dim
    else:
        dg_use = None

    if args.offset_distr_var == 0:
        offset_distr = None
    else:
        offset_distr = sts.norm(0, np.sqrt(args.offset_distr_var))

    hide_print = not args.show_prints
    orthog_partitions = args.use_orthog_partitions
    contextual_partitions = args.contextual_partitions
    model_kinds = list(ft.partial(dd.FlexibleDisentanglerAE,
                                  true_inp_dim=true_inp_dim,
                                  n_partitions=p,
                                  contextual_partitions=contextual_partitions,
                                  orthog_partitions=orthog_partitions,
                                  offset_distr=offset_distr)
                       for p in partitions)
        
    use_mp = not args.no_multiprocessing
    out = dc.test_generalization_new(dg=dg_use, est_inp_dim=est_inp_dim,
                                     inp_dim=true_inp_dim,
                                     hide_print=hide_print,
                                     dg_train_epochs=dg_train_epochs,
                                     n_reps=n_reps, model_kinds=model_kinds,
                                     use_mp=use_mp, models_n_diffs=n_train_diffs,
                                     models_n_bounds=args.n_train_bounds)
    dg, (models, th), (p, c), (lrs, scrs, sims) = out

    da.save_generalization_output(args.output_folder, dg, models, th, p, c,
                                  lrs, (scrs, sims), save_args=args,
                                  save_tf_models=save_tf_models)
