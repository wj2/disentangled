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
    parser.add_argument('-b', '--betas', nargs='*', type=float,
                        help='betas to fit models for')
    parser.add_argument('-c', '--save_datagenerator', default=None,
                        type=str, help='path to save data generator at')
    parser.add_argument('-n', '--n_reps', default=5,
                        type=int, help='number of times to train each model')
    parser.add_argument('-t', '--n_train_diffs', default=6, type=int,
                        help='number of different training data sample sizes '
                        'to use')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    betas = args.betas
    model_kinds = list(ft.partial(dd.BetaVAE, beta=b) for b in betas)
    est_inp_dim = args.latent_dims
    n_reps = args.n_reps
    n_train_diffs = args.n_train_diffs

    if args.data_generator is not None:
        dg_use = dg.FunctionalDataGenerator.load(args.data_generator)
    else:
        dg_use = None

    use_mp = not args.no_multiprocessing
    out = dc.test_generalization(dg=dg_use, est_inp_dim=est_inp_dim,
                                 n_reps=n_reps, model_kinds=model_kinds,
                                 use_mp=use_mp, n_train_diffs=n_train_diffs)
    dg, (models, th), (p, c) = out

    da.save_generalization_output(args.output_folder, dg, models, th, p, c)
    
