import argparse
import disentangled.model as dm
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft

def create_parser():
    parser = argparse.ArgumentParser(description='fit several autoencoders')
    parser.add_argument('-o', '--output_file', default='results.pkl', type=str,
                        help='file to save the output in')
    parser.add_argument('-s', '--no_multiprocessing', default=False,
                        action='store_true',
                        help='path to trained data generator')
    parser.add_argument('-l', '--latent_dims', default=None, type=int,
                        help='number of dimensions to use in latent layer')
    parser.add_argument('-b', '--betas', nargs='*', type=float,
                        help='betas to fit models for')
    parser.add_argument('-c', '--save_datagenerator', default=None,
                        type=str, help='path to save data generator at')
    parser.add_argument('-n', '--n_reps', default=5,
                        type=int, help='number of times to train each model')
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    betas = args.betas
    model_kinds = list(ft.partial(dm.BetaVAE, beta=b) for b in betas)
    est_inp_dim = args.latent_dims
    n_reps = args.n_reps

    use_mp = not args.no_multiprocessing
    out = dm.test_generalization(dg=None, est_inp_dim=est_inp_dim, n_reps=n_reps,
                                 model_kinds=model_kinds, use_mp=use_mp,
                                 n_train_diffs=1)
    
    pickle.dump(out, open(args.output_file, 'wb'))
