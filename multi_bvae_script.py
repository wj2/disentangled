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
    parser.add_argument('-d', '--data_generator', default=None, type=str,
                        help='path to trained data generator')
    parser.add_argument('-s', '--no_multiprocessing', default=False,
                        action='store_true',
                        help='path to trained data generator')
    
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    betas = (0, 1, 2, 4, 5)
    model_kinds = list(ft.partial(dm.BetaVAE, beta=b) for b in betas)
    est_inp_dim = 5

    if args.data_generator is not None:
        dg_p = pickle.load(args.data_generator)
    else:
        dg = None

    use_mp = not args.no_multiprocessing
    out = dm.test_generalization(dg=dg, est_inp_dim=est_inp_dim,
                                 model_kinds=model_kinds, use_mp=use_mp)
    
    pickle.dump(out, open(args.output_file, 'wb'))
