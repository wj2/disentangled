import argparse
import numpy as np
import pickle

import disentangled.figures as df

def create_parser():
    parser = argparse.ArgumentParser(description='fit several autoencoders')
    parser.add_argument('-o', '--output_folder', default='auto_dim.pkl',
                        type=str, help='folder to save the output in')
    parser.add_argument('-l', '--latent_dims', nargs='*', type=int,
                        default=None,
                        help='number of dimensions to use in latent layer')
    parser.add_argument('-i', '--input_dims', default=2, type=int,
                        help='true number of dimensions in input')
    parser.add_argument('--dg_dim', default=500, type=int,
                        help='dimensionality of the data generator')
    parser.add_argument('--model_epochs', default=300, type=int,
                        help='the number of epochs to train each model for')
    parser.add_argument('--layer_spec', default=None, type=int, nargs='*',
                        help='the layer sizes to use')
    parser.add_argument('--use_linear', default=False, action='store_true',
                        help='use linear activation function')
    return parser

if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    dim = args.input_dims
    inp = args.dg_dim
    if args.layer_spec is None:
        layers = ((200,), (100,), (100,), (50,), (50,))
    else:
        layers = list((x,) for x in args.layer_spec)
    if args.latent_dims is None:
        latents = (50, 35, 20, 15, 12, 10, 8, 5)
    else:
        latents = args.latent_dims
    epochs = args.model_epochs
    kwargs = {}
    if args.use_linear:
        kwargs['act_func'] = None

    out_dims = {}
    out_dims[dim] = df.explore_autodisentangling_layers(latents, layers, inp, dim, 
                                                        epochs=epochs, 
                                                        **kwargs)
    pickle.dump(out_dims, open(args.output_folder, 'wb'))
