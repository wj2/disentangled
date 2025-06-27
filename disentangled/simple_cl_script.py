import argparse
import scipy.stats as sts
import numpy as np
import pickle
import functools as ft
import tensorflow as tf

import general.utility as u

import disentangled.characterization as dc
import disentangled.auxiliary as da
import disentangled.disentanglers as dd
import disentangled.data_generation as ddg
import disentangled.regularizer as dr


def create_parser():
    parser = argparse.ArgumentParser(description="fit several autoencoders")
    parser.add_argument(
        "-o",
        "--output_folder",
        default="results-n",
        type=str,
        help="folder to save the output in",
    )
    parser.add_argument(
        "-l",
        "--latent_dims",
        default=None,
        type=int,
        help="number of dimensions to use in latent layer",
    )
    parser.add_argument(
        "-i",
        "--input_dims",
        default=2,
        type=int,
        help="true number of dimensions in input",
    )
    parser.add_argument(
        "-p",
        "--partitions",
        nargs="*",
        type=int,
        help="numbers of partitions to models with",
    )
    parser.add_argument(
        "-n",
        "--n_reps",
        default=5,
        type=int,
        help="number of times to train each model",
    )
    parser.add_argument(
        "-t",
        "--n_train_diffs",
        default=6,
        type=int,
        help="number of different training data sample sizes " "to use",
    )
    parser.add_argument(
        "--n_train_bounds",
        nargs=2,
        default=(2, 6.5),
        type=float,
        help="order of magnitudes for range to "
        "sample training differences from within (using "
        "logspace)",
    )
    parser.add_argument(
        "-m",
        "--no_models",
        default=False,
        action="store_true",
        help="do not store tensorflow models",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        help="run minimal code to test that this script works",
    )
    parser.add_argument(
        "--use_orthog_partitions",
        default=False,
        action="store_true",
        help="use only mutually orthogonal partition functions "
        "(if the number of partitions exceeds the number of "
        "dimensions, they will just be resampled",
    )
    parser.add_argument(
        "--offset_distr_var",
        default=0,
        type=float,
        help="variance of the binary partition offset "
        "distribution (will be Gaussian, default 0)",
    )
    parser.add_argument(
        "--show_prints",
        default=False,
        action="store_true",
        help="print training information for disentangler " "models",
    )
    parser.add_argument(
        "--dg_dim", default=200, type=int, help="dimensionality of the data generator"
    )
    parser.add_argument(
        "--batch_size",
        default=30,
        type=int,
        help="batch size to use for training model",
    )
    parser.add_argument(
        "--model_epochs",
        default=200,
        type=int,
        help="the number of epochs to train each model for",
    )
    parser.add_argument(
        "--dg_train_epochs",
        default=5,
        type=int,
        help="the number of epochs to train the data generator " "for",
    )
    parser.add_argument(
        "--l2pr_weights",
        default=(0, .01),
        nargs=2,
        type=float,
        help="the weights for L2-PR regularization",
    )
    parser.add_argument(
        "--l2pr_weights_mult",
        default=1,
        type=float,
        help="the weight multiplier for L2-PR regularization",
    )
    parser.add_argument(
        "--l2_weight", default=None, type=float, help="the weight for L2 regularization"
    )
    parser.add_argument(
        "--l1_weight", default=None, type=float, help="the weight for L1 regularization"
    )
    parser.add_argument(
        "--layer_spec", default=None, type=int, nargs="*", help="the layer sizes to use"
    )
    parser.add_argument(
        "--dg_layer_spec",
        default=None,
        type=int,
        nargs="*",
        help="the layer sizes to use",
    )
    parser.add_argument(
        "--nan_salt",
        default=None,
        type=float,
        help="probability an output is replaced with nan",
    )
    parser.add_argument(
        "--source_distr",
        default="normal",
        type=str,
        help="distribution to sample from (normal or uniform)",
    )
    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        help="path to config file to use, will override other " "params",
    )
    parser.add_argument(
        "--eval_intermediate",
        default=False,
        action="store_true",
        help="run generalization analysis " "on intermediate layers",
    )
    parser.add_argument(
        "--use_early_stopping",
        default=False,
        action="store_true",
        help="whether to use early stopping",
    )
    parser.add_argument(
        "--early_stopping_field",
        default="val_class_branch_loss",
        type=str,
        help="history field to use to decide early " "stopping",
    )
    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.config_path is not None:
        config_dict = pickle.load(open(args.config_path, "rb"))
        args = u.merge_params_dict(args, config_dict)

    partitions = args.partitions
    true_inp_dim = args.input_dims
    est_inp_dim = args.latent_dims
    if args.test:
        n_reps = 1
        n_train_diffs = 1
        dg_train_epochs = 0
        args.model_epochs = 1
    else:
        n_reps = args.n_reps
        n_train_diffs = args.n_train_diffs
        dg_train_epochs = args.dg_train_epochs

    save_tf_models = not args.no_models
    if args.source_distr == "uniform":
        sd = da.MultivariateUniform(true_inp_dim, (-1, 1))
    else:
        sd = None

    if args.dg_layer_spec is None and dg_train_epochs > 0:
        dg_layers = (100, 200)
    elif args.dg_layer_spec is None:
        dg_layers = (100, 200, 300, 100)
    else:
        dg_layers = args.dg_layer_spec

    no_learn_lvs = None
    compute_train_lvs = False

    prf_train_epochs = dg_train_epochs
    dg_layers = (100, 200)
    dg_use = ddg.FunctionalDataGenerator(
        true_inp_dim,
        dg_layers,
        args.dg_dim,
        source_distribution=sd,
        noise=0.01,
        use_pr_reg=True,
    )
    dg_use.fit(epochs=prf_train_epochs, batch_size=args.batch_size)

    if args.offset_distr_var == 0:
        offset_distr = None
    else:
        offset_distr = sts.norm(0, np.sqrt(args.offset_distr_var))

    reg = dr.L2PRRegularizer
    reg_weight = np.array(args.l2pr_weights) * args.l2pr_weights_mult

    act_func = tf.nn.relu
    out_act_func = tf.nn.sigmoid

    if args.layer_spec is None:
        layer_spec = ((50,), (50,), (50,))
    else:
        layer_spec = tuple((i,) for i in args.layer_spec)

    hide_print = not args.show_prints
    orthog_partitions = args.use_orthog_partitions
    net_type = dd.FlexibleDisentanglerAE
    if args.nan_salt == -1:
        nan_salt = "single"
    else:
        nan_salt = args.nan_salt
    model_kinds = list(
        ft.partial(
            net_type,
            true_inp_dim=true_inp_dim,
            n_partitions=p,
            orthog_partitions=orthog_partitions,
            offset_distr=offset_distr,
            no_autoenc=True,
            regularizer_type=reg,
            regularizer_weight=reg_weight,
            act_func=act_func,
            output_act=out_act_func,
            nan_salt=nan_salt,
            no_learn_lvs=no_learn_lvs,
            use_early_stopping=args.use_early_stopping,
            early_stopping_field=args.early_stopping_field,
        )
        for p in partitions
    )

    out = dc.test_generalization_new(
        dg_use=dg_use,
        est_inp_dim=est_inp_dim,
        inp_dim=true_inp_dim,
        hide_print=hide_print,
        dg_train_epochs=dg_train_epochs,
        n_reps=n_reps,
        model_kinds=model_kinds,
        models_n_diffs=n_train_diffs,
        models_n_bounds=args.n_train_bounds,
        dg_dim=args.dg_dim,
        model_batch_size=args.batch_size,
        model_n_epochs=args.model_epochs,
        layer_spec=layer_spec,
        generate_data=True,
        distr_type=args.source_distr,
        compute_trained_lvs=compute_train_lvs,
        plot=False,
        evaluate_intermediate=args.eval_intermediate,
    )
    dg, (models, th), (p, c), (lrs, scrs, sims), gd, other = out

    da.save_generalization_output(
        args.output_folder,
        dg,
        models,
        th,
        p,
        c,
        lrs,
        (scrs, sims),
        gd,
        other,
        save_args=args,
        save_tf_models=save_tf_models,
    )
