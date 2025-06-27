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
import disentangled.regularizer as dr
import disentangled.expanders as de

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
    parser.add_argument('--context_offset', default=False,
                        action='store_true',
                        help='use contextual partitions with offsets')
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
    parser.add_argument('--model_epochs', default=200, type=int,
                        help='the number of epochs to train each model for')
    parser.add_argument('--dg_train_epochs', default=5, type=int,
                        help='the number of epochs to train the data generator '
                        'for')
    parser.add_argument('--l2pr_weights', default=None, nargs=2, type=float,
                        help='the weights for L2-PR regularization')
    parser.add_argument('--l2pr_weights_mult', default=1, type=float,
                        help='the weight multiplier for L2-PR regularization')
    parser.add_argument('--l2_weight', default=None, type=float,
                        help='the weight for L2 regularization')
    parser.add_argument('--l1_weight', default=None, type=float,
                        help='the weight for L1 regularization')
    parser.add_argument('--full_reg', default=False, action='store_true',
                        help='use regularization on all layers, rather than just'
                        ' the rep layer')    
    parser.add_argument('--rep_noise', default=0, type=float,
                        help='std of noise to use in representation layer '
                        'during training')
    parser.add_argument('--no_data', default=False, action='store_true',
                        help='do not save representation samples')
    parser.add_argument('--use_tanh', default=False, action='store_true',
                        help='use tanh instead of relu transfer function')
    parser.add_argument('--use_linear_network', default=False, action='store_true')
    parser.add_argument('--layer_spec', default=None, type=int, nargs='*',
                        help='the layer sizes to use')
    parser.add_argument('--dg_layer_spec', default=None, type=int, nargs='*',
                        help='the layer sizes to use')
    parser.add_argument('--rf_width', default=4, type=float,
                        help='scaling of RFs for RF data generator')
    parser.add_argument('--rf_input_noise', default=0, type=float,
                        help='noise applied to latent variables before '
                        'they are passed to the RF functions (default 0)')
    parser.add_argument('--rf_output_noise', default=0, type=float,
                        help='noise applied to the output of the RF '
                        'functions (default 0)')
    parser.add_argument('--rf_random_widths', default=False, action='store_true',
                        help='whether to randomize RF widths')    
    parser.add_argument('--nan_salt', default=None, type=float,
                        help='probability an output is replaced with nan')
    parser.add_argument('--no_train_dg', default=False, action='store_true',
                        help='train data generator')
    parser.add_argument('--source_distr', default='normal', type=str,
                        help='distribution to sample from (normal or uniform)')
    parser.add_argument('--use_rbf_dg', default=False, action='store_true',
                        help='use radial basis function data generator')
    parser.add_argument('--use_periodic_dg', default=False, action='store_true',
                        help='use shift map data generator')
    parser.add_argument('--use_prf_dg', default=False, action='store_true',
                        help='use participation ratio-optimized data generator')
    parser.add_argument('--use_gp_dg', default=False, action='store_true',
                        help='use GaussianProcess data generator')
    parser.add_argument('--use_3d_dg', default=False, action='store_true',
                        help='use 3D shapes data generator')
    parser.add_argument('--use_2d_dg', default=False, action='store_true',
                        help='use 2D shapes data generator')
    parser.add_argument('--use_chairs_dg', default=False, action='store_true',
                        help='use chair data generator')
    parser.add_argument('--gp_length_scale', default=.5, type=float,
                        help='length scale for RBF kernel')
    parser.add_argument('--config_path', default=None, type=str,
                        help='path to config file to use, will override other '
                        'params')
    parser.add_argument('--use_grids_only', default=False, action='store_true',
                        help='use only grid tasks, instead of the partitions')
    parser.add_argument('--use_gp_tasks_only', default=False, action='store_true',
                        help='use only Gaussian Process tasks, instead of the '
                        'partitions')
    parser.add_argument('--gp_task_length_scale', default=.5, type=float,
                        help='length scale for Gaussian process tasks')
    parser.add_argument('--n_grids', default=0, type=int,
                        help='use n grid tasks along with the partitions')
    parser.add_argument('--n_granules', default=2, type=int,
                        help='number of grid points to use')
    parser.add_argument('--granule_sparseness', default=.5, type=float,
                        help='sparseness of granule coloring')
    parser.add_argument('--task_subset', default=None, type=int,
                        help='number of latent variables to learn for tasks')
    parser.add_argument('--use_weight_decay', default=False, action='store_true',
                        help='use AdamW optimizer to train model')
    parser.add_argument('--weight_reg_type', default='l2', type=str)
    parser.add_argument('--weight_reg_weight', default=0, type=float,
                        help='weight of L2 regularization on weights')
    parser.add_argument('--eval_intermediate', default=False,
                        action='store_true', help='run generalization analysis '
                        'on intermediate layers')
    parser.add_argument('--img_resize', default=(224, 224), nargs=2,
                        type=tuple, help='size to resize images to')
    default_pre_model = ('https://tfhub.dev/google/imagenet/'
                         'mobilenet_v3_small_100_224/feature_vector/5')
    parser.add_argument('--img_pre_net', default=default_pre_model, type=str,
                        help='string giving url for image model to use')
    parser.add_argument('--categ_frac', default=None,
                        type=float, help='if a number is supplied, it will be '
                        'used as the fraction of the unique entries of the '
                        'categorical variable used for training; 1 - the number '
                        'will be used for testing')
    parser.add_argument('--extrapolation', default=False, action='store_true',
                        help='if flagged and categ_frac is not None, then will '
                        'train decoder on old imgs and test on new images')
    parser.add_argument('--no_compute_untrained', default=False,
                        action='store_true', help='do not compute generalization '
                        'performance for untrained variables')
    parser.add_argument('--n_chairs', default=20, type=int,
                        help='number of chairs to load')
    parser.add_argument('--readout_bias_reg', default=0, type=float,
                        help='strength of the L2 norm on the bias for the '
                        'readout units')
    parser.add_argument('--training_samples_seq', default=None, type=float,
                        nargs=3, help='different numbers of training examples '
                        'to use')
    parser.add_argument('--use_early_stopping', default=False,
                        action='store_true',
                        help='whether to use early stopping')
    parser.add_argument('--early_stopping_field',
                        default='val_class_branch_loss',
                        type=str, help='history field to use to decide early '
                        'stopping')
    parser.add_argument('--gp_test_task_length_scale', default=None, type=float,
                        help='the length scale for test tasks to use')
    parser.add_argument('--use_random_rfs', default=False, action='store_true')
    parser.add_argument('--contextual_extrapolation', default=False,
                        action='store_true')
    parser.add_argument('--use_expander_net', default=False,
                        action='store_true')                        
    parser.add_argument('--use_roll_dg', default=False,
                        action='store_true')
    parser.add_argument('--roll_period', default=.4, type=float)
    parser.add_argument('--use_self_supervised_preprocessing', default=False,
                        action='store_true')
    parser.add_argument('--ssp_n_layers', default=2, type=int)
    parser.add_argument('--ssp_view_distance', default=.1, type=float)
    parser.add_argument('--ssp_algorithm', default='simclr', type=str)
    parser.add_argument('--ssp_epochs', default=100, type=int)
    return parser

def main():
    parser = create_parser()
    args = parser.parse_args()

    if args.config_path is not None:
        config_dict = pickle.load(open(args.config_path, 'rb'))
        args = u.merge_params_dict(args, config_dict)

    partitions = args.partitions
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
    if args.no_train_dg:
        dg_train_epochs = 0

    save_tf_models = not args.no_models
    if args.source_distr == 'uniform':
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

    dg_eval = None
    pre_net = False
    if args.data_generator is not None:
        dg_use = dg.FunctionalDataGenerator.load(args.data_generator)
        inp_dim = dg_use.input_dim
    elif args.use_rf_dg:
        dg_use = dg.RFDataGenerator(true_inp_dim, args.dg_dim,
                                    total_out=not args.use_random_rfs,
                                    width_scaling=args.rf_width,
                                    noise=args.rf_output_noise,
                                    input_noise=args.rf_input_noise,
                                    source_distribution=sd,
                                    random_widths=args.rf_random_widths,
                                    use_random_rfs=args.use_random_rfs)
    elif args.use_rbf_dg:
        dg_use = dg.KernelDataGenerator(true_inp_dim, None, args.dg_dim,
                                        source_distribution=sd)
    elif args.use_periodic_dg:
        dg_use = dg.ShiftMapDataGenerator(true_inp_dim, None, args.dg_dim,
                                          source_distribution=sd)
    elif args.use_prf_dg:
        prf_train_epochs = dg_train_epochs
        dg_layers = (100, 200)
        dg_use = dg.FunctionalDataGenerator(true_inp_dim, dg_layers,
                                            args.dg_dim,
                                            source_distribution=sd,
                                            noise=.01, use_pr_reg=True)
        dg_use.fit(epochs=prf_train_epochs,
                   batch_size=args.batch_size)
    elif args.use_gp_dg:
        dg_use = dg.GaussianProcessDataGenerator(
            true_inp_dim, dg_layers, args.dg_dim, source_distribution=sd,
            length_scale=args.gp_length_scale
        )
        dg_use.fit(train_samples=1000)
    elif args.use_roll_dg:
        dg_use = dg.RollDataGenerator(
            true_inp_dim, 100, args.dg_dim,
            single_period=args.roll_period,
        )
        dg_use.fit()
    elif args.use_chairs_dg:
        filter_edges = .4
        chair_file = 'disentangled/datasets/chairs_images/'

        dg_use = dg.ChairGenerator(chair_file, img_size=args.img_resize,
                                   max_load=np.inf, include_position=True,
                                   max_move=.6, filter_edges=filter_edges,
                                   pre_model=args.img_pre_net,
                                   n_unique_chairs=args.n_chairs)
        
        true_inp_dim = dg_use.input_dim
        no_learn_lvs = np.array([True, False, True, False, False])
        compute_train_lvs = True
    elif args.use_2d_dg:
        twod_file = ('disentangled/datasets/'
                     'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
        dg_use = dg.TwoDShapeGenerator(twod_file, img_size=args.img_resize,
                                       max_load=np.inf, convert_color=True,
                                       pre_model=args.img_pre_net)  
        true_inp_dim = dg_use.input_dim
        no_learn_lvs = np.array([True, False, True, False, False])
        compute_train_lvs = True
    elif args.use_3d_dg:
        threed_file = 'disentangled/datasets/3dshapes.h5'
        dg_use = dg.ThreeDShapeGenerator(threed_file, 
                                         img_size=args.img_resize,
                                         max_load=np.inf,
                                         pre_model=args.img_pre_net)   
        true_inp_dim = dg_use.input_dim
        no_learn_lvs = np.array([False, False, False, False, True, True])
        compute_train_lvs = True
    else:
        dg_use = None

    if args.use_self_supervised_preprocessing:
        ie = de.InputExpander(args.dg_dim, (args.dg_dim,)*args.ssp_n_layers,
                              args.dg_dim, (args.dg_dim,),
                              algorithm=args.ssp_algorithm)
        ie.fit(dg_use, view_distance=args.ssp_view_distance,
               pre_train_epochs=args.ssp_epochs)
        dg_use = de.ExpandedInput(dg_use, ie)

    if args.offset_distr_var == 0:
        offset_distr = None
    else:
        offset_distr = sts.norm(0, np.sqrt(args.offset_distr_var))

    if args.l2pr_weights is not None:
        reg = dr.L2PRRegularizer
        reg_weight = np.array(args.l2pr_weights)*args.l2pr_weights_mult
    elif args.l1_weight is not None:
        reg = tfk.regularizers.l1
        reg_weight = args.l1_weight
    elif args.l2_weight is not None:
        reg = tfk.regularizers.l2
        reg_weight = args.l2_weight
    else:
        reg = tfk.regularizers.l2
        reg_weight = 0

    reg_dict = {'l1':tfk.regularizers.l1,
                'l2':tfk.regularizers.l2}
    weight_reg_type = reg_dict[args.weight_reg_type]
        
    if args.use_tanh:
        act_func = tf.nn.tanh
        out_act_func = tf.nn.sigmoid
    elif args.use_linear_network:
        act_func = None
        out_act_func = None
    else:
        act_func = tf.nn.relu
        out_act_func = tf.nn.sigmoid

    if args.layer_spec is None:
        layer_spec = ((50,), (50,), (50,))
    else:
        layer_spec = tuple((i,) for i in args.layer_spec)
        
    if args.task_subset is not None:
        no_learn_lvs = np.zeros(true_inp_dim, dtype=int)
        miss_dims = true_inp_dim - args.task_subset
        no_learn_lvs[-miss_dims:] = 1
        compute_train_lvs = True
        
    hide_print = not args.show_prints
    orthog_partitions = args.use_orthog_partitions
    contextual_partitions = args.contextual_partitions
    context_offset = args.context_offset
    if args.contextual_extrapolation and not contextual_partitions:
        contextual_partitions = True
        print('setting contextual partitions')
    if args.contextual_extrapolation:
        orthog_context = True
    else:
        orthog_context = False
    if pre_net:
        net_type = dd.FlexibleDisentanglerPre
    elif args.use_expander_net:
        net_type = dd.FlexibleDisentanglerExpAE
    else:
        net_type = dd.FlexibleDisentanglerAE
    if args.nan_salt == -1:
        nan_salt = 'single'
    else:
        nan_salt = args.nan_salt
    model_kinds = list(ft.partial(
        net_type,
        true_inp_dim=true_inp_dim,
        n_partitions=p,
        contextual_partitions=contextual_partitions,
        orthog_partitions=orthog_partitions,
        offset_distr=offset_distr,
        loss_ratio=args.loss_ratio,
        no_autoenc=args.no_autoencoder,
        dropout_rate=args.dropout,
        regularizer_type=reg,
        regularizer_weight=reg_weight,
        noise=args.rep_noise,
        context_offset=context_offset,
        act_func=act_func,
        output_act=out_act_func,
        nan_salt=nan_salt,
        n_grids=args.n_grids,
        n_granules=args.n_granules,
        granule_sparseness=args.granule_sparseness,
        grid_coloring=args.use_grids_only,
        use_gp_tasks=args.use_gp_tasks_only,
        gp_task_length_scale=args.gp_task_length_scale,
        no_learn_lvs=no_learn_lvs,
        weight_reg_weight=args.weight_reg_weight,
        readout_bias_reg_str=args.readout_bias_reg,
        use_early_stopping=args.use_early_stopping,
        early_stopping_field=args.early_stopping_field,
        full_reg=args.full_reg,
        weight_reg_type=weight_reg_type,
        orthog_context=orthog_context,        
    )
                       for p in partitions)
        
    use_mp = not args.no_multiprocessing
    compute_untrained = not args.no_compute_untrained
    out = dc.test_generalization_new(
        dg_use=dg_use, est_inp_dim=est_inp_dim, inp_dim=true_inp_dim,
        hide_print=hide_print, dg_train_epochs=dg_train_epochs,
        n_reps=n_reps, model_kinds=model_kinds, use_mp=use_mp,
        models_n_diffs=n_train_diffs, models_n_bounds=args.n_train_bounds,
        dg_dim=args.dg_dim, model_batch_size=args.batch_size,
        model_n_epochs=args.model_epochs, layer_spec=layer_spec,
        generate_data=not args.no_data, distr_type=args.source_distr,
        compute_trained_lvs=compute_train_lvs, plot=False,
        compute_untrained=compute_untrained,
        categ_var=args.categ_frac,
        extrapolate_test=args.extrapolation,
        evaluate_intermediate=args.eval_intermediate,
        samples_seq=args.training_samples_seq,
        use_test_gp_tasks=args.gp_test_task_length_scale,
        contextual_extrapolation=args.contextual_extrapolation,
    )
    dg, (models, th), (p, c), (lrs, scrs, sims), gd, other = out

    da.save_generalization_output(args.output_folder, dg, models, th, p, c,
                                  lrs, (scrs, sims), gd, other,
                                  save_args=args,
                                  save_tf_models=save_tf_models)
