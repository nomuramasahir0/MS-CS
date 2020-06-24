import argparse
import numpy as np

import json
import os

from utils.util import get_path_with_time, load_class
import distutils.util
import sys
from shutil import copyfile


def get_args(rundeck=False):
    parser = make_parser()
    args = parser.parse_args()
    params_json_path = "params.json"
    if not rundeck:
        args, cont_spaces = read_info_from_json(args, params_json_path)
        args.obj_func = get_obj_func(args)
        from bedrock.spaces import Space

        args.space = Space(continuous_space=cont_spaces)
        args.obj_func.set_space(args.space)
    else:
        args.obj_func = None
    args.path = output_dir(
        args.dataset, args.estimator_type, args.optimizer, args.tuning_path
    )
    copyfile(params_json_path, args.path + "/hp_params.json")
    print("finish creating args:{}".format(args))
    return args


def get_adder(g):
    def f(*args, **kwargs):
        kwargs.setdefault("help", "Default %(default)s.")
        return g.add_argument(*args, **kwargs)

    return f


def add_args(subparser):
    conf = subparser.add_argument_group("basic configuration")
    c = get_adder(conf)
    c("--B", type=int, default=50, help="max number of evaluations")
    c("--optimizer", choices=["GPUCB"], default="GPUCB")
    c("--debug", action="store_true", default=False)
    c("--seed", type=int, default=np.random.randint(2 ** 32))
    c(
        "--alpha",
        type=float,
        default=0.0,
        help="for calculating alpha-relative density ratio",
    )
    c("--tuning_path", type=str, default="")

    method = subparser.add_argument_group("method of estimation")
    m = get_adder(method)
    m("--estimator-type", choices=["naive", "upper", "unbiased", "vr"], default="naive")


def make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="the dataset to run on")

    def add_subparser(name, **kwargs):
        subparser = subparsers.add_parser(name, **kwargs)
        subparser.set_defaults(dataset=name)
        data = subparser.add_argument_group("data parameters")
        add_args(subparser)
        return data, get_adder(data)

    mssynthetic, d = add_subparser("MSSynthetic1D")
    d("--problem", type=str, default="MSSynthetic1D")
    d("--num_sources", type=int, default=10)
    d("--source_mu_bound", type=float, default=2.0)
    d("--target_mu_bound", type=float, default=2.0)
    d("--coef_a", type=float, default=0.7)
    d("--coef_b", type=float, default=0.3)
    d("--sigma", type=float, default=1.0)
    d("--num_data", type=int, default=500)

    parkinson_svm, d = add_subparser("ParkinsonSVM")
    d("--problem", type=str, default="Parkinson")
    d("--model", type=str, default="SVR")
    d("--data_dir", type=str, default="parkinson")
    d("--target_name", type=str, default="patient_28.npy")
    d("--ratio_validation", type=float, required=True)
    d("--ratio_dre", type=float, default=None)
    d("--is_source_concat_for_naive", type=int, choices=[0, 1], default=None)
    d("--is_separate_source_dens", type=int, choices=[0, 1], default=None)

    parkinson_svm, d = add_subparser("GvHDLGBM")
    d("--problem", type=str, default="GvHD")
    d("--model", type=str, default="LGBMBinaryClassifier")
    d("--data_dir", type=str, default="gvhd")
    d("--target_name", type=str, default="task_0.npy")
    d("--ratio_validation", type=float, required=True)
    d("--ratio_dre", type=float, default=None)
    d("--is_source_concat_for_naive", type=int, choices=[0, 1], default=None)
    d("--is_separate_source_dens", type=int, choices=[0, 1], default=None)

    return parser


def convert_my_space(args, json_params):
    from bedrock.spaces import Uniform, QUniform, PowUniform

    cont_spaces = []
    if "dim" in json_params.keys():
        args.dim = json_params["dim"]
        v = json_params["continuous_space"]["theta"]
        low, high = v[1], v[2]
        keys = ["x_" + str(i) for i in range(args.dim)]
        if v[0] == "Uniform":
            for i in range(args.dim):
                cont_spaces.append(Uniform(keys[i], low, high))
        elif v[0] == "QUniform":
            for i in range(args.dim):
                cont_spaces.append(QUniform(keys[i], low, high))
        elif v[0] == "PowUniform":
            for i in range(args.dim):
                cont_spaces.append(PowUniform(keys[i], low, high, 10))
    else:
        values = json_params["continuous_space"]
        args.dim = len(values)
        for k, v in values.items():
            if v[0] == "Uniform":
                cont_spaces.append(Uniform(k, v[1], v[2]))
            elif v[0] == "QUniform":
                cont_spaces.append(QUniform(k, v[1], v[2]))
            elif v[0] == "PowUniform":
                cont_spaces.append(PowUniform(k, v[1], v[2], 10))
    return args, cont_spaces


def read_info_from_json(args, params_json_path):
    with open(params_json_path) as f:
        json_params = json.load(f)[args.dataset]
        args.fn = load_class("functions.%s.%s" % (json_params["fn"], args.problem))
        if "ylim" in json_params.keys():
            args.ylim = [json_params["ylim"][0], json_params["ylim"][1]]
        if "scatter_ylim" in json_params.keys():
            args.scatter_ylim = [
                json_params["scatter_ylim"][0],
                json_params["scatter_ylim"][1],
            ]
        if "flag_yscale_log" in json_params.keys():
            args.flag_yscale_log = distutils.util.strtobool(
                json_params["flag_yscale_log"]
            )

        if args.optimizer == "GPUCB":
            args = convert_my_space(args, json_params)
        else:
            print("[read_info_from_json] optimizer does not exist.")
            sys.exit(1)

    return args


def output_dir(obj_func_name, estimator_type, optimizer, tuning_path):
    path = get_path_with_time(obj_func_name, estimator_type, optimizer, tuning_path)
    if not os.path.isdir(path):
        os.makedirs(path)
    return path


def get_obj_func(args):
    if args.problem == "MSSynthetic1D":
        from functions.ms_synthetic import MSSynthetic1D

        return MSSynthetic1D(args)
    elif args.problem == "Parkinson":
        from functions.parkinson import Parkinson

        return Parkinson(args)
    elif args.problem == "GvHD":
        from functions.gvhd import GvHD

        return GvHD(args)
    else:
        raise NotImplementedError
