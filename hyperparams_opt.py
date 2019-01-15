import os
import pickle
import traceback
import datetime

from keras.callbacks import ModelCheckpoint, Callback
from keras.utils.training_utils import multi_gpu_model

from hyperopt import hp, tpe
from hyperopt.base import STATUS_OK, STATUS_FAIL, Trials
from hyperopt.fmin import fmin
import xarray as xr
import numpy as np

import tensorflow as tf


import model_factory
from data_processing import ds_to_data
from os.path import join
from functools import partial
import json
import sys

import ray
import ray.tune as tune
from ray.tune.registry import register_trainable
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.util import get_pinned_object, pin_in_object_store
import glob
from ray.tune.suggest.basic_variant import BasicVariantGenerator
import copy


MODEL_OUPTPUT_DIR = "/scratch/golduser/esp_fitting/ai"
EMOLECULES_DS_FILE = "/home/golduser/bin/scripts/development/esp_fit_ai/out/emolecules.nc"
CHEMBL_DS_FILE = "/home/golduser/bin/scripts/development/esp_fit_ai/out/chembl.nc"

global INPUT_DATA

def mkdir_p(path, clean=False):
    if os.path.exists(path) and clean:
        rmtree(path)
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


class TuneCallback(Callback):
    def __init__(self, reporter, logs={}):
        self.reporter = reporter

    def on_epoch_end(self, epoch, logs={}):
        self.reporter(val_loss=logs["val_loss"], training_loss=logs["loss"])


def read_ds(ds_file, output_dir, test_ds_file=None):
    ds = xr.open_dataset(ds_file)
    norm_params_outfile = join(output_dir, "norm_params.pkl")
    X, D, Y, train_mask = ds_to_data(ds, norm_param_outfile=norm_params_outfile)

    if test_ds_file is None:
        train_X = X[train_mask]
        test_X = X[~train_mask]

        train_D = D[train_mask]
        test_D = D[~train_mask]

        train_Y = Y[train_mask]
        test_Y = Y[~train_mask]
    else:
        train_X, train_D, train_Y = X, D, Y
        test_X, test_D, test_Y, _ = ds_to_data(xr.open_dataset(test_ds_file), norm_params_infile=norm_params_outfile)

    input_data = {"train": {"X": train_X, "D": train_D, "Y": train_Y},
                  "test": {"X": test_X, "D": test_D, "Y": test_Y}}
    return input_data


EMOLECULES_SPACE = {
    "lr": tune.grid_search([0.005, 0.0025, 0.01]),
    "decay": tune.grid_search([0.00001, 0.000025, 0.00005]),
    "width": tune.grid_search([100, 120, 160]),
    "dropout_p": tune.grid_search([None, 0.1, 0.25]),
    "activation": tune.grid_search(["relu", "sigmoid"]),
    "n_conv_layers": tune.grid_search([4, 5, 6]),
    "neigh_wts": tune.grid_search(["single", "all"]),
    "batch_size": tune.grid_search([128]),
    "epochs": tune.grid_search([1000])
}


CHEMBL_SPACE = {
    "lr": tune.grid_search([0.0025]),
    "decay": tune.grid_search([0.00005]),
    "width": tune.grid_search([160]),
    "dropout_p": tune.grid_search([None, 0.1]),
    "activation": tune.grid_search(["relu"]),
    "n_conv_layers": tune.grid_search([4, 5, 6]),
    "neigh_wts": tune.grid_search(["single", "all"]),
    "batch_size": tune.grid_search([128]),
    "epochs": tune.grid_search([1000])
}


def opt_hps_ray(ds_file, output_dir, space=EMOLECULES_SPACE, max_evals=100, test_ds_file=None):
    space = copy.deepcopy(space)
    mkdir_p(os.path.join(output_dir, "summary"))
    mkdir_p(os.path.join(output_dir, "ray_results"))
    mkdir_p(os.path.join(output_dir, "models"))
    space["output_dir"] = tune.grid_search([output_dir])
    ray.init()
    global INPUT_DATA
    INPUT_DATA = pin_in_object_store(read_ds(ds_file, os.path.join(output_dir, "summary"), test_ds_file=test_ds_file))
    register_trainable("exp", fit_model)
    trials = tune.run_experiments({"esp_fitting_hp_opt": {
        "local_dir": os.path.join(output_dir, "ray_results"),
        "run": "exp",
        "trial_resources": {"cpu": 0, "gpu": 1},
        "config": space
    }})
    return trials


def fit_model(space, reporter, input_data=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, ray.get_gpu_ids()))
    if input_data is None:
        input_data = get_pinned_object(INPUT_DATA)

    # multi gpu model has an issue in saving. I have not seen a massing improvemnet in speed. perhaps the network is small and data transfer
    # between cpus and gpus outweighs the gain in speed by parallelizing
    model = model_factory.build_model(input_data["train"]["X"].shape[-1], input_data["train"]["Y"].shape[-1], space, n_gpus=None)

    log_dir = os.path.join(space["output_dir"], "models", str(datetime.datetime.now()).replace(" ", "_"))
    mkdir_p(log_dir)

    callbacks = [ModelCheckpoint(os.path.join(log_dir, "model.h5"), save_best_only=True),
                 TuneCallback(reporter)]

    try:
        model_hist = model.fit([input_data["train"]["X"], input_data["train"]["D"]], [input_data["train"]["Y"]],
                               validation_data=[[input_data["test"]["X"], input_data["test"]["D"]], [input_data["test"]["Y"]]],
                               epochs=space["epochs"], batch_size=space["batch_size"], callbacks=callbacks)

        loss = min(model_hist.history["val_loss"])
        status = STATUS_OK
        message = "success"
    except Exception as e:
        loss = None
        status = STATUS_FAIL
        message = e.message + "\n" + traceback.format_exc()
        print message
    return_dict = {"loss": loss,
                   "status": status,
                   "space": space,
                   "work_dir": log_dir,
                   "message": message,
                   "history": model_hist.history,
                   "min_val_loss": loss
                   }
    with(open(join(log_dir, "results.json"), "w")) as f:
        json.dump(return_dict, f)
    return return_dict


def trials_to_csv(trials, csv_file=None):
    f = open(csv_file, "w") if csv_file else sys.stdout
    for i, t in enumerate(trials.results):
        if not i:
            keys = sorted(t["space"].keys())
            f.write(", ".join(keys) + ", loss\n")
        for k in keys:
            if k == "dropout":
                f.write(""" + str(t["space"][k]) + "",")
            else:
                f.write(str(t["space"][k]) + ",")
        f.write(str(t["loss"]) + "\n")


def analyze_ray_runs(dirname, outfile=None):
    writer = open(outfile, "w") if outfile is not None else sys.stdout
    for i, results_file in enumerate(glob.glob(os.path.join(dirname, "*", "results.json"))):
        with open(results_file) as f:
            results = json.load(f)
        if not i:
            keys = sorted(results["space"].keys())
            writer.write(", ".join(["work_dir"] + keys) + ", loss\n")
        writer.write(""%s"," % results["work_dir"])
        for k in keys:
            if k == "dropout":
                writer.write(""" + str(results["space"][k]) + "",")
            else:
                writer.write(str(results["space"][k]) + ",")
        writer.write(str(results["loss"]) + "\n")

    if outfile is not None:
        writer.close()


if __name__ == "__main__":
    # emolecules only
    # debug only
    #os.environ["TRIALRUNNER_WALLTIME_LIMIT"] = "60"
    # opt_hps_ray(EMOLECULES_DS_FILE, os.path.join(MODEL_OUPTPUT_DIR, "10_12_2018"), max_evals=2)
    # chembl as test
    opt_hps_ray(EMOLECULES_DS_FILE,
                os.path.join(MODEL_OUPTPUT_DIR, "10_12_2018_chembl"),
                space=CHEMBL_SPACE,
                test_ds_file=CHEMBL_DS_FILE,
                max_evals=12)
