from typing import Tuple
import pickle

import pandas as pd
import numpy as np
from numpy import ndarray

from keras import backend
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import rambo_master
from train_val_set import TrainSet, ValSet
from params.init_nets_params import InitNetsParams, BoostType
from rambo_master import RamboMaster

dataset_dir = "/home/anton/gitrepos/ExploringSimplicity/Datasets/"


def load_train_test_datasets(dataset_name: str):
    df_train = pd.read_csv(dataset_dir + dataset_name + "/" + "train.csv", header=None)
    df_test = pd.read_csv(dataset_dir + dataset_name + "/" + "test.csv", header=None)

    return df_train, df_test


def titanic_params(
    nr_samples: int, nr_feats: int, nr_classes: int
) -> Tuple[InitNetsParams, float]:
    boost_depth = 1

    sample_bstrap_perc = 0.64
    stratify_bstrap = False
    feat_bstrap_range = (0.5, 0.8)
    hid_layers_ranges = [
        [(5, 312), (5, 312), (5, 312), (5, 312)],
        [(5, 312), (5, 312), (5, 312), (5, 312)]
    ]
    nr_layer_probs = [0.5, 0.5]
    metric_range = (0.987, 0.99)
    nr_nets = 16

    watch_stop_val_prob = 1.0
    batch_size = 570
    swap_noise_range = (0.05, 0.15)
    swap_noise_prob = 0.0
    dropout_range = (0.05, 0.27)
    dropout_prob = 0.0

    max_epochs = 750
    hyper_sync_perc = 1.0

    init_params = InitNetsParams(
        sample_bstrap_perc=sample_bstrap_perc,
        stratify_bstrap=stratify_bstrap,
        feat_bstrap_range=feat_bstrap_range,
        hid_layer_ranges=hid_layers_ranges,
        nr_layer_probs=nr_layer_probs,
        metric_range=metric_range,
        nr_classes=nr_classes,
        nr_feats=nr_feats,
        nr_samples=nr_samples,
        max_epochs=max_epochs,
        nr_nets=nr_nets,
        boost_depth=boost_depth,
        boost_type=BoostType.fifty_fifty,
        watch_stop_val_prob=watch_stop_val_prob,
        batch_size=batch_size,
        activation="tanh",
        swap_noise_range=swap_noise_range,
        swap_noise_prob=swap_noise_prob,
        dropout_range=dropout_range,
        dropout_prob=dropout_prob,
    )

    return init_params, hyper_sync_perc


def train_cross_model(x: ndarray, y: ndarray, dataset_name: str):

    nr_feats = x.shape[1]
    nr_classes = np.unique(y).size
    nr_samples = x.shape[0]

    param_func_dict = {"titanic": titanic_params}

    param_func = param_func_dict[dataset_name]

    init_params, hyper_sync_perc = param_func(nr_samples, nr_feats, nr_classes)

    # cv_scores = []
    # train_cv_scores = []
    # ret_model = None
    counter = 0
    acc_score = 0
    # test_scores = []

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf.get_n_splits(x, y)
    #
    for train_index, val_index in skf.split(x, y):
        counter = counter + 1
        if counter < 2:
            print("SKIPPING LOOP")
            continue
        else:
            print("INSIDE LOOP")

        rambo_model: RamboMaster = RamboMaster(
            nr_classes,
            last_activation="softmax",
            boost_depth=init_params.boost_depth,
            name="ensemble",
            verbose=False,
            dataset_name=dataset_name,
        )

        x_train, x_val = x[train_index], x[val_index]
        y_train, y_val = y[train_index], y[val_index]
        train_set = TrainSet(x_train, y_train)
        val_set = ValSet(x_val, y_val)

        rambo_model.fit_with_hyperopt(
            train_set,
            init_params,
            hyper_sync_perc=hyper_sync_perc,
            nr_nets=init_params.nr_nets,
            nr_processes=8,
        )

        rambo_master.store_rambo_master(rambo_model, dataset_name)
        preds = rambo_model.predict_classes(val_set.x_val)
        acc_score = accuracy_score(val_set.y_val, preds)
        break

    print("ACC_SCORE:")
    print(acc_score)


def train_for_submit(x_train: ndarray, y_train: ndarray, x_test: ndarray, dataset_name: str):
    nr_feats = x_train.shape[1]
    nr_classes = np.unique(y_train).size
    nr_samples = x_train.shape[0]

    param_func_dict = {"titanic": titanic_params}

    param_func = param_func_dict[dataset_name]

    init_params, hyper_sync_perc = param_func(nr_samples, nr_feats, nr_classes)

    rambo_model: RamboMaster = RamboMaster(
        nr_classes,
        last_activation="softmax",
        boost_depth=init_params.boost_depth,
        name="ensemble",
        verbose=False,
        dataset_name=dataset_name,
    )

    train_set = TrainSet(x_train, y_train)

    rambo_model.fit_with_hyperopt(
        train_set,
        init_params,
        hyper_sync_perc=hyper_sync_perc,
        nr_nets=init_params.nr_nets,
        nr_processes=8,
    )

    rambo_master.store_rambo_master(rambo_model, dataset_name)
    preds = rambo_model.predict_classes(x_test)

    print("--- PREDS ---")
    for pred in preds:
        print(pred)
    print("--- END PREDS ---")


def eval_trained(x_train: ndarray, y_train: ndarray, rambo_id: str):
    with open("rambo_masters/titanic/rambo_master_"+rambo_id+".pickle", "rb") as pickle_file:
        rambo_model: RamboMaster = pickle.load(pickle_file)

    rambo_model.trim_nets(x_train, y_train, acc_based=True)
    preds = rambo_model.predict_classes(x_train)
    acc_score = accuracy_score(y_train, preds)

    print("eval_ACC: " + str(acc_score))


def load_and_pred(x_train, y_train, x_test: ndarray, rambo_id: str):
    with open("rambo_masters/titanic/rambo_master_"+rambo_id+".pickle", "rb") as pickle_file:
        rambo_model: RamboMaster = pickle.load(pickle_file)

    rambo_model.trim_nets(x_train, y_train, acc_based=True)
    preds = rambo_model.predict_classes(x_test)

    print("--- PREDS ---")
    for pred in preds:
        print(pred)
    print("--- END PREDS ---")


def get_x_y(df: pd.DataFrame):
    x = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values

    return x, y


def train_and_cross(dataset_name: str):
    df_train, _ = load_train_test_datasets(dataset_name)

    x_train, y_train = get_x_y(df_train)

    train_cross_model(x_train, y_train, dataset_name)
    backend.clear_session()


def train_and_submit(dataset_name: str):
    df_train, df_test = load_train_test_datasets(dataset_name)

    x_train, y_train = get_x_y(df_train)
    x_test = df_test.values

    train_for_submit(x_train, y_train, x_test, dataset_name)
    backend.clear_session()


def load_and_eval(dataset_name: str, rambo_id: str):
    df_train, _ = load_train_test_datasets(dataset_name)

    x_train, y_train = get_x_y(df_train)
    eval_trained(x_train, y_train, rambo_id)
    backend.clear_session()


def load_old_and_pred(dataset_name: str, rambo_id: str):
    df_train, df_test = load_train_test_datasets(dataset_name)

    x_train, y_train = get_x_y(df_train)
    x_test = df_test.values
    load_and_pred(x_train, y_train, x_test, rambo_id)
    backend.clear_session()


print("running kaggle")
# train_and_submit("titanic")
load_old_and_pred('titanic', '40086')
# load_and_eval('titanic','40086')
