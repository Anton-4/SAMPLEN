import random
from datetime import datetime
from multiprocessing import Process
from typing import Tuple
import copy

import pandas as pd
import numpy as np
from numpy import ndarray
from pandas import DataFrame

from keras import backend, losses
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import rambo_master
from adam_optimizer import AdamOptimizer
from rambo_nets.regular_net import RegularNet
from train_val_set import TrainSet, TrainValSet
from params.init_nets_params import InitNetsParams, BoostType
from rambo_master import RamboMaster

dataset_dir = "/home/anton/gitrepos/ExploringSimplicity/Datasets/"


def load_dataset(dataset_name: str):
    df = pd.read_csv(
        dataset_dir + dataset_name + "/" + dataset_name + ".csv", header=None
    )

    return df


def load_train_test_datasets(dataset_name: str):
    df_train = pd.read_csv(dataset_dir + dataset_name + "/" + "train.csv", header=None)
    df_test = pd.read_csv(dataset_dir + dataset_name + "/" + "test.csv", header=None)

    return df_train, df_test


def mnist_params(
    nr_samples: int, nr_feats: int, nr_classes: int
) -> Tuple[InitNetsParams, float]:
    boost_depth = 1

    sample_bstrap_perc = 0.64
    stratify_bstrap = False
    feat_bstrap_range = (0.999, 1.0)
    hid_layers_ranges = [
        [(20, 400), (20, 400)],
        [(30, 400), (30, 400), (30, 400)],
        [(20, 400), (20, 400), (20, 400), (20, 400)],
    ]
    nr_layer_probs = [0.33, 0.33, 0.34]
    metric_range = (0.98, 0.99)
    nr_nets = 12

    watch_stop_val_prob = 0.0
    batch_size = 32
    swap_noise_range = (0.05, 0.15)
    swap_noise_prob = 0.0
    dropout_range = (0.05, 0.27)
    dropout_prob = 0.0

    max_epochs = 250
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
        boost_type=BoostType.wrong_only,
        watch_stop_val_prob=watch_stop_val_prob,
        batch_size=batch_size,
        activation="tanh",
        swap_noise_range=swap_noise_range,
        swap_noise_prob=swap_noise_prob,
        dropout_range=dropout_range,
        dropout_prob=dropout_prob,
    )

    return init_params, hyper_sync_perc


def steel_params(
    nr_samples: int, nr_feats: int, nr_classes: int
) -> Tuple[InitNetsParams, float]:
    boost_depth = 1

    sample_bstrap_perc = 0.70
    stratify_bstrap = False
    feat_bstrap_range = (0.999, 1.0)
    hid_layers_ranges = [
        [(5, 20)],
        [(15, 20), (5, 10)],
        [(30, 40), (20, 30)],
        [(20, 40), (20, 40), (20, 40)],
        [(100, 90), (80, 70), (70, 60)],
        [(250, 200), (200, 150), (150, 100)],
        [(400, 300), (350, 250), (300, 200), (250, 150)],
        [(20, 40), (20, 40), (20, 40), (20, 40), (20, 40)],
    ]
    nr_layer_probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
    metric_range = (0.98, 0.99)
    nr_nets = 12

    watch_stop_val_prob = 1.0
    batch_size = 32
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
        boost_type=BoostType.wrong_only,
        watch_stop_val_prob=watch_stop_val_prob,
        batch_size=batch_size,
        activation="tanh",
        swap_noise_range=swap_noise_range,
        swap_noise_prob=swap_noise_prob,
        dropout_range=dropout_range,
        dropout_prob=dropout_prob,
    )

    return init_params, hyper_sync_perc


def census_params(
    nr_samples: int, nr_feats: int, nr_classes: int
) -> Tuple[InitNetsParams, float]:
    boost_depth = 1

    sample_bstrap_perc = 0.70
    stratify_bstrap = False
    feat_bstrap_range = (0.999, 1.0)
    # benches for metric range: (0.885, 0.895)
    # 0.852, 0.85486, 0.84497, 0.8500, 0.8532645, 0.842024, 0.839076, 0.83508384, 0.84423, 0.84675388
    # mean: 0.8462
    # std: 0.0064
    hid_layers_ranges = [
        [(24, 37), (24, 37)],
        [(60, 100), (60, 100)],
        [(24, 37), (24, 37), (24, 37), (24, 37)],
        [(100, 256), (100, 256), (100, 256), (100, 256)],
        [(100, 256), (100, 256), (100, 256), (100, 256), (100, 256), (100, 256)]
    ]
    nr_layer_probs = [0.2, 0.2, 0.2, 0.2, 0.2]
    metric_range = (0.987, 0.99)
    nr_nets = 12

    watch_stop_val_prob = 0.0
    batch_size = 32
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
        boost_type=BoostType.wrong_only,
        watch_stop_val_prob=watch_stop_val_prob,
        batch_size=batch_size,
        activation="tanh",
        swap_noise_range=swap_noise_range,
        swap_noise_prob=swap_noise_prob,
        dropout_range=dropout_range,
        dropout_prob=dropout_prob,
    )

    return init_params, hyper_sync_perc


def cancer_params(
    nr_samples: int, nr_feats: int, nr_classes: int
) -> Tuple[InitNetsParams, float]:
    boost_depth = 1

    sample_bstrap_perc = 0.60
    stratify_bstrap = False
    feat_bstrap_range = (0.999, 1.0)
    hid_layers_ranges = [
        [(5, 20)],
        [(15, 20), (5, 10)],
        [(30, 40), (20, 30)],
        [(100, 90), (80, 70), (70, 60)],
        [(250, 200), (200, 150), (150, 100)],
        [(400, 300), (350, 250), (300, 200), (250, 150)],
    ]
    nr_layer_probs = [0.166, 0.166, 0.17, 0.166, 0.166, 0.166]
    metric_range = (0.987, 0.99)
    nr_nets = 12

    watch_stop_val_prob = 0.0
    batch_size = 32
    swap_noise_range = (0.05, 0.15)
    swap_noise_prob = 0.0
    dropout_range = (0.05, 0.27)
    dropout_prob = 0.0

    max_epochs = 500
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
        boost_type=BoostType.wrong_only,
        watch_stop_val_prob=watch_stop_val_prob,
        batch_size=batch_size,
        activation="tanh",
        swap_noise_range=swap_noise_range,
        swap_noise_prob=swap_noise_prob,
        dropout_range=dropout_range,
        dropout_prob=dropout_prob,
    )

    return init_params, hyper_sync_perc


def train_eval_model(
    x_train: ndarray,
    y_train: ndarray,
    dataset_name: str,
    x_test: ndarray = None,
    y_test: ndarray = None,
    boost_depth: int = None,
) -> (float, str):

    nr_feats = x_train.shape[1]
    nr_classes = np.unique(y_train).size
    nr_samples = x_train.shape[0]

    param_func_dict = {
        "steel": steel_params,
        "census": census_params,
        "breast_cancer": cancer_params,
        "MNIST": mnist_params,
    }
    param_func = param_func_dict[dataset_name]

    orig_init_params, hyper_sync_perc = param_func(nr_samples, nr_feats, nr_classes)

    if boost_depth is not None:
        init_params = copy.deepcopy(orig_init_params)
        init_params.boost_depth = boost_depth
    else:
        init_params = orig_init_params

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
    # test
    print(str(len(rambo_model.rambo_nets)) + " nets in ensemble")
    rambo_model.test_rambo_nets(x_test, y_test)
    preds = rambo_model.predict_classes(x_test)
    acc_score = accuracy_score(y_test, preds)

    print(acc_score)

    return acc_score, rambo_model.rambo_master_id


def get_x_y(df: DataFrame):
    x = df.iloc[:, 0:-1].values
    y = df.iloc[:, -1].values

    return x, y


def get_x_y_train_val(df_train: DataFrame, df_val: DataFrame):
    x_train, y_train = get_x_y(df_train)
    x_val, y_val = get_x_y(df_val)

    return x_train, y_train, x_val, y_val


def train_test_single(
    x_train: ndarray, y_train: ndarray, x_test: ndarray, y_test: ndarray
):

    nr_feats = x_train.shape[1]
    nr_classes = np.unique(y_train).size
    nr_samples = x_train.shape[0]

    hid_layers_sizes = [381, 377]
    act_fun = "relu"
    metric_stop = 0.99
    batch_size = 32
    max_epochs = 500

    reg_net = RegularNet(
        nr_feats,
        nr_classes,
        nr_samples,
        batch_size,
        max_epochs,
        metric_stop,
        hid_layers_sizes,
        activation=act_fun,
        verbose=False,
        watch_stop_val=True,
        generation=0,
        rambo_master_id="NO_MASTER",
        net_id="reg_net_no_ensem_" + str(random.randint(0, 100000)),
    )
    #
    # train_set = TrainSet(x_train, y_train)
    # val_size = 0.25
    # train_val_set = train_set.split_into_train_val(val_size)

    train_val_set = TrainValSet(x_train, y_train, x_val=None, y_val=None)

    loss = losses.binary_crossentropy
    if nr_classes > 2:
        loss = losses.categorical_crossentropy

    dt = datetime.now()
    time_seed = dt.microsecond
    np.random.seed(time_seed)  # otherwise seeds are the same when process is restarted
    reg_net.compile(AdamOptimizer(), loss)

    reg_net.fit(train_val_set)

    acc_score = accuracy_score(y_test, reg_net.predict_classes(x_test))

    return acc_score


def train_and_report(dataset_name: str, boost_depth: int = None) -> (float, str):
    df_train, df_test = load_train_test_datasets(dataset_name)

    x_train, y_train = get_x_y(df_train)

    label_enc = LabelEncoder()
    y_train = label_enc.fit_transform(y_train)

    x_test, y_test = get_x_y(df_test)
    y_test = label_enc.transform(y_test)

    acc_score, rambo_master_id = train_eval_model(
        x_train,
        y_train,
        dataset_name,
        x_test=x_test,
        y_test=y_test,
        boost_depth=boost_depth,
    )
    backend.clear_session()

    return acc_score, rambo_master_id


def train_report_single(dataset_name: str) -> (float, str):
    df_train, df_test = load_train_test_datasets(dataset_name)

    x_train, y_train = get_x_y(df_train)

    label_enc = LabelEncoder()
    y_train = label_enc.fit_transform(y_train)

    x_test, y_test = get_x_y(df_test)
    y_test = label_enc.transform(y_test)

    acc_score = train_test_single(
        x_train,
        y_train,
        x_test=x_test,
        y_test=y_test
    )
    backend.clear_session()

    return acc_score


all_scores = []

dataset = "steel"
results_file = dataset + "_bench_results.txt"


def main(boost_depth: int = None):
    with open(results_file, "a") as res_file:
        score, master_id = train_and_report(dataset, boost_depth)  # train_report_single(dataset)
        perc_score = 100*score
        formatted_score = "%.4f" % perc_score
        res_file.write(formatted_score + " " + master_id + "\n")


def exec_boost_analysis():
    experiment_name = "boost level analysis 50/50 c8 SIMPLE NETS"
    with open(results_file, "a") as res_file:
        res_file.write("\n------" + experiment_name + "------\n")

    for depth in [1, 2, 3, 4, 6, 8, 12]:
        experiment_name = "boost level " + str(depth)
        with open(results_file, "a") as res_file:
            res_file.write("\n------" + experiment_name + "------\n")

        for _ in range(0, 10):
            p = Process(target=main, args=(depth,))
            p.start()
            p.join()


def exec_bench():
    experiment_name = "randVSgrad 4per_proc 8proc sync_perc1.0 40%val boost depth 2 high overfit"
    with open(results_file, "a") as res_file:
        res_file.write("\n------" + experiment_name + "------\n")

    depth = 2

    for _ in range(0, 10):
        p = Process(target=main, args=(depth,))
        p.start()
        p.join()


exec_bench()
# all_scores.append(score)
# score = train_and_report('steel')
# score = train_and_report('census')
#
# print(all_scores)
# print("%.3f" % np.mean(all_scores))
# print("%.3f" % np.std(all_scores))

# best_score = 0.73
# for _ in range(0, 200):
#    score = train_and_report('steel', epochs=150, nr_folds=5, score_to_beat=best_score)

#    if score > best_score:
#        best_score = score
