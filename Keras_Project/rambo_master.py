import copy
import math
from typing import Tuple, List, Optional, Dict

import csv
import os.path
import time
import uuid
import random
import pickle
import glob

import numpy as np
from numpy import ndarray
from sklearn.metrics import accuracy_score
from multiprocessing import Manager
import tensorflow as tf

import csv_util
from hyperopt import hyper_util
from hyperopt.deep_opt import DeepOpt
from params.init_nets_params import InitNetsParams
from processes.vote_process import VoteProcess

from rambo_nets.rambo_net import RamboNet
import compacter
from rambo_nets.regular_net import RegularNet
from train_val_set import TrainSet, TrainValBloom


class RamboMaster:
    def __init__(
        self,
        nr_classes,
        last_activation: str,
        rambo_nets: List[RamboNet] = None,
        boost_depth: int = 2,
        name: str = None,
        verbose: bool = True,
        dataset_name: str = None,
    ):
        self.rambo_nets: List[RegularNet] = []
        self.nr_classes = nr_classes
        self.last_activation = last_activation
        self.built = False
        self.boost_depth = boost_depth
        self.verbose = verbose
        self.ensemble_id = str(uuid.uuid4())
        self.dataset_name = dataset_name
        self.dropped_nets = []
        self.overlap_original = []
        self.overlap_trim = []
        self.threads = []
        self.mean_train_acc = -1.0
        self.nn_queue = Manager().Queue()
        self.init_params = None
        self.chosen_feats_dict: Dict[str, List[int]] = {}
        self.rambo_master_id = "rambo_master_" + str(random.randint(0, 100000))

        rambo_path = "models/" + self.rambo_master_id + "/"
        dir_path = os.path.dirname(rambo_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        if not name:
            prefix = "rambo_master_"
            name = prefix
        self.name = name

        if rambo_nets:
            for rambo_net in rambo_nets:
                self.add(rambo_net)

        print("MASTER_ID=" + self.rambo_master_id)

    def save_str_params(self, params: InitNetsParams):
        date_time = time.strftime("%Y-%m-%d %H:%M")
        all_params = [
            self.ensemble_id,
            date_time,
            self.dataset_name,
            self.nr_classes,
            self.last_activation,
            self.boost_depth,
            params.sample_bstrap_perc,
            self.round_range(params.feat_bstrap_range),
            params.hid_layer_ranges,
            params.nr_layer_probs,
            self.round_range(params.metric_range),
            params.nr_feats,
            params.nr_samples,
            params.max_epochs,
            params.nr_nets,
            params.watch_stop_val_prob,
            self.round_range(params.swap_noise_range),
            params.swap_noise_prob,
        ]

        self.str_params = [str(param) for param in all_params]

    def round_range(self, range_tup: Tuple[float, float]):
        first = "%.3f" % range_tup[0]
        second = "%.3f" % range_tup[1]

        return first, second

    def list_params(self) -> List[str]:
        params = [
            "ensemble_id",
            "date_time",
            "dataset_name",
            "nr_classes",
            "last_activation",
            "boost_depth",
            "sample_bstrap_range",
            "feat_bstrap_range",
            "hid_layer_ranges",
            "nr_layer_probs",
            "metric_range",
            "nr_feats",
            "nr_samples",
            "max_epochs",
            "nr_nets",
            "watch_stop_val_prob",
            "swap_noise",
            "swap_noise_prob",
            "swap_same_class",
            "dropped_nets",
            "gini_original",
            "gini_trim",
            "mean_train_acc",
            "mean_acc",
            "median_acc",
            "std_dev_acc",
        ]

        return params

    def add(self, rambo_net: RamboNet):
        self.rambo_nets.append(rambo_net)
        self.built = False

    def rand_hid_layers(
        self, nr_layer_probs: List[float], hid_layer_ranges: List[List[Tuple[int, int]]]
    ):
        arch_indx = np.random.choice(
            np.arange(0, len(nr_layer_probs)), p=nr_layer_probs
        )
        my_hid_layer_ranges = hid_layer_ranges[arch_indx]
        hid_layers_sizes = [
            int(self.rand_in_range(param_range)) for param_range in my_hid_layer_ranges
        ]

        return hid_layers_sizes

    # def random_init_nets(self,
    #                      params: InitNetsParams
    #                      ):
    #
    #     self.batch_size = params.batch_size
    #     self.nr_feats = params.nr_feats
    #     self.nr_samples = params.nr_samples
    #     self.max_epochs = params.max_epochs
    #     self.nr_nets = params.nr_nets
    #
    #     self.save_str_params(params)
    #     results_list = Manager().list()
    #     for i in range(0, int(params.nr_nets / params.boost_depth)):
    #         nn_process = RamboNNProcess(i, params.boost_depth, self.nn_queue, results_list, self.rambo_master_id)
    #         nn_process.set_params(params)
    #         self.threads.append(nn_process)

    def rand_in_range(self, param_range: Tuple[float, float]):
        rand_float = random.uniform(param_range[0], param_range[1])

        return rand_float

    def rand_bool(self, prob_true: float):
        ret_bool = np.random.choice([True, False], p=[prob_true, 1 - prob_true])
        return ret_bool

    def remove_rambo_net(self, name: str):
        self.rambo_nets = [rn for rn in self.rambo_nets if rn.name != name]

    def get_rambo_net(self, name: str) -> Optional[RegularNet]:
        if not self.built:
            self.build()
        for r_net in self.rambo_nets:
            if r_net.name == name:
                return r_net

        return None

    def build(self):
        self.built = True

    def compile(self, loss):
        self.build()
        self.loss = loss

    def get_session_config(self):
        num_cores = 4
        num_CPU = 1
        num_GPU = 0

        config = tf.ConfigProto(
            intra_op_parallelism_threads=num_cores,
            inter_op_parallelism_threads=num_cores,
            allow_soft_placement=True,
            device_count={"CPU": num_CPU, "GPU": num_GPU},
        )

        return config

    def fit_with_hyperopt(
        self,
        orig_train_set: TrainSet,
        init_params: InitNetsParams,
        hyper_sync_perc: float,
        nr_nets: int,
        nr_processes: int,
    ):

        bstrap_size = int(
            math.floor(init_params.sample_bstrap_perc * orig_train_set.size())
        )
        init_params.nr_samples = bstrap_size
        if init_params.stratify_bstrap:
            with_replacement = True
        else:
            with_replacement = False

        all_feats = list(range(0, init_params.nr_feats))
        max_boost_depth = init_params.boost_depth
        nr_boost_series = int(nr_nets / max_boost_depth)
        net_ctr = 0

        for i in range(0, nr_boost_series):
            boost_group_votes = []

            for bst_level in range(0, max_boost_depth):
                nr_chosen_feats = int(
                    math.floor(
                        hyper_util.rand_in_range(init_params.feat_bstrap_range)
                        * init_params.nr_feats
                    )
                )
                mod_init_params = copy.deepcopy(init_params)
                mod_init_params.nr_feats = nr_chosen_feats

                chosen_feats = random.sample(all_feats, nr_chosen_feats)

                bstrap_bloom = self.bootstrap(
                    orig_train_set,
                    bstrap_size,
                    chosen_feats,
                    stratified=init_params.stratify_bstrap,
                    with_replacement=with_replacement,
                )

                train_set = bstrap_bloom.into_dataset(orig_train_set, save_bloom=True)

                if bst_level > 0:
                    wrong_pred_inds = self.get_wrong_pred_inds(boost_group_votes, orig_train_set.y_train)

                    train_set = train_set.merge_toss(
                        wrong_pred_inds, orig_train_set, chosen_feats, mod_init_params.boost_type
                    )

                deep_optimizer = DeepOpt(
                    nr_processes,
                    train_set,
                    mod_init_params,
                    self.rambo_master_id,
                    hyper_sync_perc,
                    main_net_ctr=net_ctr,
                    use_rambo=False
                )

                best_net_id = deep_optimizer.run_opt(
                    self.dataset_name + "/" + self.rambo_master_id
                )

                self.chosen_feats_dict[best_net_id] = chosen_feats

                if bst_level != max_boost_depth-1:
                    # do voting in child process because tensorflow
                    nn_path = "models/" + self.rambo_master_id + "/" + best_net_id + ".pickle"
                    votes_container = Manager().list()
                    vote_process = VoteProcess(nn_path, orig_train_set.x_train[:, chosen_feats], votes_container)
                    vote_process.start()
                    vote_process.join()
                    best_net_votes = votes_container[0]

                    if bst_level == 0:
                        boost_group_votes = best_net_votes
                    else:
                        boost_group_votes = np.vstack((boost_group_votes, best_net_votes))

                net_ctr += 1

        self.load_rambo_nets()
        print("MASTER_ID=" + self.rambo_master_id)

    def load_rambo_nets(self):
        model_files = glob.glob("models/" + self.rambo_master_id + "/*.pickle")
        for rambo_net_file in model_files:
            with open(rambo_net_file, "rb") as pickle_file:
                rambo_net: RegularNet = pickle.load(pickle_file)
                self.rambo_nets.append(rambo_net)

    def get_net_ind_for_id(self, net_id: str):
        for i in range(0,len(self.rambo_nets)):
            if self.rambo_nets[i].net_id == net_id:
                return i

        return -1

    def bootstrap(
        self,
        train_set: TrainSet,
        bstrap_size: int,
        chosen_feats: List[int],
        stratified: bool = False,
        with_replacement: bool = False,
    ) -> TrainValBloom:
        selected_indices = []

        all_indices_to_sample_from = np.asarray(
            list(range(0, train_set.y_train.shape[0]))
        )

        if stratified:
            bstrap_per_class = int(math.floor(bstrap_size / float(self.nr_classes)))

            labels = train_set.y_train
            label_set = np.unique(labels)

            for i in range(0, self.nr_classes):
                indices_to_sample_from = np.where(labels == label_set[i])[0]

                selected_indices.extend(
                    np.random.choice(
                        indices_to_sample_from,
                        size=bstrap_per_class,
                        replace=with_replacement,
                    )
                )
        else:

            selected_indices.extend(
                np.random.choice(
                    all_indices_to_sample_from,
                    size=bstrap_size,
                    replace=with_replacement,
                )
            )

        return TrainValBloom(selected_indices, chosen_feats, train_set.size())

    def reset_weights(self):
        for rambo_net in self.rambo_nets:
            rambo_net.reset_weights()

    # def fit_remaining(self, x: ndarray, y: ndarray, epochs: int, batch_size: int):
    #     for rambo_net in self.rambo_nets:
    #         rambo_net.fit_remaining(x, y, epochs, batch_size)

    def calc_f1_score(self, preds, y):
        true_pos = 0
        false_pos = 0
        false_neg = 0
        for i in range(0, len(y)):
            if preds[i] == 1 and preds[i] == y[i]:
                true_pos = true_pos + 1
            if preds[i] == 1 and preds[i] != y[i]:
                false_pos = false_pos + 1
            if preds[i] == 0 and preds[i] != y[i]:
                false_neg = false_neg

        recall = float(true_pos) / (true_pos + false_neg)
        precision = float(true_pos) / (true_pos + false_pos)

        f1_score = 2.0 * ((precision * recall) / (precision + recall))

        return recall, precision, f1_score

    def save_params(self):

        results_path = (
            "run_results/" + self.dataset_name + "/" + self.rambo_master_id + "/"
        )

        dir_path = os.path.dirname(results_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        results_list = [nn.get_params() for nn in self.rambo_nets]

        csv_util.save_param_list(
            results_list, results_path + "/params_results" + "_all_nets" + ".csv"
        )

    def save_results(self):
        directory = "benchmarks/" + self.dataset_name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)

        ensemble_csv_path = directory + "benchmarks.csv"
        delimiter = "\t"

        if not os.path.exists(ensemble_csv_path):
            with open(ensemble_csv_path, "w") as csvfile:
                writer = csv.writer(csvfile, delimiter=delimiter)
                header = self.list_params()
                writer.writerow(header)

        with open(ensemble_csv_path, "a") as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            row = self.str_params
            row.extend(
                [
                    str(self.dropped_nets),
                    str(self.overlap_original),
                    str(self.overlap_trim),
                    "%.3f" % self.mean_train_acc,
                    "%.3f" % self.mean,
                    "%.3f" % self.median,
                    "%.3f" % self.std,
                ]
            )
            writer.writerow(row)

        rambonets_csv_path = directory + self.ensemble_id + ".csv"
        with open(rambonets_csv_path, "w") as csvfile:
            writer = csv.writer(csvfile, delimiter=delimiter)
            header = self.rambo_nets[0].list_params()
            writer.writerow(header)
            for rambo_net in self.rambo_nets:
                row = rambo_net.get_params()
                writer.writerow(row)

    def save_result_metrics(self, mean_train_acc, mean, median, std):
        self.mean_train_acc = mean_train_acc
        self.mean = mean
        self.median = median
        self.std = std

    def take_votes(
        self, x: ndarray, net_inds: List[int] = None
    ) -> ndarray:
        preds = []

        if net_inds is None:
            for rambo_net in self.rambo_nets:
                if len(self.chosen_feats_dict.keys()) > 0:
                    chosen_feats = self.chosen_feats_dict[rambo_net.net_id]
                    chosen_feat_x = x[:, chosen_feats]
                    pred = rambo_net.predict_classes(chosen_feat_x)
                else:
                    pred = rambo_net.predict_classes(x)
                preds.append(pred)
            j_max = len(self.rambo_nets)
        else:
            for net_ind in net_inds:

                if len(self.chosen_feats_dict.keys()) > 0:
                    chosen_feats = self.chosen_feats_dict[self.rambo_nets[net_ind].net_id]
                    chosen_feat_x = x[:, chosen_feats]
                    pred = self.rambo_nets[net_ind].predict_classes(chosen_feat_x)
                else:
                    pred = self.rambo_nets[net_ind].predict_classes(x)
                preds.append(pred)
            j_max = len(net_inds)

        all_votes = []
        for i in range(0, x.shape[0]):
            votes = []
            for j in range(0, j_max):
                votes.append(preds[j][i])
            all_votes.append(votes)

        return np.array(all_votes)

    def get_wrong_pred_inds(self, votes_arr: ndarray, y: ndarray):
        preds = compacter.most_occuring(votes_arr)
        wrong_pred_inds = np.nonzero(preds != y)[0]

        return wrong_pred_inds.tolist()

    def predict(self, x, net_inds: List[int] = None):
        all_votes = self.take_votes(x, net_inds=net_inds)

        fin_preds = []

        for i in range(0, all_votes.shape[0]):
            fin_preds.append(np.array(self.occurences_to_prob(all_votes[i, :])))

        return np.array(fin_preds)

    def occurences_to_prob(self, arr):
        unique, counts = np.unique(arr, return_counts=True)
        unique = list(unique)
        sum_counts = float(sum(counts))
        probs = [(count / sum_counts) for count in counts]
        all_probs = []

        for i in range(0, self.nr_classes):
            try:
                all_probs.append(probs[unique.index(i)])
            except ValueError:
                all_probs.append(0.0)

        return all_probs

    def predict_classes(self, x, net_inds: List[int] = None):
        proba = self.predict(x, net_inds=net_inds)
        pred_classes = [np.argmax(arr) for arr in proba]

        return pred_classes

    def test_rambo_nets(self, x_test: ndarray, y_test: ndarray):
        all_test_scores = []

        for rambo_net in self.rambo_nets:
            if len(self.chosen_feats_dict.keys()) > 0:
                chosen_feats = self.chosen_feats_dict[rambo_net.net_id]
                chosen_feat_x = x_test[:, chosen_feats]
                acc_score = accuracy_score(
                    y_test, rambo_net.predict_classes(chosen_feat_x)
                )
                rambo_net.test_acc = acc_score
                all_test_scores.append(acc_score)

        print(
            "mean test acc: "
            + str(np.mean(all_test_scores))
            + " std: "
            + str(np.std(all_test_scores))
        )

    def trim_nets(self, x_val, y_val, log_file: str = None, acc_based: bool = True):
        votes_arr = self.take_votes(x_val, log_file)
        nets_to_drop_inds, overlap = compacter.trim_ensemble(
            votes_arr, y_val, acc_based=acc_based
        )
        if acc_based:
            self.overlap_trim.append(overlap)
            self.dropped_nets.append(nets_to_drop_inds)
            self.overlap_original.append(compacter.calc_overlap(votes_arr, y_val))
        nets_to_drop = [self.rambo_nets[ind] for ind in nets_to_drop_inds]
        self.old_nets = list(self.rambo_nets)
        for net in nets_to_drop:
            self.rambo_nets.remove(net)

    def trim_bottom_up(self, x_val, y_val, log_file: str = None):
        votes_arr = self.take_votes(x_val, log_file)
        best_acc = 0
        best_ind = -1
        for i in range(0, len(self.rambo_nets)):
            preds = self.predict_classes(x_val, net_inds=[i])
            acc = accuracy_score(y_val, preds)

            if acc > best_acc:
                best_acc = acc
                best_ind = i

        nets_to_drop_inds = compacter.trim_bottom_up(
            votes_arr, y_val, best_ind, best_acc
        )

        nets_to_drop = [self.rambo_nets[ind] for ind in nets_to_drop_inds]
        self.old_nets = list(self.rambo_nets)
        for net in nets_to_drop:
            self.rambo_nets.remove(net)

    def restore_nets(self):
        self.rambo_nets = self.old_nets

    # used to modify what gets pickled
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["rambo_nets"]
        del state["nn_queue"]
        del state["threads"]

        return state

    # used to modify what gets restored from pickle
    def __setstate__(self, state):
        self.__dict__.update(state)

        model_files = glob.glob("models/" + self.rambo_master_id + "/*.pickle")
        self.rambo_nets = []
        for rambo_net_file in model_files:
            with open(rambo_net_file, "rb") as pickle_file:
                rambo_net = pickle.load(pickle_file)

                if rambo_net.bag_train_acc:
                    self.rambo_nets.append(rambo_net)
        print(str(len(self.rambo_nets)) + " rambo nets loaded")


def init_rambo_dambo_from_str(dataset_name: str, ensem_id: str = None):
    if ensem_id is None:
        bench_row, header = get_best_benched_model(dataset_name)
    else:
        bench_row, header = get_specific_model(dataset_name, ensem_id)
    nr_classes = int(bench_row[header.index("nr_classes")])
    last_activation = bench_row[header.index("last_activation")]
    boost_depth = int(bench_row[header.index("boost_depth")])
    dataset_name = bench_row[header.index("dataset_name")]

    return RamboMaster(
        nr_classes,
        last_activation,
        boost_depth=boost_depth,
        dataset_name=dataset_name,
        verbose=False,
    )


def get_best_benched_model(dataset_name: str):
    bench_file_name = "benchmarks/" + dataset_name + "/benchmarks.csv"
    max_score = 0.0
    max_row = None
    with open(bench_file_name) as bench_file:
        header = bench_file.readline().split(sep="\t")
        for bench_row in bench_file:
            bench_row = bench_row.split(sep="\t")
            score = float(bench_row[header.index("mean_acc")])
            if score > max_score:
                max_score = score
                max_row = bench_row
        return max_row, header


def get_specific_model(dataset_name: str, ensem_id: str):
    bench_file_name = "benchmarks/" + dataset_name + "/benchmarks.csv"

    with open(bench_file_name) as bench_file:
        header = bench_file.readline().split(sep="\t")
        for bench_row in bench_file:
            bench_row = bench_row.split(sep="\t")
            pos_id = bench_row[header.index("ensemble_id")]
            if pos_id == ensem_id:
                return bench_row, header


def store_rambo_master(rambo_master: RamboMaster, dataset_name: str):
    rambo_path = "rambo_masters/" + dataset_name + "/"
    dir_path = os.path.dirname(rambo_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(
        rambo_path + rambo_master.rambo_master_id + ".pickle", "wb"
    ) as pickle_file:
        pickle.dump(rambo_master, pickle_file)
