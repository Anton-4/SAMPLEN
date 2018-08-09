
import random

import compacter
from params.net_params import NetParams
from rambo_nets.rambo_net import RamboNet
from rambo_nets.regular_net import RegularNet
from train_val_set import TrainValSet
from params.init_nets_params import InitNetsParams
from hyperopt import hyper_util as util

from keras import losses
from adam_optimizer import AdamOptimizer
from multiprocessing import Process, Queue, Barrier
import pickle
import numpy as np
import tensorflow as tf

from typing import List
from numpy import ndarray


class NNProcess(Process):

    def __init__(
        self,
        process_id: int,
        nr_nets: int,
        ret_queue: Queue,
        results_list: List,  # store params + validation score
        rambo_master_id: str,
        hyper_sync_perc: float,  # compare if params are good after processing perc of dataset
        keep_alive_best: bool,
        barrier: Barrier,
        early_stop: bool,
        main_net_ctr: int
    ):
        super(NNProcess, self).__init__()
        self.process_id: int = process_id
        self.nr_nets: int = nr_nets
        self.nn_list: List[RamboNet] = []
        self.train_val_set: TrainValSet = None
        self.params: InitNetsParams = None
        self.optimizer = None
        self.loss = None
        self.ret_queue = ret_queue
        self.results_list = results_list
        self.hyper_sync_perc = hyper_sync_perc
        self.best_val_acc = 0.0
        self.smart_save = False
        self.rambo_master_id = rambo_master_id
        self.keep_alive_best = keep_alive_best
        self.best_nn = None
        self.barrier = barrier
        self.early_stop = early_stop
        self.main_net_ctr = main_net_ctr

    def set_train_val(self, train_val_set: TrainValSet):
        self.train_val_set = train_val_set

    def get_session_config(self):
        num_cores = 1
        num_CPU = 1
        num_GPU = 0

        config = tf.ConfigProto(
            intra_op_parallelism_threads=num_cores,
            inter_op_parallelism_threads=num_cores,
            allow_soft_placement=False,
            device_count={"CPU": num_CPU, "GPU": num_GPU},
        )

        return config

    def run(self):
        print("process " + str(self.process_id) + " starting...")
        # seed needs to be changed otherwise all processes use the same one
        np.random.seed()

        with tf.Session(graph=tf.Graph(), config=self.get_session_config()) as _:
            self.nn_list = self.random_init_nets(self.params, self.nr_nets)

            part_size = np.math.ceil(self.params.nr_samples * self.hyper_sync_perc)
            total_nr_parts = int(np.math.ceil(self.params.nr_samples / part_size))

            for i in range(0, total_nr_parts):
                self.compile(self.train_val_set.nr_classes)
                self.fit_nets(self.train_val_set.get_part(self.hyper_sync_perc, i), i)
                if i != total_nr_parts - 1:
                    self.explore_params(self.results_list)

            # lock here, if everyone is done, compare to best acc and store if best
            self.barrier.wait()
            best_net_id = self.top_20perc_nets(self.results_list)[0].net_id
            if self.best_nn.net_id == best_net_id:
                self.save_nn(self.best_nn)

        print("process " + str(self.process_id) + " finished.")

    def save_nn(self, nn: RegularNet):
        file_name = "_".join(nn.model_path.split("_")[0:-2]) + ".pickle"
        with open(file_name, "wb") as pickle_file:
            pickle.dump(nn, pickle_file)
        self.ret_queue.put(file_name)

    def random_init_nets(self, params: InitNetsParams, nr_nets: int):
        return util.random_init_nets(self.params, self.nr_nets, self.rambo_master_id, self.process_id)

    def explore_params(self, params_list: List[NetParams]):
        top_20perc_params = self.top_20perc_nets(params_list)

        if not self.keep_alive_best:
            random.shuffle(top_20perc_params)

        chosen_params = top_20perc_params[
            0:self.nr_nets
        ]  # take nr_sets random sets of params from top 20 perc

        if not self.keep_alive_best:
            self.nn_list = self.random_perturb_params(chosen_params, self.params)
        else:
            self.nn_list = self.keep_params(chosen_params, self.params)

    def top_20perc_nets(self, params_list: List[NetParams]):
        if self.early_stop:
            sorted_params = sorted(params_list, key=lambda k: k.val_acc, reverse=True)
        else:
            sorted_params = sorted(params_list, key=lambda k: k.bag_train_acc, reverse=True)
        twenty_perc = int(len(sorted_params) * 0.2)
        if twenty_perc < 1:
            twenty_perc = 1
        top_20perc_params = sorted_params[0:twenty_perc]

        return top_20perc_params

    def random_perturb_params(
        self, params_list: List[NetParams], init_params: InitNetsParams
    ):
        return util.random_perturb_params(
            params_list, init_params, self.rambo_master_id, self.process_id
        )

    def keep_params(self, params_list: List[NetParams], init_params: InitNetsParams):
        ret_nn_list = []
        counter = 0

        for params in params_list:
            nn = RegularNet(
                init_params.nr_feats,
                init_params.nr_classes,
                init_params.nr_samples,
                params.batch_size,
                init_params.max_epochs,
                params.target_stop_val,
                params.hid_layer_sizes,
                activation=params.activation,
                verbose=False,
                watch_stop_val=params.watch_stop_val,
                generation=params.generation + 1,
                parent_id=params.net_id,
                rambo_master_id=self.rambo_master_id,
                net_id="reg_net" + "_proc" + str(self.process_id) + '_hyp' + str(counter) +
                       '_gen' + str(params.generation + 1)
            )

            counter += 1
            ret_nn_list.append(nn)

        return ret_nn_list

    def set_params(self, params: InitNetsParams):
        self.params = params

    def compile(self, nr_classes: int):
        loss = losses.binary_crossentropy
        if nr_classes > 2:
            loss = losses.categorical_crossentropy

        for nn in self.nn_list:
            nn.compile(AdamOptimizer(), loss)

    def get_wrong_pred_inds(self, votes_arr: ndarray, y: ndarray) -> List[int]:
        preds = compacter.most_occuring(votes_arr)
        wrong_pred_inds = np.nonzero(preds != y)[0]

        return wrong_pred_inds.tolist()

    def get_wrong_pred_data(self, votes_arr: ndarray, y: ndarray):
        preds = compacter.most_occuring(votes_arr)
        wrong_pred_inds = np.nonzero(preds != y)[0]

        return wrong_pred_inds

    def fit_nets(self, train_val_set: TrainValSet, generation: int):
        for i in range(0, len(self.nn_list)):
            nn = self.nn_list[i]
            nn.fit(train_val_set)

            if self.smart_save and generation > 1:
                try:
                    if self.early_stop:
                        curr_acc = nn.val_acc
                        top_20_acc = self.top_20perc_nets(self.results_list)[
                            -1
                        ].val_acc
                    else:
                        curr_acc = nn.bag_train_acc
                        top_20_acc = self.top_20perc_nets(self.results_list)[
                            -1
                        ].bag_train_acc

                    if curr_acc > top_20_acc:
                        params = nn.get_params()
                        self.results_list.append(params)
                except IndexError:
                    params = nn.get_params()
                    self.results_list.append(params)
            else:
                params = nn.get_params()
                self.results_list.append(params)

        if self.early_stop:
            self.nn_list.sort(key=lambda net: net.val_acc, reverse=True)
        else:
            self.nn_list.sort(key=lambda net: net.bag_train_acc, reverse=True)

        if self.best_nn is None:
            self.best_nn = self.nn_list[0]
        else:
            if self.early_stop:
                curr_better = self.best_nn.val_acc < self.nn_list[0].val_acc
            else:
                curr_better = self.best_nn.val_acc < self.nn_list[0].bag_train_acc
            if (
                self.best_nn is not None
                and curr_better
            ):
                self.best_nn = self.nn_list[0]

    def take_votes(self, x) -> ndarray:
        preds = []

        for nn in self.nn_list:
            pred = nn.predict(x)
            preds.append(pred)
        j_max = len(self.nn_list)

        all_votes = []
        for i in range(0, x.shape[0]):
            votes = []
            for j in range(0, j_max):
                max_ind = np.argwhere(preds[j][i] == np.amax(preds[j][i]))
                # if multiple max classes don't vote
                if len(max_ind) < 2:
                    votes.append(max_ind[0][0])
            all_votes.append(votes)

        return np.array(all_votes)

    def save_nets(self):
        for nn in self.nn_list:
            nn.store_model()
