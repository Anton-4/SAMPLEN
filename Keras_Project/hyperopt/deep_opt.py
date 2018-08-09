import os
from multiprocessing import Manager

import csv_util
from params.init_nets_params import InitNetsParams
from params.net_params import NetParams
from processes.nn_process import NNProcess
from processes.rambo_nn_process import RamboNNProcess
from train_val_set import TrainValSet


class DeepOpt:

    def __init__(
        self,
        nr_processes: int,
        train_val_set: TrainValSet,
        params: InitNetsParams,
        rambo_master_id: str,
        hyper_sync_perc: float,  # compare if params are good after processing perc of dataset
        main_net_ctr: int,
        use_rambo: bool
    ):
        self.train_val_set = train_val_set
        self.params = params
        self.hyper_sync_perc = hyper_sync_perc

        self.nn_processes = []
        self.nn_queue = Manager().Queue()
        # store params + validation score
        self.results_list = Manager().list()
        self.nn_list = []
        self.use_rambo = use_rambo
        keep_alive_best = False

        barrier = Manager().Barrier(parties=nr_processes)

        self.early_stop = params.watch_stop_val_prob == 0.0

        for i in range(0, nr_processes):
            if i == nr_processes - 1:
                keep_alive_best = True

            if use_rambo:
                nn_process = RamboNNProcess(
                    i,
                    1,
                    self.nn_queue,
                    self.results_list,
                    rambo_master_id,
                    hyper_sync_perc,
                    keep_alive_best,
                    barrier,
                    self.early_stop,
                    main_net_ctr
                )
            else:
                nn_process = NNProcess(
                    i,
                    4,
                    self.nn_queue,
                    self.results_list,
                    rambo_master_id,
                    hyper_sync_perc,
                    keep_alive_best,
                    barrier,
                    self.early_stop,
                    main_net_ctr
                )

            nn_process.set_params(params)
            nn_process.set_train_val(train_val_set)
            self.nn_processes.append(nn_process)

    def set_train_val(self, train_val_set: TrainValSet):
        self.train_val_set = train_val_set

    def run_opt(self, params_folder: str):
        for nn_process in self.nn_processes:
            nn_process.start()

        for nn_process in self.nn_processes:
            nn_process.join()

        print(self.results_list)

        best_score = 0.0
        best_params: NetParams = None
        for net_params in self.results_list:

            if self.early_stop:
                score = net_params.val_acc
            else:
                score = net_params.bag_train_acc
            if score > best_score:
                best_score = score
                best_params = net_params

        score_str = ("%.3f" % best_score).split(".")[1]

        results_path = "hyperopt_results/" + params_folder + "/"

        dir_path = os.path.dirname(results_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        csv_util.save_param_list(
            self.results_list,
            results_path
            + "/params_results"
            + "_"
            + best_params.net_id
            + "_acc"
            + score_str
            + ".csv",
            rambo_params=self.use_rambo
        )

        if self.early_stop:
            print("best val score: " + "%.3f" % best_score)
        else:
            print("best bag_train score: " + "%.3f" % best_score)
        print(best_params)

        return best_params.net_id
