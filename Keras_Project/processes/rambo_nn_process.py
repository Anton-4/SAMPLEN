
from params.net_params import NetParams
from params.rambo_net_params import RamboNetParams
from processes.nn_process import NNProcess
from params.init_nets_params import InitNetsParams
from hyperopt import hyper_util as util

from multiprocessing import Queue, Barrier

from typing import List, cast

from rambo_nets.rambo_net import RamboNet
from train_val_set import TrainSet, TrainValBloom


class RamboNNProcess(NNProcess):
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
        super().__init__(
            process_id,
            nr_nets,
            ret_queue,
            results_list,
            rambo_master_id,
            hyper_sync_perc,
            keep_alive_best,
            barrier,
            early_stop,
            main_net_ctr
        )

        self.orig_train_set: TrainSet = None
        self.train_val_blooms: List[TrainValBloom] = None
        self.all_chosen_feats = []

    def random_init_nets(self, params: InitNetsParams, nr_nets: int):
        return util.random_init_rambo_nets(
            self.params, self.nr_nets, self.rambo_master_id, self.process_id, self.main_net_ctr
        )

    def random_perturb_params(
        self, params_list: List[NetParams], init_params: InitNetsParams
    ):
        # cast only for typechecker
        rambo_params_list = cast(List[RamboNetParams], params_list)
        return util.random_perturb_rambo_params(
            rambo_params_list, init_params, self.rambo_master_id
        )

    def keep_params(self, params_list: List[NetParams], init_params: InitNetsParams):
        ret_nn_list = []
        rambo_params_list = cast(List[RamboNetParams], params_list)

        for params in rambo_params_list:
            nn = RamboNet(
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
                swap_noise=params.swap_noise,
                dropout=params.dropout,
            )

            ret_nn_list.append(nn)

        return ret_nn_list
