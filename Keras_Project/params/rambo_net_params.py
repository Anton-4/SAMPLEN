import datetime

from params.net_params import NetParams
from typing import *
from numpy import ndarray


class RamboNetParams(NetParams):

    def __init__(
        self,
        nr_feats: int,
        swap_noise: float,
        dropout: float,
        hid_layer_sizes: List[int],
        activation: str,
        target_stop_val: float,
        bag_train_acc: float,
        oob_train_acc: float,
        full_train_acc: float,
        val_acc: float,
        test_acc: float,
        start_val_acc: float,
        watch_stop_val: bool,
        epochs_trained: int,
        batch_size: int,
        net_weights: List[ndarray],
        layer_sizes: List[int],
        nr_hid_layers: int = 0,
        generation: int = 0,
        net_id: str = "no_id",
        parent_id: str = "no_parent",
        time_created: datetime = datetime.datetime.now(),
    ):
        super().__init__(
            hid_layer_sizes,
            activation,
            target_stop_val,
            bag_train_acc,
            oob_train_acc,
            full_train_acc,
            val_acc,
            test_acc,
            start_val_acc,
            watch_stop_val,
            epochs_trained,
            batch_size,
            net_weights,
            layer_sizes,
            nr_hid_layers,
            generation,
            net_id,
            parent_id,
            time_created,
        )
        self.nr_feats = nr_feats
        self.swap_noise = swap_noise
        self.dropout = dropout

    def __str__(self):
        return str(self.nr_feats) + ";" + str(self.swap_noise) + ";" + str(
            self.dropout
        ) + ";" + str(
            super().__str__()
        )
