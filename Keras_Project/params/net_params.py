from typing import *

import datetime
from numpy import ndarray


class NetParams:

    def __init__(
        self,
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
        self.hid_layer_sizes = hid_layer_sizes
        self.activation = activation
        self.target_stop_val = target_stop_val
        self.bag_train_acc = bag_train_acc
        self.oob_train_acc = oob_train_acc
        self.full_train_acc = full_train_acc
        self.val_acc = val_acc
        self.test_acc = test_acc
        self.start_val_acc = start_val_acc
        self.watch_stop_val = watch_stop_val
        self.epochs_trained = epochs_trained
        self.batch_size = batch_size
        self.net_weights = net_weights
        self.layer_sizes = layer_sizes
        self.nr_hid_layers = len(hid_layer_sizes)
        self.generation = generation
        self.net_id = net_id
        self.parent_id = parent_id
        self.time_created = time_created

    def __str__(self):
        return str(
            self.hid_layer_sizes
        ) + ";" + self.activation + ";" + "%.3f" % self.target_stop_val + ";" + str(
            self.bag_train_acc
        ) + ";" + str(
            self.oob_train_acc
        ) + ";" + str(
            self.full_train_acc
        ) + ";" + str(
            self.val_acc
        ) + ";" + str(
            self.test_acc
        ) + ";" + str(
            self.start_val_acc
        ) + ";" + str(
            self.watch_stop_val
        ) + ";" + str(
            self.epochs_trained
        ) + ";" + str(
            self.batch_size
        ) + ";" + str(
            self.layer_sizes
        ) + ";" + str(
            self.nr_hid_layers
        ) + ";" + str(
            self.generation
        ) + ";" + self.net_id + ";" + str(
            self.parent_id
        ) + ";" + str(
            self.time_created
        )
