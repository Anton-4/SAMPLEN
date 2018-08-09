from typing import Tuple, List
from enum import Enum


class InitNetsParams:

    def __init__(
        self,
        sample_bstrap_perc: float,
        stratify_bstrap: bool,
        feat_bstrap_range: Tuple[float, float],
        hid_layer_ranges: List[List[Tuple[int, int]]],
        nr_layer_probs: List[float],
        metric_range: Tuple[float, float],
        nr_classes: int,
        nr_feats: int,
        nr_samples: int,
        max_epochs: int,
        nr_nets: int,
        boost_depth: int,
        boost_type: 'BoostType',
        watch_stop_val_prob: float,
        batch_size: int,
        activation: str,
        swap_noise_range: Tuple[float, float],
        swap_noise_prob: float,
        dropout_range: Tuple[float, float],
        dropout_prob: float,
    ):
        self.sample_bstrap_perc = sample_bstrap_perc
        self.stratify_bstrap = stratify_bstrap
        self.feat_bstrap_range = feat_bstrap_range
        self.hid_layer_ranges = hid_layer_ranges
        self.nr_layer_probs = nr_layer_probs
        self.metric_range = metric_range
        self.nr_classes = nr_classes
        self.nr_feats = nr_feats
        self.nr_samples = nr_samples
        self.max_epochs = max_epochs
        self.nr_nets = nr_nets
        self.boost_depth = boost_depth,
        self.boost_type = boost_type,
        self.watch_stop_val_prob = watch_stop_val_prob
        self.batch_size = batch_size
        self.activation = activation
        self.swap_noise_range = swap_noise_range
        self.swap_noise_prob = swap_noise_prob
        self.dropout_range = dropout_range
        self.dropout_prob = dropout_prob


class BoostType(Enum):
    repl_from_bstrap = 0
    fifty_fifty = 1
    wrong_only = 2
