import numpy as np
import tensorflow as tf

from typing import List, Tuple

from numpy import ndarray
import random

from params.init_nets_params import InitNetsParams
from params.net_params import NetParams
from params.rambo_net_params import RamboNetParams
from rambo_nets.rambo_net import RamboNet
from rambo_nets.regular_net import RegularNet


def random_init_nets(params: InitNetsParams, nr_nets: int, rambo_master_id: str, process_id: int):
    nn_list = []

    for i in range(0, nr_nets):
        metric_stop = rand_in_range(params.metric_range)
        hid_layers_sizes = rand_hid_layers_probs(
            params.nr_layer_probs, params.hid_layer_ranges
        )

        nn = RegularNet(
            params.nr_feats,
            params.nr_classes,
            params.nr_samples,
            params.batch_size,
            params.max_epochs,
            metric_stop,
            hid_layers_sizes,
            activation=random.choice(["tanh", "selu", "relu", "sigmoid"]),
            verbose=False,
            watch_stop_val=np.random.choice(
                [True, False],
                p=[params.watch_stop_val_prob, 1 - params.watch_stop_val_prob],
            ),
            generation=0,
            rambo_master_id=rambo_master_id,
            net_id="reg_net" + "_proc" + str(process_id) + '_hyp' + str(i) + '_gen0'
        )

        nn_list.append(nn)

    return nn_list


def random_init_rambo_nets(params: InitNetsParams, nr_nets: int, rambo_master_id: str,
                           process_id: int, main_net_nr: int):
    nn_list = []

    for i in range(0, nr_nets):
        metric_stop = rand_in_range(params.metric_range)
        hid_layers_sizes = rand_hid_layers_probs(
            params.nr_layer_probs, params.hid_layer_ranges
        )

        if random.random() < params.swap_noise_prob:
            swap_noise = rand_in_range(params.swap_noise_range)
        else:
            swap_noise = 0

        if random.random() < params.dropout_prob:
            dropout = rand_in_range(params.dropout_range)
        else:
            dropout = 0

        nn = RamboNet(
            params.nr_feats,
            params.nr_classes,
            params.nr_samples,
            params.batch_size,
            params.max_epochs,
            metric_stop,
            hid_layers_sizes,
            activation=random.choice(["tanh", "relu", "selu"]),
            # activation=random.choice(["tanh", "selu", "relu", "sigmoid","linear"]),
            verbose=False,
            watch_stop_val=np.random.choice(
                [True, False],
                p=[params.watch_stop_val_prob, 1 - params.watch_stop_val_prob],
            ),
            generation=0,
            swap_noise=swap_noise,
            dropout=dropout,
            rambo_master_id=rambo_master_id,
            net_id="rambo_net" + str(main_net_nr) + "_proc" + str(process_id) + '_hyp' + str(i) + '_gen0'
        )

        nn_list.append(nn)

    return nn_list


def random_perturb_params(
    params_list: List[NetParams], init_params: InitNetsParams, rambo_master_id: str, process_id: int
):
    nn_list = []
    activations = ["tanh", "selu", "relu", "sigmoid"]
    keep_act_prob = 0.8
    rest_prob = 1.0 - keep_act_prob
    change_act_prob = rest_prob / (len(activations) - 1)
    counter = 0

    for params in params_list:

        target_stop_val = params.target_stop_val
        if params.target_stop_val < 0:
            target_stop_val = (
                init_params.metric_range[1] - init_params.metric_range[0]
            ) / 2
            target_stop_val = target_stop_val + init_params.metric_range[0]

        metric_range = perturb_target_acc_range(target_stop_val)

        activation_probs = [
            keep_act_prob if act_fun == params.activation else change_act_prob
            for act_fun in activations
        ]

        hid_sizes = params.hid_layer_sizes

        hid_layer_ranges = perturb_hid_layer_ranges(hid_sizes)

        nn = RegularNet(
            init_params.nr_feats,
            init_params.nr_classes,
            init_params.nr_samples,
            random_batch_size(params.batch_size),
            init_params.max_epochs,
            rand_in_range(metric_range),
            # params.hid_layer_sizes,
            rand_hid_layers(hid_layer_ranges),
            activation=np.random.choice(activations, p=activation_probs),
            verbose=False,
            # watch_stop_val=np.random.choice(
            #     [params.watch_stop_val, not params.watch_stop_val], p=[0.8, 0.2]
            # ),
            watch_stop_val=params.watch_stop_val,
            generation=params.generation + 1,
            parent_id=params.net_id,
            rambo_master_id=rambo_master_id,
            net_id="reg_net" + "_proc" + str(process_id) + '_hyp' + str(counter) + '_gen' + str(params.generation + 1)
        )

        init_weights = nn.get_weights()
        new_weights = copy_net_weights(params.net_weights, init_weights)
        nn.model.set_weights(new_weights)
        nn_list.append(nn)
        counter = counter + 1

    return nn_list


def random_perturb_rambo_params(
    params_list: List[RamboNetParams], init_params: InitNetsParams, rambo_master_id: str
):
    nn_list = []
    activations = ["tanh", "selu", "relu", "sigmoid"]
    keep_act_prob = 0.8
    rest_prob = 1.0 - keep_act_prob
    change_act_prob = rest_prob / (len(activations) - 1)
    counter = 0

    for params in params_list:

        target_stop_val = params.target_stop_val
        if params.target_stop_val < 0:
            target_stop_val = (
                init_params.metric_range[1] - init_params.metric_range[0]
            ) / 2
            target_stop_val = target_stop_val + init_params.metric_range[0]

        metric_range = perturb_target_acc_range(target_stop_val)

        activation_probs = [
            keep_act_prob if act_fun == params.activation else change_act_prob
            for act_fun in activations
        ]

        hid_sizes = params.hid_layer_sizes

        hid_layer_ranges = perturb_hid_layer_ranges(hid_sizes)

        nn = RamboNet(
            init_params.nr_feats,
            init_params.nr_classes,
            init_params.nr_samples,
            random_batch_size(params.batch_size),
            init_params.max_epochs,
            rand_in_range(metric_range),
            # params.hid_layer_sizes,
            rand_hid_layers(hid_layer_ranges),
            activation=np.random.choice(activations, p=activation_probs),
            verbose=False,
            watch_stop_val=np.random.choice(
                [params.watch_stop_val, not params.watch_stop_val], p=[0.8, 0.2]
            ),
            generation=params.generation + 1,
            parent_id=params.net_id,
            swap_noise=perturb_float(params.swap_noise),
            dropout=perturb_float(params.dropout),
            rambo_master_id=rambo_master_id,
        )

        init_weights = nn.get_weights()
        new_weights = copy_net_weights(params.net_weights, init_weights)
        nn.model.set_weights(new_weights)
        nn_list.append(nn)
        counter = counter + 1

    return nn_list


def perturb_float(float_nr: float):
    bot_float = float_nr * 0.8
    top_float = float_nr * 1.2
    top_float = 1.0 if top_float > 1.0 else top_float

    return rand_in_range((bot_float, top_float))


def perturb_target_acc_range(target_acc: float):
    max_target_acc = float(target_acc) + 0.2
    min_target_acc = float(target_acc) - 0.2
    if max_target_acc > 0.99:
        max_target_acc = 0.99
    if min_target_acc <= 0.51:
        min_target_acc = 0.51
    metric_range = (min_target_acc, max_target_acc)

    return metric_range


def perturb_hid_layer_ranges(hid_layer_sizes: List[int]):
    hid_layer_ranges = [
        perturb_layer_size(layer_size) for layer_size in hid_layer_sizes
    ]

    return hid_layer_ranges


def perturb_layer_size(layer_size: int) -> Tuple[int, int]:
    if layer_size < 4:
        return cap_to_1(layer_size - 2), layer_size + 2
    elif layer_size < 8:
        return layer_size - 3, layer_size + 3
    else:
        return int(layer_size * 0.7), int(layer_size * 1.3)


def copy_net_weights(trained_weights: List[ndarray], init_weights: List[ndarray]):
    nr_copy_layers = min(len(init_weights), len(trained_weights))

    for i in range(0, nr_copy_layers):
        max_nr_rows = min(init_weights[i].shape[0], trained_weights[i].shape[0])
        try:
            max_nr_cols = min(init_weights[i].shape[1], trained_weights[i].shape[1])
            init_weights[i][:max_nr_rows, :max_nr_cols] = trained_weights[i][
                :max_nr_rows, :max_nr_cols
            ]
        except IndexError:
            init_weights[i][:max_nr_rows] = trained_weights[i][:max_nr_rows]

    return init_weights


def rand_tup(tup_list: List[Tuple]):
    idx = np.random.choice(len(tup_list))

    return [tup_list[idx]]


def rand_hid_layers(hid_layer_ranges: List[Tuple[int, int]]):
    hid_layers_sizes = [
        int(rand_in_range(param_range)) for param_range in hid_layer_ranges
    ]

    return hid_layers_sizes


def rand_hid_layers_probs(
    nr_layer_probs: List[float], all_hid_layer_ranges: List[List[Tuple[int, int]]]
):

    nr_layers_indx = np.random.choice(
        np.arange(0, len(nr_layer_probs)), p=nr_layer_probs
    )
    hid_layer_ranges = all_hid_layer_ranges[nr_layers_indx]
    hid_layers_sizes = [
        rand_in_range_int(param_range) for param_range in hid_layer_ranges
    ]

    return hid_layers_sizes


def cap_to_1(nr_neurons: int):
    if nr_neurons < 1:
        return 1
    else:
        return nr_neurons


def random_layers(
    nr_layers_min: int,
    nr_layers_max: int,
    range_start_min: int,
    range_start_max: int,
    range_end_min: int,
    range_end_max: int,
):
    nr_layers = random.randint(nr_layers_min, nr_layers_max)
    layer_sizes = []
    for i in range(0, nr_layers):
        layer_sizes.append(
            random_range_int(
                range_start_min, range_start_max, range_end_min, range_end_max
            )
        )
    return layer_sizes


def random_range(
    range_start_min: float,
    range_start_max: float,
    range_end_min: float,
    range_end_max: float,
):
    range_start = random.uniform(range_start_min, range_start_max)
    range_end_min_adj = max(range_start, range_end_min)
    range_end = random.uniform(range_end_min_adj, range_end_max)

    return range_start, range_end


def random_range_int(
    range_start_min: int, range_start_max: int, range_end_min: int, range_end_max: int
):
    range_start = random.randint(range_start_min, range_start_max)
    range_end_min_adj = max(range_start, range_end_min)
    range_end = random.randint(range_end_min_adj, range_end_max)

    return range_start, range_end


def random_batch_size(original: int):
    new_batch_size = np.random.choice(
        [original / 2, original, original * 2], p=[0.15, 0.7, 0.15]
    )
    if new_batch_size < 4:
        new_batch_size = 4

    return int(new_batch_size)


def rand_in_range(param_range: Tuple[float, float]) -> float:
    rand_float = random.uniform(param_range[0], param_range[1])

    return rand_float


def rand_in_range_int(param_range: Tuple[int, int]) -> int:
    rand_int = int(random.uniform(param_range[0], param_range[1]))

    return rand_int


def rand_bool(prob_true: float):
    ret_bool = np.random.choice([True, False], p=[prob_true, 1 - prob_true])
    return ret_bool


def get_session_config():
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
