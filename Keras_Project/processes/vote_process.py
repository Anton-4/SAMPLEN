import pickle
from multiprocessing import Process
from numpy import ndarray

from typing import List

import tensorflow as tf

from rambo_nets.rambo_net import RamboNet
from train_val_set import TrainSet


class VoteProcess(Process):

    def __init__(
        self,
        nn_path: str,
        dataset: TrainSet,
        votes_container: List
    ):
        super(VoteProcess, self).__init__()
        self.nn_path = nn_path
        self.dataset = dataset
        self.votes_container = votes_container

    def get_session_config(self):
        num_cores = 1
        num_CPU = 4
        num_GPU = 0

        config = tf.ConfigProto(
            intra_op_parallelism_threads=num_cores,
            inter_op_parallelism_threads=num_cores,
            allow_soft_placement=False,
            device_count={"CPU": num_CPU, "GPU": num_GPU},
        )

        return config

    def run(self):
        with tf.Session(graph=tf.Graph(), config=self.get_session_config()) as _:
            rambo_net = self.load_rambo_net(self.nn_path)
            net_votes: ndarray = rambo_net.predict_classes(self.dataset)

            self.votes_container.append(net_votes)

    def load_rambo_net(self, nn_path: str) -> RamboNet:
        with open(nn_path, "rb") as pickle_file:
            rambo_net: RamboNet = pickle.load(pickle_file)
            return rambo_net
