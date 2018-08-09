import keras
from keras import backend
from keras.layers import Dense, Dropout

from params.rambo_net_params import RamboNetParams
from rambo_nets.regular_net import RegularNet


import numpy as np
from numpy import ndarray

from early_stopper import EarlyStopper
from train_val_set import TrainValSet, TrainSet

from typing import *


class RamboNet(RegularNet):

    def __init__(
        self,
        nr_feats: int,
        nr_classes: int,
        nr_samples: int,
        batch_size: int,
        epochs: int,
        metric_stop_val: float,
        hid_layers_sizes: List[int],
        activation: str = "tanh",
        name: str = None,
        ensemble_id=None,
        verbose: bool = True,
        watch_stop_val: bool = True,
        generation: int = 0,
        parent_id: str = "no_parent",
        net_id: str = None,
        rambo_master_id: str = "MISSING_RAMBO_MASTER_ID",
        swap_noise: float = None,
        dropout: float = 0.0,
    ):
        self.dropout = dropout

        super().__init__(
            nr_feats,
            nr_classes,
            nr_samples,
            batch_size,
            epochs,
            metric_stop_val,
            hid_layers_sizes,
            activation,
            name,
            ensemble_id,
            verbose,
            watch_stop_val,
            generation,
            parent_id,
            net_id,
            rambo_master_id
        )

        self.nr_feats = nr_feats
        self.swap_noise = swap_noise
        self.is_boost_net = False

        self.run_fit_count = 0

        if not name:
            prefix = "rambo_net_"
            name = prefix + str(backend.get_uid(prefix))
        self.name = name

        self.model_path = "models/" + rambo_master_id + "/" + self.net_id + "_keras_model.h5"

    def get_params(self) -> RamboNetParams:
        base_params = super(RamboNet, self).get_params()

        params = RamboNetParams(
            self.nr_feats,
            self.swap_noise,
            self.dropout,
            base_params.hid_layer_sizes,
            base_params.activation,
            base_params.target_stop_val,
            base_params.bag_train_acc,
            base_params.oob_train_acc,
            base_params.full_train_acc,
            base_params.val_acc,
            base_params.test_acc,
            base_params.start_val_acc,
            base_params.watch_stop_val,
            base_params.epochs_trained,
            base_params.batch_size,
            base_params.net_weights,
            base_params.layer_sizes,
            base_params.nr_hid_layers,
            base_params.generation,
            base_params.net_id,
            base_params.parent_id,
            base_params.time_created,
        )

        return params

    def list_params(self) -> List[str]:
        base_params = super(RamboNet, self).list_params()
        params = ["nr_feats", "swap_noise", "dropout"] + base_params

        return params

    def init_model(self, nr_feats):
        units = self.hid_layers_sizes[0]

        self.model.add(
            Dense(
                units=units,
                input_shape=(nr_feats,),
                activation=self.activation,
                name="input_layer_dense",
            )
        )
        self.model.add(Dropout(self.dropout))

        for i in range(1, len(self.hid_layers_sizes)):
            units = self.hid_layers_sizes[i]
            self.model.add(
                Dense(
                    units=units,
                    activation=self.activation,
                    name="hidden_layer_" + str(i),
                )
            )

        if self.nr_classes == 2:
            self.model.add(
                Dense(units=1, activation="sigmoid", name="final_layer_dense")
            )
        else:
            self.model.add(
                Dense(
                    units=self.nr_classes,
                    activation="softmax",
                    name="final_layer_dense",
                )
            )

    def fit(self, init_train_val_set: TrainValSet):
        self.run_fit_count = 0

        def run_fit():
            self.label_enc.fit(init_train_val_set.y_train)

            x_train, y_train = (init_train_val_set.x_train, init_train_val_set.y_train)

            if self.swap_noise is not None:
                x_train = self.swap_noise_in_x(x_train, y_train, self.swap_noise)

            y_train = self.transform_y(y_train)

            # x_train, y_train = self.mixup_augment(x_train, y_train, 0.25, int(x_train.shape[0]*0.5))

            self.early_stopper = EarlyStopper(
                stop_metric="acc",
                stop_val=self.metric_stop_val,
                model=self.model,
                max_epochs=self.epochs,
                watch_stop_val=self.watch_stop_val,
                verbose=self.verbose,
            )

            x_val = init_train_val_set.x_val

            y_val = self.transform_y(init_train_val_set.y_val)

            if self.watch_stop_val:
                _ = self.model.fit(
                    x_train,
                    y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    callbacks=[self.early_stopper],
                    validation_data=(x_val, y_val),
                    verbose=self.verbose,
                )
            else:
                _ = self.model.fit(
                    x_train,
                    y_train,
                    batch_size=self.batch_size,
                    epochs=self.epochs,
                    callbacks=[self.early_stopper],
                    verbose=self.verbose,
                    validation_data=(x_val, y_val),
                )

            # csv_util.save_history(self.net_id, history)

        run_fit()

        self.bag_train_acc = self.early_stopper.train_acc
        self.val_acc = self.early_stopper.val_acc

        if self.early_stopper.current is None:
            return -1
        else:
            return self

    def transform_y(self, y: ndarray) -> ndarray:
        if self.nr_classes == 2:
            y_trans = self.label_enc.transform(y)
        else:
            y_trans = keras.utils.to_categorical(
                self.label_enc.transform(y), self.nr_classes
            )

        return y_trans

    def fit_remaining(self, x: ndarray, y: ndarray, epochs: int, batch_size: int):
        rem_y = self.transform_y(y)
        rem_x = x
        self.model.fit(
            rem_x, rem_y, batch_size=batch_size, epochs=epochs, verbose=self.verbose
        )

    def swap_noise_in_x(
        self, x_train: ndarray, y_train: ndarray, noise_rate: float = 0.15
    ) -> ndarray:

        classes, inds = self.get_class_indices(y_train)

        for i in range(0, x_train.shape[0]):
            row_a = x_train[i]
            class_ind = np.where(classes == y_train[i])[0][0]
            choice_inds = inds[class_ind]
            rand_ind = np.random.choice(choice_inds)
            row_b = x_train[rand_ind]
            x_train[i] = self.swap_row_vals(row_a, row_b, noise_rate)

        return x_train

    def swap_row_vals(
        self, row_a: ndarray, row_b: ndarray, swap_rate: float = 0.15
    ) -> ndarray:
        nr_feats = row_a.size
        nr_swap_feats = int(swap_rate * nr_feats)
        swap_inds = np.random.choice(
            list(range(0, nr_feats)), size=nr_swap_feats, replace=False
        )
        for ind in swap_inds:
            row_a[ind] = row_b[ind]

        return row_a

    def mixup_augment(
        self, x_train: ndarray, one_hot_y: ndarray, alpha: float, amount: int
    ):
        all_inds = list(range(0, x_train.shape[0]))
        rand_inds = np.random.choice(all_inds, amount * 2)
        new_x = []
        new_y = []

        for i in range(0, len(rand_inds), 2):
            lam = np.random.beta(alpha, alpha)
            mixup_x: ndarray = lam * x_train[rand_inds[i], :] + (1.0 - lam) * x_train[
                rand_inds[i + 1], :
            ]
            mixup_y: ndarray = lam * one_hot_y[rand_inds[i], :] + (
                1.0 - lam
            ) * one_hot_y[
                rand_inds[i + 1], :
            ]
            new_x.append(mixup_x)
            new_y.append(mixup_y)

        aug_x = np.asarray(new_x)
        aug_y = np.asarray(new_y)
        new_train_x = np.vstack((x_train, aug_x))
        new_one_hot_y = np.vstack((one_hot_y, aug_y))

        return new_train_x, new_one_hot_y

    def predict(self, x_test) -> ndarray:
        preds = self.model.predict(x_test, batch_size=self.batch_size)

        return preds

    def predict_classes(self, x_test) -> ndarray:
        return self.model.predict_classes(x_test).flatten()

    def get_wrong_pred_inds(self, x_train, y_train) -> ndarray:
        preds = self.predict_classes(x_train).reshape((-1,))
        wrong_pred_inds = np.nonzero(preds != y_train)[0]

        return wrong_pred_inds

    def get_wrong_pred_data(self, x_train, y_train) -> TrainSet:
        x_train = x_train
        y_train = y_train

        preds = self.predict_classes(x_train).reshape((-1,))
        wrong_pred_inds = np.nonzero(preds != y_train)[0]
        if wrong_pred_inds.size > 0:
            wrong_pred_set = TrainSet(x_train[wrong_pred_inds, :], y_train[wrong_pred_inds])
            return wrong_pred_set
        else:
            return TrainSet(np.array([]), np.array([]))
