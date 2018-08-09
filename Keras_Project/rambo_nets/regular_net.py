import datetime
import keras
from keras import backend
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model, save_model

from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from params.net_params import NetParams

import numpy as np
from numpy import ndarray

from early_stopper import EarlyStopper
from train_val_set import TrainValSet

from typing import List


class RegularNet:

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
        rambo_master_id: str = "MISSING_RAMBO_MASTER_ID"
    ):
        self.model = Sequential()
        self.built = False
        self.all_feats = nr_feats
        self.hid_layers_sizes = hid_layers_sizes
        self.nr_classes = nr_classes
        self.nr_samples = nr_samples
        self.activation = activation
        self.batch_size = batch_size
        self.epochs = epochs
        self.metric_stop_val = metric_stop_val
        self.label_enc = LabelEncoder()
        self.ensemble_id = ensemble_id
        self.verbose = verbose
        self.early_stopper = None
        self.watch_stop_val = watch_stop_val
        self.bag_train_acc = 0.0
        self.oob_train_acc = 0.0
        self.full_train_acc = 0.0
        self.val_acc = 0.0
        self.generation = generation
        self.parent_id = parent_id
        self.time_created = datetime.datetime.now()
        self.test_acc = -1.0

        if not name:
            prefix = "regular_net_"
            name = prefix + str(backend.get_uid(prefix))
        self.name = name

        self.init_model(nr_feats)
        self.net_id = net_id
        self.model_path = "models/" + rambo_master_id + "/" + self.net_id + "_keras_model.h5"

    def get_params(self) -> NetParams:

        target_stop_val = self.metric_stop_val if self.watch_stop_val else -1.0
        start_val_acc = self.early_stopper.start_val_acc

        params = NetParams(
            self.hid_layers_sizes,
            self.activation,
            target_stop_val,
            self.bag_train_acc,
            self.oob_train_acc,
            self.full_train_acc,
            self.val_acc,
            self.test_acc,
            start_val_acc,
            self.watch_stop_val,
            self.early_stopper.stopped_epoch,
            self.batch_size,
            self.get_weights(),
            self.get_layer_sizes(),
            generation=self.generation,
            net_id=self.net_id,
            parent_id=self.parent_id,
            time_created=self.time_created,
        )

        return params

    def get_weights(self):
        return self.model.get_weights()

    def get_layer_sizes(self):
        layer_sizes = []
        for l in self.model.layers:
            layer_sizes.append(l.output_shape[1])

        return layer_sizes

    def list_params(self) -> List[str]:
        params = [
            "hid_layers_sizes",
            "activation",
            "target_stop_val",
            "bag_train_acc",
            "oob_train_acc",
            "full_train_acc",
            "val_acc",
            "test_acc" "watch_stop_val",
            "batch_size",
        ]

        return params

    def init_model(self, nr_feats: int):
        units = self.hid_layers_sizes[0]

        self.model.add(
            Dense(
                units=units,
                input_shape=(nr_feats,),
                activation=self.activation,
                name="input_layer_dense",
            )
        )

        for i in range(1, len(self.hid_layers_sizes)):
            units = self.hid_layers_sizes[i]
            self.model.add(
                Dense(
                    units=units,
                    activation=self.activation,
                    name="hidden_layer_" + str(i),
                )
            )

        self.model.add(
            Dense(units=self.nr_classes, activation="softmax", name="final_layer_dense")
        )

    def fit(
        self, init_train_val_set: TrainValSet
    ):

        self.label_enc.fit(init_train_val_set.y_train)

        x_train, y_train = init_train_val_set.x_train, init_train_val_set.y_train
        x_val = init_train_val_set.x_val

        y_train = keras.utils.to_categorical(
            self.label_enc.transform(y_train), self.nr_classes
        )
        if init_train_val_set.y_val is None:
            y_val = None
        else:
            y_val = keras.utils.to_categorical(
                self.label_enc.transform(init_train_val_set.y_val), self.nr_classes
            )

        self.early_stopper = EarlyStopper(
            stop_metric="acc",
            stop_val=self.metric_stop_val,
            model=self.model,
            max_epochs=self.epochs,
            watch_stop_val=self.watch_stop_val,
            verbose=self.verbose,
        )

        validation_data = (x_val, y_val)
        if x_val is None:
            validation_data = None

        if self.watch_stop_val:
            self.model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=[self.early_stopper],
                validation_data=validation_data,
                verbose=self.verbose,
            )
        else:
            self.model.fit(
                x_train,
                y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                callbacks=[self.early_stopper],
                verbose=self.verbose,
                validation_data=validation_data,
            )

        self.bag_train_acc = self.early_stopper.train_acc
        self.val_acc = self.early_stopper.val_acc

        return self

    def calc_extra_accs(
        self,
        x_train: ndarray,
        y_train: ndarray,
        x_train_oob: ndarray,
        y_train_oob: ndarray,
        x_val: ndarray,
        y_val: ndarray,
    ):
        self.full_train_acc = accuracy_score(y_train, self.predict_classes(x_train))
        self.oob_train_acc = accuracy_score(
            y_train_oob, self.predict_classes(x_train_oob)
        )
        self.val_acc = accuracy_score(y_val, self.predict_classes(x_val))

    def reset_weights(self):
        session = backend.get_session()
        for layer in self.model.layers:
            for v in layer.__dict__:
                v_arg = getattr(layer, v)
                if hasattr(v_arg, "initializer"):
                    initializer_method = getattr(v_arg, "initializer")
                    initializer_method.run(session=session)

    def get_class_indices(self, y_train: ndarray):
        vals, inverse, count = np.unique(
            y_train, return_inverse=True, return_counts=True
        )

        idx_vals_repeated = np.where(count >= 1)[0]
        vals_repeated = vals[idx_vals_repeated]

        rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
        _, inverse_rows = np.unique(rows, return_index=True)
        res = np.split(cols, inverse_rows[1:])

        return vals_repeated, res

    def m_print(self, msg: str):
        if self.verbose:
            print(msg)

    def compile(self, optimizer, loss):
        self.optimizer = optimizer

        self.model.compile("adam", loss, metrics=["accuracy"])

    def predict(self, x_test) -> ndarray:
        preds = self.model.predict(x_test, batch_size=self.batch_size)

        return preds

    def predict_classes(self, x_test) -> ndarray:
        probs_arrs = self.predict(x_test)

        return np.array([np.argmax(prob_arr) for prob_arr in probs_arrs])

    def evaluate(self, x, y) -> float:
        preds = self.predict_classes(x)

        correct = 0
        for i in range(0, len(y)):
            if y[i] == preds[i]:
                correct = correct + 1

        score = correct / float(len(y))

        return score

    def report(self, x_test, y_test):
        print(classification_report(y_test, self.predict_classes(x_test)))

    # used to modify what gets pickled
    def __getstate__(self):
        self.store_model()
        state = self.__dict__.copy()
        del state["model"]
        del state["optimizer"]

        return state

    # used to modify what gets restored from pickle
    def __setstate__(self, state):
        self.__dict__.update(state)

        self.model = load_model(self.model_path)

    def store_model(self):
        save_model(self.model, self.model_path, include_optimizer=True)

    def reload_model(self):
        self.model = load_model(self.model_path)
