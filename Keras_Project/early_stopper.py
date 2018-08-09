from keras.callbacks import Callback
import warnings


class EarlyStopper(Callback):

    def __init__(
        self,
        stop_metric,
        stop_val,
        model,
        max_epochs: int,
        watch_stop_val: bool,
        verbose: bool = False,
    ):
        super(EarlyStopper, self).__init__()

        self.monitor = stop_metric
        self.stopped_epoch = 0
        self.stop_val = stop_val
        self.model = model
        self.best_weights = None
        self.reached_val = 0.0
        self.verbose = verbose
        self.current = None
        self.max = 0.0
        self.min_val_loss = 100.0
        self.max_val_acc = 0.0
        self.epochs_since_min_adjust = 0
        self.watch_stop_val = watch_stop_val
        self.val_acc = 0.0
        self.train_acc = 0.0
        self.start_val_acc = 0.0
        self.max_epochs = max_epochs
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.stopped_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        # TODO detect go for 0 with watch_stop_val true
        current = logs.get(self.monitor)
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_acc")
        self.reached = current

        if self.start_val_acc == 0.0:
            self.start_val_acc = val_acc

        if self.reached > self.max:
            self.max = self.reached
            self.best_weights = self.model.get_weights()

        if not self.watch_stop_val and val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.epochs_since_min_adjust = 0
            self.best_weights = self.model.get_weights()
            self.train_acc = logs.get("acc")
            self.val_acc = val_acc

        if current is None:
            warnings.warn(
                "Early stopping conditioned on metric `%s` "
                "which is not available. Available metrics are: %s"
                % (self.monitor, ",".join(list(logs.keys()))),
                RuntimeWarning,
            )
            return

        if current > 0.99:
            self.current = current
            self.stopped_epoch = epoch
            self.model.stop_training = True

        if self.watch_stop_val and current >= self.stop_val:
            self.current = current
            self.stopped_epoch = epoch
            self.model.stop_training = True
            self.train_acc = logs.get("acc")
            self.val_acc = logs.get("val_acc")
        elif epoch == self.max_epochs - 1:
            self.train_acc = logs.get("acc")
            self.val_acc = logs.get("val_acc")
            self.current = current
            self.stopped_epoch = epoch
            self.model.stop_training = True

        if not self.watch_stop_val and self.epochs_since_min_adjust > 5:
            # print("performed early stop after " + str(epoch) + " epochs")
            self.current = current
            self.stopped_epoch = epoch
            self.model.stop_training = True

        self.epochs_since_min_adjust = self.epochs_since_min_adjust + 1

    def on_train_end(self, logs=None):
        self.current = self.max
        self.model.set_weights(self.best_weights)
        if self.current is not None:
            self.reached_val = "%.3f" % self.max
            if self.stopped_epoch > 0 and self.verbose:
                print(
                    "stop on Epoch "
                    + str(self.stopped_epoch)
                    + ", "
                    + self.monitor
                    + " reached "
                    + self.reached_val
                )
        else:
            print("max reached " + str(self.max))
        self.max = 0.0
        self.min_val_loss = 100.0
        self.epochs_since_min_adjust = 0

    # used to modify what gets pickled
    def __getstate__(self):
        state = self.__dict__.copy()
        del state["model"]

        return state
