from keras import backend
from keras.optimizers import Adam


class AdamOptimizer(Adam):

    def __init__(self, **kwargs):
        super(AdamOptimizer, self).__init__(**kwargs)
        with backend.name_scope(self.__class__.__name__):
            self.iterations = backend.variable(0, dtype="int64", name="iterations")
        self.weight_min = 0.0
        self.weight_max = 0.0

    def get_config(self):
        config = {}
        base_config = super(AdamOptimizer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
