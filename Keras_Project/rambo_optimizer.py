from keras import backend
from keras.optimizers import Optimizer
from keras.legacy import interfaces


class RamboOptimizer(Optimizer):

    def __init__(self, weight_min: float, weight_max: float, **kwargs):
        super(RamboOptimizer, self).__init__(**kwargs)
        with backend.name_scope(self.__class__.__name__):
            self.iterations = backend.variable(0, dtype="int64", name="iterations")
        self.weight_min = weight_min
        self.weight_max = weight_max

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):

        self.updates = [backend.update_add(self.iterations, 1)]

        shapes = [backend.int_shape(p) for p in params]
        moments = [backend.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p in params:

            new_p = backend.random_uniform(
                p.shape, minval=self.weight_min, maxval=self.weight_max
            )

            # Apply constraints.
            if getattr(p, "constraint", None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(backend.update(p, new_p))

        return self.updates

    def get_config(self):
        config = {}
        base_config = super(RamboOptimizer, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))
