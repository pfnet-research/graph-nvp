import chainer
from chainer.functions import relu
from chainer import links


class MLP(chainer.Chain):
    """
    Basic implementation for MLP

    """

    def __init__(self, units, in_size=None, activation=relu):
        super(MLP, self).__init__()
        assert isinstance(units, (tuple, list))
        assert len(units) >= 1
        n_layers = len(units)

        units_list = [in_size] + list(units)
        # layers = [links.Linear(None, hidden_dim) for i in range(n_layers - 1)]
        layers = [links.Linear(units_list[i], units_list[i+1]) for i in range(n_layers)]
        with self.init_scope():
            self.layers = chainer.ChainList(*layers)
        self.activation = activation
        self.n_layers = n_layers

    def __call__(self, x):
        h = x
        for i in range(self.n_layers - 1):
            h = self.activation(self.layers[i](h))
        h = self.layers[-1](h)
        return h
