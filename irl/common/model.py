import chainer
import chainer.functions as F
import chainer.links as L


def pass_fn(x):
    return x


class MLP(chainer.ChainList):
    def __init__(self, n_layer, n_units, n_out, activation=F.leaky_relu, out_activation=pass_fn, hook=None, hook_params=None):
        super().__init__()

        for _ in range(n_layer-1):
            self.add_link(L.Linear(None, n_units))
        self.add_link(L.Linear(None, n_out))
        self.activations = [activation] * (n_layer - 1) + [out_activation]

        if hook:
            hook_params = dict() if hook_params is None else hook_params
            for link in self.children():
                link.add_hook(hook(**hook_params))

    def forward(self, x):
        for link, act in zip(self.children(), self.activations):
            x = act(link(x))
        return x

    def __call__(self, x):
        return self.forward(x)