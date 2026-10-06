"""Model base class with computation graph visualization."""

from mydezero import Layer
from mydezero import utils
import mydezero.functions as F
import mydezero.layers as L


# =============================================================================
# Model (Base)
# =============================================================================
class Model(Layer):
    """Base class for networks composed of registered layers.

    Subclasses implement forward and inherit recursive parameter collection and
    gradient clearing from Layer.
    """

    def plot(self, *inputs, to_file='model.png'):
        """Run forward and render the resulting computation graph using Graphviz.

        Args:
            *inputs (Variable or np.ndarray): Inputs accepted by forward.
            to_file (str): Output filename including its extension. Relative paths
                are resolved under the project's draft directory.

        Returns:
            IPython.display.Image or None: Image for notebook display when possible.

        Raises:
            FileNotFoundError: If the Graphviz dot executable is unavailable.
        """
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)

# =============================================================================
# Sequential, MLP
# =============================================================================
class Sequential(Model):
    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class MLP(Model):
    def __init__(self, fc_output_sizes, activation = F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
