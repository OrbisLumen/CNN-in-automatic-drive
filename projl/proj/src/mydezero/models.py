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
    """Apply layers in their supplied order.

    Each layer is registered as l0, l1, and so on, allowing params and
    cleargrads to recurse into it. An empty sequence returns its input.

    Args:
        *layers (Layer): Layers accepting and returning a single value.

    Attributes:
        layers (list[Layer]): Layers in forward execution order.
    """

    def __init__(self, *layers):
        super().__init__()
        self.layers = []
        for i, layer in enumerate(layers):
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        """Pass one input through every layer.

        Args:
            x (Variable or np.ndarray): Input accepted by the first layer.

        Returns:
            Variable or np.ndarray: Last layer's output, or x for an empty model.
        """
        for layer in self.layers:
            x = layer(x)
        return x

class MLP(Model):
    """Build fully connected layers with activation on hidden outputs only.

    Linear layers infer their input widths on the first forward pass. The last
    layer produces an affine output without applying the activation.

    Args:
        fc_output_sizes (Sequence[int]): Nonempty sequence of layer output widths,
            including the final output width.
        activation (Callable): Differentiable hidden-layer activation, defaulting
            to mydezero.functions.sigmoid.

    Attributes:
        activation (Callable): Activation applied between Linear layers.
        layers (list[Linear]): Registered layers in forward execution order.
    """

    def __init__(self, fc_output_sizes, activation = F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self, x):
        """Apply hidden affine transforms and activations, then the output layer.

        Args:
            x (Variable or np.ndarray): Input batch of shape (N, in_size).

        Returns:
            Variable: Output batch of shape (N, fc_output_sizes[-1]).
        """
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)
