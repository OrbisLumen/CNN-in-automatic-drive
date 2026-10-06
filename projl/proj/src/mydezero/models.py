"""Model base class with computation graph visualization."""

from mydezero import Layer
from mydezero import utils

# =============================================================================
# Model
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
