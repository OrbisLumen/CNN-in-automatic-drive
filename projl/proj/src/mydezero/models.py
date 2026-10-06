import numpy as np
from mydezero import Layer
import mydezero.functions as F
import mydezero.layers as L
from mydezero import utils

# =============================================================================
# Model / Sequential / MLP
# =============================================================================
class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)