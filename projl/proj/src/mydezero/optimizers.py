"""Optimizer base class and stochastic gradient descent updates."""


# =============================================================================
# Optimizer (base class)
# =============================================================================

class Optimizer:
    """Base optimizer that updates registered parameters with gradients.

    Call setup with a layer or model before update. Subclasses implement
    update_one to define the parameter update rule.

    Attributes:
        target (Layer or None): Layer or model whose parameters will be updated.
        hooks (list[Callable]): Callbacks run before parameter updates, in the
            order in which they were registered.
    """

    def __init__(self):
        self.target = None
        self.hooks = []

    def setup(self, target):
        """Attach a layer or model to this optimizer.

        Args:
            target (Layer): Object providing a params iterator.

        Returns:
            Optimizer: This optimizer, allowing chained setup calls.
        """
        self.target = target
        return self

    def update(self):
        """Run hooks and update parameters whose gradients are present.

        Parameters with grad set to None are excluded before hooks run. Gradient
        clearing remains the caller's responsibility through target.cleargrads().
        """
        params = [p for p in self.target.params() if p.grad is not None]

        for f in self.hooks:
            f(params)

        for param in params:
            self.update_one(param)

    def update_one(self, param):
        """Update one parameter in place according to a subclass's rule.

        Args:
            param (Parameter): Initialized parameter with a Variable gradient.

        Raises:
            NotImplementedError: If a subclass does not implement the update rule.
        """
        raise NotImplementedError

    def add_hook(self, f):
        """Register a callback that receives the parameters selected for update.

        Args:
            f (Callable[[list[Parameter]], None]): Callback that may modify the
                parameter list or gradients before updates begin.
        """
        self.hooks.append(f)


class SGD(Optimizer):
    """Apply stochastic gradient descent to parameters in place.

    Args:
        lr (float): Learning rate. Defaults to 0.01.

    Attributes:
        lr (float): Learning rate used for each update.
    """

    def __init__(self, lr=0.01):
        super().__init__()
        self.lr = lr

    def update_one(self, param):
        """Subtract the learning rate times the gradient from a parameter.

        Args:
            param (Parameter): Initialized parameter with a Variable gradient.
        """
        param.data -= self.lr * param.grad.data
