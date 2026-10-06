"""Recursive parameter collection and gradient clearing on models."""

import numpy as np

from mydezero import Variable


def test_nested_model_collects_parameters(regression_model):
    model = regression_model

    assert set(model.params()) == {model.linear.W, model.linear.b, model.unused}


def test_nested_model_clears_all_gradients(regression_model):
    model = regression_model
    for param in model.params():
        param.grad = Variable(np.ones_like(param.data))

    model.cleargrads()

    assert all(param.grad is None for param in model.params())
