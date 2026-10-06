"""Public framework objects and arithmetic operator initialization."""

from mydezero.core import Variable
from mydezero.core import Parameter
from mydezero.core import Function
from mydezero.core import using_config
from mydezero.core import no_grad
from mydezero.core import as_array
from mydezero.core import as_variable
from mydezero.core import setup_variable

from mydezero.layers import Layer

from mydezero.models import Model

import mydezero.functions
import mydezero.utils

setup_variable()
