import numpy as np
from MyFlows import Variable, square, exp
x = Variable(0.5)
a = square(x); b = exp(a); y = square(b)
print("前向", *(float(v.data) for v in (x,a,b,y)))
y.grad = np.ones_like(y.data)
y.backward()
print("反向", *(float(v.grad) for v in (y,b,a,x)))
