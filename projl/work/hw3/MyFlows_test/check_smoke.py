from MyFlows import Variable, square, exp
x=Variable(0.5)
y=square(exp(square(x)))
assert abs(float(y.data)-1.648721270700128)<1e-12
assert y.creator.input.creator is not None
print("前向与建图检查通过")
