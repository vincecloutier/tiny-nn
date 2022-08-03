import numpy as np


#sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))