import numpy as np

# sigmoid function
def nonlin(x, deriv=False):
    if deriv:
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input data
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])

# output data
y = np.array([[0,0,0,1]]).T

# seed random numbers to make calculation deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1
