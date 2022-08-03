import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input data
X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])

# output data
y = np.array([[0,1,1,0]])

# seed random numbers to make calculation deterministic (just a good practice)
np.random.seed(1)

# randomly initialize weights with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1


for j in range(60000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))

    # check error
    l2_error = y - l2
    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))
    # adjust based on certainty
    l2_delta = l2_error*nonlin(l2,deriv=True)

    # check how much l1 value contributed to l2 error
    l1_error = l2_delta.dot(syn1.T)
    # adjust based on certainty
    l1_delta = l1_error * nonlin(l1, True)

    # update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("Output After Training:")
print(l1)