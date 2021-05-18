import numpy as np

def randomization(n):
    A = np.random.random([n,1])
    return A

def operations(h, w):
    A = np.random.random([h,w])
    B = np.random.random([h,w])
    s = A + B
    return A,B,s


def norm(A, B):
    s = np.linalg.norm(A + B)
    return s


def neural_network(inputs, weights):
    x = np.tanh(weights.T.dot(inputs))
    return x

def scalar_function(x, y):
    if x<=y:
       return (np.dot(x,y))
    else:
       return (x/y)

def vector_function(x, y):
    vector_function = np.vectorize(scalar_function)
    return vector_function(x,y)
