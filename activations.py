import math
import numpy as np

def relu(pixel_vals, bias=0):
    '''Takes tuple (r,g,b) and returns the relu transformation.
    For use within each individual node in a dense layer before being passed onto the next layer
    bias 0'ed out by default
    '''
    a, b, c = pixel_vals
    return (a * (a > 0) + bias,
            b * (b > 0) + bias,
            c * (c > 0) + bias)

def sigmoid(pixel_vals, bias=0):
    '''returns sigmoid activation of a pixel
    bias 0'ed out by default'''
    a, b, c = pixel_vals
    return (1/(1+np.exp(-a)) + bias,
            1/(1+np.exp(-b)) + bias,
            1/(1+np.exp(-c)) + bias, )

def tanh(pixel_vals, bias=0):
    '''returns hyperbolic tan of a tuple of pixel_vals
    bias 0'ed out by default'''
    a, b, c = pixel_vals
    return (np.tanh(a) + bias,
            np.tanh(b) + bias,
            np.tanh(c) + bias)
