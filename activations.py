import math
import numpy as np

def relu(pixel_vals, bias=0):
    '''Takes tuple (r,g,b) and returns the relu transformation.
    For use within each individual node in a dense layer before being passed onto the next layer
    bias 0'ed out by default
    '''
    return (pixel_vals * (pixel_vals > 0) + bias,)

def sigmoid(pixel_vals, bias=0):
    '''returns sigmoid activation of a pixel
    bias 0'ed out by default'''
    return (1/(1+np.exp(-pixel_vals) + bias))

def tanh(pixel_vals, bias=0):
    '''returns hyperbolic tan of a tuple of pixel_vals
    bias 0'ed out by default'''
    return (np.tanh(pixel_vals) + bias)
