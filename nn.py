import random
import pandas as pd
import numpy as np

class Node(object):
    def __init__(self, val):
        self.val = val
        self.next_layer = []


    def set_next_layer(self, nodes):
        for node in nodes:
            weight = random.uniform(-0.05, 0.05)
            self.next_layer.append((weight, node))


    def get_next_layer(self):
        return self.next_layer

    
    def set_val(self, val):
        self.val = val


    def get_val(self):
        return self.val


def backpropagation(_set, learning_rate, n_inputs, n_outputs, n_hidden, n_layers=2):
    for data in _set:
        training, actual = data[:len(data) - 1], data[len(data) - 1:]
    
    layers = []
    # Make nodes for each layer
    for i in range(4):
        layers.append([])
    
    nums = [n_inputs, n_hidden, n_hidden, n_outputs]

    for i, num in (range(len(nums)), nums):
        for j in range(nums):
            node = Node(0)
            layers[i].append(node)

'''
    for i in range(len(layers) - 1):
        for node in layers[i]:
            node.set_next_layer(layers[i + 1])
'''

s = [[0, 1, 2, 3, 4],[1, 1, 2, 3, 4]]
backpropagation(s, 0.1, len(s[0]) - 1, 5, len(s[0])/10)
