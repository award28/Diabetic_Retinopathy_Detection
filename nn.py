import random
import pandas as pd
import numpy as np


class Node(object):
    def __init__(self, val):
        self.val = val
        self.prev_layer = {}
        self.next_layer = {}


    def set_val(self, val):
        self.val = val


    def get_val(self):
        return self.val


    def set_next_layer(self, nodes):
        for node in nodes:
            weight = random.uniform(-0.05, 0.05)
            self.next_layer[node] = weight


    def get_next_layer(self):
        return self.next_layer


    def set_prev_layer(self, nodes):
        for node in nodes:
            weight = node.get_next_layer()[self]
            self.prev_layer[node] = weight


    def get_prev_layer(self):
        return self.prev_layer



def backpropagation(df, learning_rate, n_inputs, n_outputs, n_hidden, n_layers=2):
    layers = []
    # Make nodes for each layer
    for i in range(4):
        layers.append([])
    
    nums = [n_inputs, n_hidden, n_hidden, n_outputs]

    for i, num in zip(range(len(nums)), nums):
        for j in range(num):
            node = Node(0)
            layers[i].append(node)

    for i in range(len(layers) - 1):
        for node in layers[i]:
            node.set_next_layer(layers[i + 1])

    for i in range(1,len(layers)):
        for node in layers[i]:
            node.set_prev_layer(layers[i - 1])


d = {
        1 : pd.Series([(255., 0., 50.), (0, 0, 255,), (20,20,20)]),
        2 : pd.Series([(0,0,255), (0,3,4), (0,0,0)]),
        }
s = pd.DataFrame(d)
s["dr"] = [1,2,3]

backpropagation(s, 0.1, len(s.columns), 5, 10)
