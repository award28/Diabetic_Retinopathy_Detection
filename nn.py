import random
import pickle
import pandas as pd
import numpy as np
from activations import sigmoid
from PIL import Image
import pandas as pd
import glob
from get_class import get_class_for_img



class Nueron(object):
    def __init__(self, val):
        self.val = val
        self.prev_layer = {}
        self.next_layer = {}


    def set_val(self, val):
        self.val = val


    def get_val(self):
        return self.val


    def set_next_layer(self, neurons):
        for neuron in neurons:
            weight = random.uniform(-0.05, 0.05)
            self.next_layer[neuron] = weight


    def get_next_layer(self):
        return self.next_layer


    def set_prev_layer(self, neurons):
        for neuron in neurons:
            weight = neuron.get_next_layer()[self]
            self.prev_layer[neuron] = weight


    def get_prev_layer(self):
        return self.prev_layer


def RGB_to_int(r, g, b):
    return r*(256**2) + g*256 + b


def int_to_RGB(num):
    r = num//(256**2)
    g = (num//256) % 256
    b = num % 256
    return (r, g, b)


def cost(predicted, actual):
    rv = 0
    for p, a in zip(predicted, actual):
        rv += (p - a)**2
    return rv


def backpropagation(df, learning_rate, n_inputs, n_outputs, n_hidden, n_layers=2):
    print("---------------------Starting ANN---------------------")
    layers = []
    # Make neurons for each layer
    for i in range(4):
        layers.append([])
    
    nums = [n_inputs, n_hidden, n_hidden, n_outputs]

    print("---------------------Creating Neuron Layers---------------------")
    for i, num in zip(range(len(nums)), nums):
        for j in range(num): 
            neuron = Nueron(0)
            layers[i].append(neuron)

    for i, neuron in zip(range(len(layers)), layers[len(layers) - 1]):
        neuron.set_val(i)

    print("---------------------Setting Next---------------------")
    for i in range(len(layers) - 1):
        for neuron in layers[i]:
            neuron.set_next_layer(layers[i + 1])

    print("---------------------Setting Prev---------------------")
    for i in range(1,len(layers)):
        for neuron in layers[i]:
            neuron.set_prev_layer(layers[i - 1])

    for i in range(1000):
        for idx, row in df.iterrows():
             print("*******FIRST LAYER*********")
             for i, neuron in zip(range(row.count() - 1), layers[0]):
                 neuron.set_val(RGB_to_int(*row[i]))
         
             print("*******SECOND LAYER*********")
             for neuron in layers[1]:
                 sigmoid_ = 0
                 for n, weight in neuron.get_prev_layer().items():
                     sigmoid_ += n.get_val()*weight
                 neuron.set_val(sigmoid(sigmoid_, 2))

             print("*******THIRD LAYER*********")
             for neuron in layers[2]:
                 sigmoid_ = 0
                 for n, weight in neuron.get_prev_layer().items():
                     sigmoid_ += n.get_val()*weight
                 neuron.set_val(sigmoid(sigmoid_, 2))

             predicted = []
             actual = []
             outputs = []
             idx = 0
             for neuron in layers[3]:
                 sigmoid_ = 0
                 for n, weight in neuron.get_prev_layer().items():
                     sigmoid_ += n.get_val()*weight
                 observed = sigmoid(sigmoid_, 2)
                 target = int(row["dr"] == idx)
                 # ****** Update the weights via stochastic gradient decent ******
                 # eta * (target - observed) * observed(1 - observed) * (previous neuron value)
                 delta = observed * (target - observed) * (1 - observed)
                 outputs.append((neuron, delta))
                 for n in neuron.get_prev_layer().keys():
                     change = learning_rate * delta * n.get_val()
                     neuron.get_prev_layer()[n] += change
                     n.get_next_layer()[neuron] += change
                 idx += 1
             
             next_outputs = []
             for neuron in layers[2]:
                 sigmoid_ = 0
                 for n, weight in neuron.get_prev_layer().items():
                     sigmoid_ += n.get_val()*weight
                 observed = sigmoid(sigmoid_, 2)
                 out_sum = 0
                 for n, d in outputs:
                     out_sum = neuron.get_next_layer()[n] * d
                 delta = observed * (1 - observed) * out_sum
                 next_outputs.append((neuron, delta))
                 for n in neuron.get_prev_layer().keys():
                     change = learning_rate * delta * n.get_val()
                     neuron.get_prev_layer()[n] += change
                     n.get_next_layer()[neuron] += change
                   

             for neuron in layers[1]:
                 sigmoid_ = 0
                 for n, weight in neuron.get_prev_layer().items():
                     sigmoid_ += n.get_val()*weight
                 observed = sigmoid(sigmoid_, 2)
                 out_sum = 0
                 for n, d in next_outputs:
                     out_sum = neuron.get_next_layer()[n] * d
                 delta = observed * (1 - observed) * out_sum
                 for n in neuron.get_prev_layer().keys():
                     change = learning_rate * delta * n.get_val()
                     neuron.get_prev_layer()[n] += change
                     n.get_next_layer()[neuron] += change

    return layers
        
def predict(layers, df):
    for idx, row in df.iterrows():
        for i, neuron in zip(range(row.count() - 1), layers[0]):
            neuron.set_val(RGB_to_int(*row[i]))
    
        for neuron in layers[1]:
            sigmoid_ = 0
            for n, weight in neuron.get_prev_layer().items():
                sigmoid_ += n.get_val()*weight
            neuron.set_val(sigmoid(sigmoid_, 2))

        for neuron in layers[2]:
            sigmoid_ = 0
            for n, weight in neuron.get_prev_layer().items():
                sigmoid_ += n.get_val()*weight
            neuron.set_val(sigmoid(sigmoid_, 2))

        highest, val = -1, 0
        i = 0
        for neuron in layers[3]:
            sigmoid_ = 0
            for n, weight in neuron.get_prev_layer().items():
                sigmoid_ += n.get_val()*weight
            observed = sigmoid(sigmoid_, 2)
            if observed > highest:
                highest = observed
                val = i
            i += 1

        print("=========================")
        print("Prediction: " + str(val))
        print("Actual: " + str(row["dr"]))
        

'''
d = {
        1 : pd.Series([(255., 0., 50.), (0, 0, 255,), (20,20,20), (230, 5, 67)]),
        2 : pd.Series([(0,0,255), (0,3,4), (0,0,0), (23, 85, 6)]),
        3 : pd.Series([(88,9,71), (128,115,169), (102,71,133), (33,55,152)]),
        }
s = pd.DataFrame(d)
s = s.transpose()
s["dr"] = [1,2,0]

n_ins = s.columns.values[len(s.columns.values) - 2] + 1
layers = backpropagation(s, 0.1, n_ins, 5, 10)
predict(layers, s)
'''

data = pd.DataFrame()
size = 0, 0
for filename in glob.glob('test_ds/train/*.jpeg'):
    im = Image.open(filename, 'r')
    width, height = im.size
    if (im.size > size):
        size = width, height

data = pd.read_csv('./test_ds/labels.csv')
index = 1
d = {}
drs = []
for filename in glob.glob('test_ds/train/*.jpeg'):
    name = filename.split('/')[-1].split('.')[0]
    dr = get_class_for_img(name, data) 
    drs.append(dr)
    im = Image.open(filename, 'r')
    new_width, new_height = size
    im = im.resize((720, 720), Image.ANTIALIAS)
    px_vals = list(im.getdata())
    d[index] = pd.Series([(x[0], x[1], x[2]) for x in px_vals])
    index += 1
print("Finished parsing files") 
s = pd.DataFrame(d)
s = s.transpose()
print("Drs here: " + str(drs))
s.insert(720 * 720, 'dr', drs)
    
n_ins = s.columns.values[len(s.columns.values) - 2] + 1
layers = backpropagation(s, 0.1, n_ins, 5, 10)
predict(layers, s)
