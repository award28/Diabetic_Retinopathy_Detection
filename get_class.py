import pandas as pd
import numpy as np

data = pd.read_csv('./test_ds/labels.csv')

def get_class_for_img(name, data):
    return int(data.loc[data['image'] == name]['level'])
