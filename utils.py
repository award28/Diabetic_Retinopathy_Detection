import pandas as pd
import numpy as np

data = pd.read_csv('./test_ds/labels.csv')

def get_classification_for_image_name(name, data):
    return data.loc[data['image'] == name]
