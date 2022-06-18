import h5py
from matplotlib import pyplot as plt
import numpy as np
def read_hdf5(path):

    weights = {}

    keys = []
    with h5py.File(path, 'r') as f: # open file
        f.visit(keys.append) # append all keys to list
        for key in keys:
            if ':' in key: # contains data if ':' in key
                print(f[key].name)
                weights[f[key].name] = f[key].value
    return weights

result = read_hdf5('linear_survival_data.h5')
print(len(result))
print(result)
