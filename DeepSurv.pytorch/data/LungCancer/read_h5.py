import h5py
from matplotlib import pyplot as plt
import numpy as np

result = h5py.File('clean_radiomics_table.h5','r')
print(list(result.keys()))
