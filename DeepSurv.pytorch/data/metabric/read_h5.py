import h5py
from matplotlib import pyplot as plt
import numpy as np

result = h5py.File('metabric_IHC4_clinical_train_test.h5','r')
keyw = list(result.keys())
for keyword in keyw:
    print(result[keyword])
    sec_keyw = list(result[keyword].keys())
    for seckeyword in sec_keyw:
        f = open("chek.txt",'w')
        f.write(str(list(result[keyword][seckeyword])))

