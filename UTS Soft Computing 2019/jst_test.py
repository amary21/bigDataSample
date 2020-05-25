import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from jst_train import w, b1, v, b2, n_input, n_hidden, n_output, data_uji, n_data_uji

#testing

output_benar = 0
for idx_data in range(0, n_data_uji):
    label = data_uji[idx_data,0]
    feature = data_uji[idx_data,1:]
        
    #hitung nilai pada hidden layer
    z = np.zeros(n_hidden)
    for i in range(0,n_hidden):
        net = np.sum(feature * w[i]) + b1[i]
        z[i] = 1/(1 + math.exp(-net))

    #hitung nilai pada output layer
    y = np.zeros(n_output)
    f_output = np.zeros(n_output)
    for i in range(0,n_output):
        net = np.sum(z * v[i]) + b2[i]
        y[i] = 1/(1 + math.exp(-net))

        #pembulatan y[i] ke 0 atau 1
        if (y[i] >= 0.5):
            y[i] = 1
        else:
            y[i] = 0
        
    #bandingkan output MLP dengan label dengan teknik ONE HOT ENCODING
    if (y == label):
        output_benar += 1
    #end for

#hitung Akurasi MLP
akurasi = output_benar / n_data_uji
print("Akurasi MLP: " + str(akurasi))

#mencegah plot MSE tertutup
input("Press any key to continue")