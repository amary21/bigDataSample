import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import math

#baca data csv
csv_data = pd.read_csv("emailspam.csv", delimiter=';', header=0)
data = np.array(csv_data)   #konversi data csv menjadi array
data = data.astype(float)   #konversi data menjadi tipe float
n_data = len(data[:,0])     #menghitung banyaknya data

#membaca jumlah feature    
n_feature = len(data[0,:]) - 1

#membagi data: data latih dan uji
rasio_data_latih = 0.7
n_data_latih = int(n_data * rasio_data_latih)
data_latih = data[:n_data_latih,:]

data_uji = data[n_data_latih:,:]
n_data_uji = len(data_uji[:,0])

#normalisasi data latih dalam rentang [0.1, 0.9]
for i in range(1, n_feature + 1):
    data_latih[:,i] = 0.1 + ((data_latih[:,i] - min(data_latih[:,i]))/(max(data_latih[:,i])-min(data_latih[:,i]))) * 0.8

#normalisasi data uji dalam rentang [0.1, 0.9]
for i in range(1, n_feature + 1):
    data_uji[:,i] = 0.1 + ((data_uji[:,i] - min(data_uji[:,i]))/(max(data_uji[:,i])-min(data_uji[:,i]))) * 0.8

#inisialisasi parameter jst
n_input = n_feature     #jumlah neuron pada input layer
n_hidden = 8           #jumlah neuron pada hidden layer
n_output = 1            #jumlah neuron pada output layer
n_epoch = 300           #jumlah epoch/ iterasi maksimal            
alfa = 0.8             #learning rate

#inisialisasi bobot MLP dalam rentang [-1, 1]
w = np.random.rand(n_hidden,n_input) * 2 - 1
b1 = np.random.rand(n_hidden) * 2 - 1
v = np.random.rand(n_output, n_hidden) #* 2 - 1
b2 = np.random.rand(n_output) * 2 - 1

#learning
itr = 0
MSE = np.zeros(n_epoch + 1)
while(itr <= n_epoch):
    print("Epoch ke-" + str(itr))

    for idx_data in range(0, n_data_latih):
        label = data_latih[idx_data,0]
        feature = data_latih[idx_data,1:]
        
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

        #hitung error pada output layer
        error = label - y

        #hitung Jumlah error
        sum_squared_error = sum(error**2)

        #hitung faktor koreksi pada output layer
        for i in range(0, n_output):
            f_output[i] = error * y[i] * (1 - y[i])     

        #hitung perbaikan bobot antara output dan hidden layer
        delta_v = np.zeros(shape=(n_output, n_hidden))
        for i in range(0,n_output):
            delta_v[i,:] = alfa * f_output[i] * z

        #hitung perbaikan bobot BIAS (b2) antara output dan hidden layer
        delta_b2 = np.zeros(n_output)
        for i in range(0,n_output):
            delta_b2[i] = alfa * f_output[i] * 1
        
        #hitung faktor koreksi pada hidden layer
        f_hidden = np.zeros(n_hidden)
        for i in range (0, n_hidden):
            #langkah 1 - hitung f_hidden_net
            f_hidden_net = sum(f_output * v[:,i])

            #langkah 2 - hitung f_hidden
            f_hidden[i] = f_hidden_net * z[i] * (1 - z[i])

        #hitung perbaikan bobot antara hidden dan input layer
        delta_w = np.zeros(shape=(n_hidden, n_input))
        for i in range(n_hidden):
            delta_w = alfa * f_hidden[i] * feature

        #hitung perbaikan bobot antara hidden dan input layer
        delta_b1 = np.zeros(n_hidden)
        for i in range(n_hidden):
            delta_b1 = alfa * f_hidden[i] * 1

        #update semua bobot
        w = w + delta_w
        b1 = b1 + delta_b1
        v = v + delta_v
        b2 = b2 + delta_b2
    #end for

    #hitung Mean Squared Error (MSE)
    MSE[itr] = sum_squared_error / n_data_latih
    itr += 1
#end while
print("--------------RESULT---------------")
print("Mean Squared Error: " +str(MSE[n_epoch]))

#print grafik MSE hasil training
plt.title("Mean Squared Error hasil training")
plt.plot(MSE)
plt.autoscale(enable=True, axis='both', tight=None)
plt.show(block=False)
