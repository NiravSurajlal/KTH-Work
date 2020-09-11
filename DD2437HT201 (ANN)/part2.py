class Data:
    def __init__(self, beta, gamma, tau, n, x0):
        self.beta = beta
        self.gamma = gamma
        self.tau = tau
        self.n = n
        self.x0 = x0
        self.mac_glass_output = []
        self.mac_glass_output.append(x0)

    def gen_mackey_glass(self, t):
        for i in range(1, t):
            if i - 25 < 0:
                x_old = 0
            else: 
                x_old = self.mac_glass_output[i-25]

            x_next = self.mac_glass_output[-1] + (self.beta*x_old)/(1+x_old**10) - 0.1*self.mac_glass_output[-1] 
            self.mac_glass_output.append(x_next)

    def gen_data_set(self, data_range, data_shifts):
        num_of_pts = data_range[1]-data_range[0]
        self.x = []
        for i in range(0, len(data_shifts)):
            start = data_range[0]+data_shifts[i]
            stop = data_range[1]+data_shifts[i]+1
            self.x.append(self.mac_glass_output[start:stop])
        
        self.x = np.array(self.x).T

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import tensorflow as tf 
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior() 

# data parameters
beta = 0.2
gamma = 0.1
tau = 25
n = 10
x0 = 1.5
t_total = 1600

t = np.arange(301, 310, 1, dtype=float)
data = Data(beta, gamma, tau, n, x0)
data.gen_mackey_glass(t_total)

data_range = [301,1500]
data_shifts = [-20, -15, -10, -5, 0]

data.gen_data_set(data_range, data_shifts)
_input = data.x

data_shifts = [5]
data.gen_data_set(data_range, data_shifts)
_output = data.x

# input data
input_training_data = _input[0:500, :]
input_testing_data = _input[500:1000, :]
input_validation_data = _input[1000:1200, :]
# input_training_data = _input[0::2, :]
# input_testing_data = _input[1::4, :]
# input_validation_data = _input[3::4, :]
# output data
output_training_data = _output[0:500, :]
output_testing_data = _output[500:1000, :]
output_validation_data = _output[1000:1200, :]
# output_training_data = _output[0::2, :]
# output_testing_data = _output[1::4, :]
# output_validation_data = _output[3::4, :]

plt.figure()
plt.plot(_output)
plt.show() 
