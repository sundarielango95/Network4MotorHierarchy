# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 13:48:32 2022

@author: Sundari Elango
"""

import numpy as np
import model_architecture
import matplotlib.pyplot as plt
import torch
import math

# loading plant model
plant_model = model_architecture.plant_net_xandy_v2()
plant_model.load_state_dict(torch.load("models/plant/plant_xandy_1_data_v3"))
num_inputs = 8
time_steps = 40
targets = np.array(([1,0],[0.7,0.8],[0,1],[-0.7,0.8],[-1,0],[-0.7,-0.8],[0,-1],[0.7,-0.8]))
X = np.ones((num_inputs,time_steps,1,2),dtype='float32')
for i in range(num_inputs):
    X[i,:,0,0] = X[i,:,0,0]*targets[i,0]
    X[i,:,0,1] = X[i,:,0,1]*targets[i,1]
test_x = torch.from_numpy(X)
outputp = torch.zeros(num_inputs,time_steps,1,2)
outpute = torch.zeros(num_inputs,time_steps,1,2)
theta = 15; thresh = 4
for j in range(num_inputs):
    for t in range(time_steps):
        outpute[j,t,0,0] = outputp[j,t-1,0,0]
        outpute[j,t,0,1] = outputp[j,t-1,0,1]
#        plant_x = torch.zeros(1,1,1,3) # [uk, zk, z(k-1)] [T(t), o(t-1), o(t-2) ]
#        plant_x = torch.cat((test_x[j,t,:,0],outpute[j,t-1,:,0],outpute[j,t,:,0]),0)
#        plant_y = torch.zeros(1,1,1,3) # [uk, zk, z(k-1)] [T(t), o(t-1), o(t-2) ]
#        plant_y = torch.cat((test_x[j,t,:,1],outpute[j,t-1,:,1],outpute[j,t,:,1]),0)
#        op = plant_model(plant_x,plant_y)
#        outputp[j,t,0,0] = op[0] 
#        outputp[j,t,0,1] = op[1]
#            
#       x and y
        plant_x = torch.zeros(1,1,1,6) # [uk, zk, z(k-1)] [T(t), o(t-1), o(t-2) ]
        plant_x = torch.cat((test_x[j,t,:,0],outpute[j,t-1,:,0],outpute[j,t,:,0],test_x[j,t,:,1],outpute[j,t-1,:,1],outpute[j,t,:,1]),0)
        op = plant_model(plant_x)
        outputp[j,t,0,:] = op 

outputp = outputp.detach().numpy()
outputp = np.reshape(outputp,[num_inputs,time_steps,2])
test_x = test_x.detach().numpy()
test_x = np.reshape(test_x,[num_inputs,time_steps,2])

plt.figure()
for i in range(num_inputs):
    plt.scatter(targets[i,0],targets[i,1],color='b')
    plt.plot(outputp[i,:,0],outputp[i,:,1],color='b')
