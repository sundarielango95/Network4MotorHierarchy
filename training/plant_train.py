# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 21:04:39 2022

@author: SVC
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss#, CrossEntropyLoss, BCELoss
from torch.optim import Adam
import model_architecture
import random
    
num_inputs = 500
time_steps = 20
fname = "data/train_random_theta_vel_dist.xlsx"
data = pd.read_excel(fname)
data = np.array(data)
uk = np.zeros((num_inputs,time_steps,4),dtype='float32') # reference signal T(t)
yk = np.zeros((num_inputs,time_steps,2),dtype='float32') # next plant output o(t)
for i in range(num_inputs):
    uk[i,:,0] = data[i*time_steps:(i+1)*time_steps,6] # T(t)
    uk[i,:,1] = data[i*time_steps:(i+1)*time_steps,7] # T(t)
    yk[i,1:,0] = data[i*time_steps:(((i+1)*time_steps)-1),2] # o(t = 1) to o(t = 20)
    yk[i,1:,1] = data[i*time_steps:(((i+1)*time_steps)-1),3] # o(t = 1) to o(t = 20)
X = np.zeros((num_inputs,time_steps,4),dtype='float32')
X = uk
Y = yk
x,y = list(X),list(Y)#,list(a),list(b)#,list(t)
x_sh,y_sh = [],[]#,[]#,[],[]
ind = list(range(len(y)))
random.shuffle(ind)
for i in ind:
    x_sh.append(x[i])
    y_sh.append(y[i])
X = np.array(x_sh)
Y = np.array(y_sh)
# converting training data into torch format
train_x  = torch.from_numpy(X)
train_y  = torch.from_numpy(Y)
# Deleting unused variable to save space
del X
del Y
del uk
del yk
# Define hyperparameters
n_epochs = 500
batch = 1
lr = 0.001
num_batches = int(num_inputs/batch)
# plant model parameters
model = model_architecture.plant_net_xandy()
model.load_state_dict(torch.load("models/plant/plant_xandy_3_data_v1_random"))
loss_all = np.zeros((n_epochs,num_batches))
# Define Loss, Optimizer
criterion = MSELoss()
optimizer = Adam(model.parameters(),lr=lr)
prev_loss = 0.1
theta = 15
print("epoch loop begins")
for epoch in range(1, n_epochs + 1): #loop for epochs
    permutation = torch.randperm(train_x.size()[0])
    j = 0
    print('\nEpoch: {}/{}.............'.format(epoch, n_epochs))
    for i in range(0,train_x.size()[0], batch): #loop for batches
        #Get current batch data
        indices = permutation[i:i+batch]
        batch_x, batch_y = train_x[indices], train_y[indices]#, train_y2[indices]
        output_p = torch.zeros(batch,time_steps, 2)
        for t in range(time_steps):
##            x y
#            plant_x = torch.zeros(batch,1,1,3) # [uk, zk, z(k-1)] [T(t), o(t-1), o(t-2) ]
#            plant_x = torch.cat((batch_x[:,t,0],output_p[:,t-2,0],output_p[:,t-1,0]),0)
#            plant_y = torch.zeros(batch,1,1,3) # [uk, zk, z(k-1)] [T(t), o(t-1), o(t-2) ]
#            plant_y = torch.cat((batch_x[:,t,1],output_p[:,t-2,1],output_p[:,t-1,1]),0)
#            op = model(plant_x,plant_y)
#            output_p[:,t,0] = op[0] 
#            output_p[:,t,1] = op[1] 
            
#           x and y
            plant_x = torch.zeros(batch,1,1,6) # [uk, zk, z(k-1)] [T(t), o(t-1), o(t-2) ]
            plant_x = torch.cat((batch_x[:,t,0],output_p[:,t-2,0],output_p[:,t-1,0],batch_x[:,t,1],output_p[:,t-2,1],output_p[:,t-1,1]),0)
            op = model(plant_x)
            output_p[:,t,:] = op 

        # Backpropagate the error
        optimizer.zero_grad()
        loss = criterion(output_p, batch_y) # Calculate loss
        loss.backward() # Does backpropagation and calculates gradients
        optimizer.step() # Updates the weights accordingly
        loss_all[epoch-1,j] = loss.item()
        j = j+1    
        # print("\nLoss: {:.8f}".format(loss.item()))
    avg_loss = np.sum(loss_all[epoch-1,:])/num_batches
    print("\nLoss:",avg_loss)
    if prev_loss > avg_loss:
        ## Save model
        torch.save(model.state_dict(), "models/plant/plant_bar")
        prev_loss = avg_loss
        
 # Save model
losses = np.zeros((n_epochs,1))
for i in range(n_epochs):
    losses[i,0] = np.sum(loss_all[i,:])/num_batches  
plt.figure()
plt.plot(losses)

