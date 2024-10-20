# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:07:28 2022

@author: Sundari Elango
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import pandas as pd
#from tqdm import tqdm
#from tqdm import trange
import matplotlib.pyplot as plt
import random
import torch
from torch.nn import MSELoss#, CrossEntropyLoss, BCELoss
from torch.optim import Adam
import model_architecture
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# plant model parameters
plant_model = model_architecture.plant_net_xandy()
plant_model.load_state_dict(torch.load("models/plant/plant_xandy_2_data_v1_random"))
# plant_model.to(device)
# subject ID number
sub_num = 1
# m1 model
m1_model = model_architecture.ctrlr_m1()
m1_model.load_state_dict(torch.load("models/rppc_lppc/m1/m1_dist_"+str(sub_num)+"_200"))
# m1_model.to(device)
fname = "data/train_random_theta_vel_dist.xlsx"
data = pd.read_excel(fname)
data = np.array(data)    
num_inputs = 500
time_steps = 20 
uk = np.zeros((num_inputs,time_steps,4),dtype='float32') # reference signal [Txl Tyl HSl Txr Tyr HSr] 
yk = np.zeros((num_inputs,time_steps,2),dtype='float32') # next plant output o(t)
for i in range(num_inputs):
    # common inputs
    uk[i,:,0] = data[i*time_steps:(i+1)*time_steps,0]
    uk[i,:,1] = data[i*time_steps:(i+1)*time_steps,1]
    uk[i,:,2] = data[i*time_steps:(i+1)*time_steps,6]
    uk[i,:,3] = data[i*time_steps:(i+1)*time_steps,7]
    # uk[i,:,4] = data[i*time_steps:(i+1)*time_steps,8]
    yk[i,1:,0] = data[i*time_steps:(((i+1)*time_steps)-1),2]
    yk[i,1:,1] = data[i*time_steps:(((i+1)*time_steps)-1),3]
    
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
# # import to GPU
# train_x.to(device)
# train_y.to(device)
# Deleting unused variable to save space
del X
del Y
del uk
del yk
# Define hyperparameters
n_epochs = 500 # 50 - mse loss
batch = 1
lr = 0.001 #0.005 #0.001 - mse loss
num_batches = int(num_inputs/batch)
rppc_model = model_architecture.rppc()
# rppc_model.load_state_dict(torch.load("models/rppc_lppc/rppc/PM/normal/rppc_v1_"+str(sub_num)+"_200_epochs"))
# # uncomment for V2 mode of training
# rppc_model.fc_1x[0].weight.requires_grad = False
# rppc_model.fc_1x[0].bias.requires_grad = False
# rppc_model.fc_2x[0].weight.requires_grad = False
# rppc_model.fc_2x[0].bias.requires_grad = False
# rppc_model.fc_3x[0].weight.requires_grad = False
# rppc_model.fc_3x[0].bias.requires_grad = False
# rppc_model.fc_4x[0].weight.requires_grad = False
# rppc_model.fc_4x[0].bias.requires_grad = False
# rppc_model.fc_1y[0].weight.requires_grad = False
# rppc_model.fc_1y[0].bias.requires_grad = False
# rppc_model.fc_2y[0].weight.requires_grad = False
# rppc_model.fc_2y[0].bias.requires_grad = False
# rppc_model.fc_3y[0].weight.requires_grad = False
# rppc_model.fc_3y[0].bias.requires_grad = False
# rppc_model.fc_4y[0].weight.requires_grad = False
# rppc_model.fc_4y[0].bias.requires_grad = False

# Define Loss, Optimizer
loss_all = np.zeros((n_epochs,num_batches))
criterion = MSELoss()
optimizer = Adam(rppc_model.parameters(),lr=lr)
prev_loss = 0.1
prev_output = torch.zeros(batch,1)
print("epoch loop begins")
theta = 15
for epoch in range(1, n_epochs + 1): #loop for epochs
    permutation = torch.randperm(train_x.size()[0])
    j = 0
    print('\nEpoch: {}/{}.............'.format(epoch, n_epochs))
    for i in range(0,train_x.size()[0], batch): #loop for batches
#        print('\nBatch: {}/{}.............'.format(i, train_x.size()[0]))
        #Get current batch data
        indices = permutation[i:i+batch]
        batch_x, batch_y = train_x[indices], train_y[indices]
        output_ppc = torch.zeros(batch,time_steps, 3) # ppc output
        output_c = torch.zeros(batch,time_steps, 2) # m1 output
        output_p = torch.zeros(batch,time_steps, 2) # plant output
        output_e = torch.zeros(batch,time_steps, 2) # estimator output
        # output_s = torch.zeros(batch,time_steps, 2) # visual output
        for t in range(time_steps):
#            # estimator output
            output_e[:,t,0] = output_p[:,t-1,0]
            output_e[:,t,1] = output_p[:,t-1,1]
            # dist
            ppc_d = torch.zeros((1,1,4))
            ppc_d = torch.cat((output_e[:,t-1,0],output_e[:,t-1,1],output_e[:,t,0],output_e[:,t,1]),0)
            ppc_d = torch.reshape(ppc_d,(batch,1,4))
            # ppc_d = ppc_d.to(device)
            # X
            ppc_x = torch.zeros((batch,1,1,4))
            ppc_x = torch.cat((batch_x[:,t,0],output_ppc[:,t-1,0],output_e[:,t-1,0],output_e[:,t,0]),0)
            ppc_x = torch.reshape(ppc_x,(batch,1,4))
            # ppc_x = ppc_x.to(device)
            # Y
            ppc_y = torch.zeros((batch,1,1,4))
            ppc_y = torch.cat((batch_x[:,t,1],output_ppc[:,t-1,1],output_e[:,t-1,1],output_e[:,t,1]),0)
            ppc_y = torch.reshape(ppc_y,(batch,1,4))
            # ppc_y = ppc_y.to(device)
            
            outputppc = rppc_model(ppc_x,ppc_y,ppc_d,1)
            output_ppc[:,t,:] = torch.reshape(outputppc,(batch,3))
            # M1
            # X
            control_x = torch.zeros((batch,1,1,3))
            control_x = torch.cat((output_ppc[:,t,0],batch_x[:,t,2],output_ppc[:,t,2],output_c[:,t-1,0],output_e[:,t-1,0],output_e[:,t,0]),0)
            control_x = torch.reshape(control_x,(batch,1,6))
            # control_x = control_x.to(device)
            # Y
            control_y = torch.zeros((batch,1,1,3))
            control_y = torch.cat((output_ppc[:,t,1],batch_x[:,t,2],output_ppc[:,t,2],output_c[:,t-1,1],output_e[:,t-1,1],output_e[:,t,1]),0)
            control_y = torch.reshape(control_y,(batch,1,6))
            # control_y = control_y.to(device)
            outputc = m1_model(control_x,control_y)
            output_c[:,t,:] = torch.reshape(outputc,(batch,2))
            # plant model for x- and y-coordinate
            plant_x = torch.zeros(batch,1,1,6) # [uk, zk, z(k-1)] [T(t), o(t-1), o(t-2) ]
            plant_x = torch.cat((output_c[:,t,0],output_p[:,t-2,0],output_p[:,t-1,0],output_c[:,t,1],output_p[:,t-2,1],output_p[:,t-1,1]),0)
            plant_x = torch.reshape(plant_x,(batch,1,6))
            # plant_x = plant_x.to(device)
            op = plant_model(plant_x)
            op = torch.reshape(op,(batch,2))
            output_p[:,t,:] = op#torch.cat((op,output_ppc[:,t,2:]),1)#torch.reshape(opx,(batch,))
            # spx = op[0,0]*(1/2)#((math.cos(math.radians(theta)))*op[0,0] - (math.sin(math.radians(theta)))*op[0,1])*(1/2)            
            # spy = op[0,1]*(1/2)#((math.sin(math.radians(theta)))*op[0,0] + (math.cos(math.radians(theta)))*op[0,1])*(1/2)
            # output_s[:,t,0] = spx#torch.reshape(opx,(batch,))
            # output_s[:,t,1] = spy#torch.reshape(opy,(batch,))

        loss = criterion(output_p,batch_y) # Calculate loss
#        print("\nLoss: {:.8f}".format(loss.item()))
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step() # Updates the weights accordingly
        loss_all[epoch-1,j] = loss.item()
        j = j+1    
    # torch.save(rppc_model.state_dict(), "models/rppc_lppc/rppc/hypothesis/v1_gain_0_5/v7_only_dir_train_more_epochs/rppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_v7")
    avg_loss = np.sum(loss_all[epoch-1,:])/num_batches
    print("\nLoss:",avg_loss)
    if avg_loss < prev_loss:
        ## Save model
        torch.save(rppc_model.state_dict(), "models/rppc_lppc/rppc/extensions/1_normal_500")
        prev_loss = avg_loss    
        
# Save model
losses = np.zeros((n_epochs,1))
for i in range(n_epochs):
    losses[i,0] = np.sum(loss_all[i,:])/num_batches  
plt.figure()
plt.plot(losses)

#fname = "models/rppc_lppc/rppc/gain_adaptation/losses.xlsx"
#loss_gain = pd.read_excel(fname)
#loss_gain = np.array(loss_gain)
#x = np.arange(60)
#plt.figure()
#plt.plot(x,loss_gain[:,0],color='red',label='one direction')
#plt.plot(x,loss_gain[:,1],color='blue',label='two directions')
#plt.plot(x,loss_gain[:,2],color='green',label='four directions')
#plt.plot(x,loss_gain[:,3],color='yellow',label='eight directions')
#plt.legend()
#plt.xlabel('Number of training epochs')
#plt.ylabel('Error')
#plt.savefig("models/rppc_lppc/figures/extent_only/ga_training_loss.png")

#plt.figure()
##plt.plot(uk[0,:,0],uk[0,:,1],color='r')
##plt.plot(yk[0,:,0],yk[0,:,1],color='b')
#plt.plot(uk[0,:,3],color='black')