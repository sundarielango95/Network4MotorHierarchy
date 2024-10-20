# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 16:54:33 2023

@author: sundari
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import torch
import model_architecture
import math
import pandas as pd
import random

def angle(vector1, vector2):
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
#    ip = torch.round(inner_product*10**5)/(10**5)
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    if inner_product/(len1*len2) > 1:
        ip = 1
    else:
        ip = inner_product/(len1*len2)
    return -1*(math.degrees(math.acos(ip)))#*10**2)/(10**2)))

# plant model parameters
plant_model = model_architecture.plant_net_xandy()
plant_model.load_state_dict(torch.load("models/plant/plant_xandy_3_data_v1_random"))
num_subjects = 1
num_epochs = 320
num_targets = 8
time_steps = 20 
# re = np.zeros((num_subjects,num_epochs,num_targets))
# de = np.zeros((num_subjects,num_epochs,num_targets))
peak_vel = np.zeros((num_subjects,num_epochs,num_targets))
vel = np.zeros((num_epochs,num_subjects,time_steps-1,num_targets))
outputp_all = np.zeros((num_subjects,num_epochs,num_targets,time_steps,2))
dirs = np.array(([320,320,320,320,320,160,160,160,160,160,80,80,80,80,80,40,40,40,40,40]))
for sub_num in range(1,num_subjects+1):
    m1_model = model_architecture.ctrlr_m1()
    m1_model.load_state_dict(torch.load("models/rppc_lppc/m1/m1_dist_"+str(sub_num)+"_200"))
    epoch_dir = 1#dirs[sub_num-1]
    for epoch in range(1,epoch_dir+1):
        print(sub_num,epoch)
        lppc_model = model_architecture.lppc()
        lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/normal/lppc_v1_"+str(sub_num)+"_200_epochs"))
        rppc_model = model_architecture.rppc()
        rppc_model.load_state_dict(torch.load("models/rppc_lppc/rppc/PM/normal/rppc_v1_"+str(sub_num)+"_200_epochs"))          
        fem_model = model_architecture.race()
        fem_model.load_state_dict(torch.load("models/fem_model/trial_1_v3"))
        # PM model
        targets = np.array(([1,0],[0.7,0.8],[0,1],[-0.7,0.8],[-1,0],[-0.7,-0.8],[0,-1],[0.7,-0.8]))
        num_inputs = targets.shape[0]
        uk = np.zeros((num_inputs,time_steps,4),dtype='float32') # reference signal [Txl Tyl HSl Txr Tyr HSr] 
        yk = np.zeros((num_inputs,time_steps,2),dtype='float32') # next plant output o(t)
        for j in range(num_inputs):
            # common inputs
            uk[j,:,0] = targets[j,0]#data[i*time_steps:(i+1)*time_steps,0]#-data[i*time_steps:(i+1)*time_steps,2] # T(t)
            uk[j,:,1] = targets[j,1]#data[i*time_steps:(i+1)*time_steps,1]#-data[i*time_steps:(i+1)*time_steps,3] # T(t)
        test_xc = torch.from_numpy(uk)
        test_xc = torch.reshape(test_xc,[num_inputs,time_steps,1,4])
        outputp = torch.zeros((num_inputs,time_steps,1,2))
        outpute = torch.zeros((num_inputs,time_steps,1,2))
        outputs = torch.zeros((num_inputs,time_steps,1,2))
        outputm1 = torch.zeros((num_inputs,time_steps,1,2))
        outputlppc = torch.zeros((num_inputs,time_steps,1,3))
        outputrppc = torch.zeros((num_inputs,time_steps,1,3))
        
        for i in range(num_inputs):        
            for t in range(time_steps):
                # if t < 8:
                #     outpute[i,t,:,0] = outputp[i,t-1,:,0]
                #     outpute[i,t,:,1] = outputp[i,t-1,:,1]
                # else:
                #     outpute[i,t,:,0] = outputs[i,t-1,:,0]
                #     outpute[i,t,:,1] = outputs[i,t-1,:,1]
                # outpute[i,t,:,0] = outputp[i,t-1,:,0]
                # outpute[i,t,:,1] = outputp[i,t-1,:,1] 
                fem = torch.zeros((1,1,1,4))
                fem = torch.cat((outputp[i,t,:,0],outputp[i,t,:,1],outputs[i,t,:,0],outputs[i,t,:,1]),0)
                output_e = fem_model(fem)
                outpute[i,t,:,:] = output_e
                # PPC - theta       
                ppc_t = torch.zeros((1,1,4))
                ppc_t = torch.cat((test_xc[i,t,:,0],test_xc[i,t,:,1],outpute[i,t,:,0],outpute[i,t,:,1]),0)#,batch_x[:,t,0],batch_x[:,t,1]),0)
                # PPC - distance
                ppc_d = torch.zeros((1,1,4))
                ppc_d = torch.cat((outpute[i,t-1,:,0],outpute[i,t-1,:,1],outpute[i,t,:,0],outpute[i,t,:,1]),0)#outpute[i,t-1,:,1],outpute[i,t,:,0],outpute[i,t,:,1]),0) 
                # PPC - left
                xppc_l = torch.zeros(1,1,4)
                yppc_l = torch.zeros(1,1,4)
                xppc_l = torch.cat((test_xc[i,t,:,0],outputlppc[i,t-1,:,0],outpute[i,t-1,:,0],outpute[i,t,:,0]),0)
                yppc_l = torch.cat((test_xc[i,t,:,1],outputlppc[i,t-1,:,1],outpute[i,t-1,:,1],outpute[i,t,:,1]),0)        
                oppcl = lppc_model(xppc_l,yppc_l,ppc_t,1)
                outputlppc[i,t,0,:] = torch.reshape(oppcl,(1,1,3))
                # PPC - right
                xppc_r = torch.zeros(1,1,4)
                yppc_r = torch.zeros(1,1,4)
                xppc_r = torch.cat((test_xc[i,t,:,0],outputrppc[i,t-1,:,0],outpute[i,t-1,:,0],outpute[i,t,:,0]),0)
                yppc_r = torch.cat((test_xc[i,t,:,1],outputrppc[i,t-1,:,1],outpute[i,t-1,:,1],outpute[i,t,:,1]),0)        
                oppcr = rppc_model(xppc_r,yppc_r,ppc_d)#
                outputrppc[i,t,0,:] = torch.reshape(oppcr,(1,1,3))
                # MCM
                m1_x = torch.zeros((1,1,4))
                m1_x = torch.cat((outputlppc[i,t,:,0],outputlppc[i,t,:,2],outputrppc[i,t,:,2],outputm1[i,t-1,:,0],outpute[i,t-1,:,0],outpute[i,t,:,0]),0)
                m1_y = torch.zeros((1,1,4))
                m1_y = torch.cat((outputlppc[i,t,:,1],outputlppc[i,t,:,2],outputrppc[i,t,:,2],outputm1[i,t-1,:,1],outpute[i,t-1,:,1],outpute[i,t,:,1]),0)
                outputc = m1_model(m1_x,m1_y)
                outputm1[i,t,0,:] = torch.reshape(outputc,(1,1,2))
                # plant model for x- and y-coordinate
                plant_x = torch.zeros(1,1,6) # [uk, zk, z(k-1)] [T(t), o(t-1), o(t-2) ]
                plant_x = torch.cat((outputm1[i,t,:,0],outputp[i,t-2,:,0],outputp[i,t-1,:,0],outputm1[i,t,:,1],outputp[i,t-2,:,1],outputp[i,t-1,:,1]),0)
                op = plant_model(plant_x)
                outputp[i,t,0,:] = op
                spx = (((math.cos(math.radians(15)))*op[0]) - (math.sin(math.radians(15)))*op[1])           
                spy = (((math.sin(math.radians(15)))*op[0]) + (math.cos(math.radians(15)))*op[1])
                outputs[i,t,0,0] = spx#torch.reshape(opx,(batch,))
                outputs[i,t,0,1] = spy#torch.reshape(opy,(batch,))
                
        outputs = outputs.detach().numpy()
        outputs = np.reshape(outputs,[num_inputs,time_steps,2])
        outputp = outputp.detach().numpy()
        outputp = np.reshape(outputp,[num_inputs,time_steps,2])
        targets = np.reshape(targets,[num_inputs,2])
        outputp_all[sub_num-1,epoch-1,:,:,:] = outputp           
        # for kk in range(num_inputs):
        #     re[sub_num-1,epoch-1,kk] = (np.sqrt((targets[kk,0]-outputs[kk,19,0])**2+(targets[kk,1]-outputs[kk,19,1])**2))#/num_inputs
        #     de[sub_num-1,epoch-1,kk] = -1*angle(targets[kk,:],outputs[kk,2,:])

    # plotting
    # plot trajectory
    plt.figure()
    for i in range(num_inputs):
        # plt.figure()
        plt.scatter(targets[i,0],targets[i,1],color='r')
        plt.plot(outputp[i,:,0],outputp[i,:,1],'black')
        plt.plot(outputs[i,:,0],outputs[i,:,1],'blue')

# mean_de = np.average(de,axis=2)
