# -*- coding: utf-8 -*-
"""
Created on Mon May  9 11:40:23 2022

@author: Sundari Elango
"""

import numpy as np
import model_architecture
import matplotlib.pyplot as plt
import torch
import math

def angle(vector1, vector2):
    x1, y1 = vector1
    x2, y2 = vector2
    inner_product = x1*x2 + y1*y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return (math.degrees(math.acos(inner_product/(len1*len2)))*-1)

# Plant model
plant_model = model_architecture.plant_net_v1()
plant_model.load_state_dict(torch.load("models/plant/plant_1_v1"))
for mm in range(20):
    # M1 model
    m1_model = model_architecture.ctrlr_m1()
    m1_model.load_state_dict(torch.load("models/controller/normal/Right arm/L_M1_4_R_arm_200_"+str(mm+1)))
    # PPC Model
    ppc_model = model_architecture.ctrlr_ppc()
    ppc_model.load_state_dict(torch.load("models/controller/normal/Right arm/L_PPC_v2_200_"+str(mm+1)))
    num_inputs = 8
    time_steps = 20
    targets = np.array(([1,0],[0.7,0.8],[0,1],[-0.7,0.8],[-1,0],[-0.7,-0.8],[0,-1],[0.7,-0.8])) # T2
    #targets_rot = np.zeros((8,2))
    #for j in range(8):
    #    targets_rot[j,0] = (math.cos(math.radians(-15)))*targets[j,0] - (math.sin(math.radians(-15)))*targets[j,1]
    #    targets_rot[j,1] = (math.sin(math.radians(-15)))*targets[j,0] + (math.cos(math.radians(-15)))*targets[j,1]
    X = np.ones((num_inputs,time_steps,1,4),dtype='float32')
    plt.figure()
    for k in range(num_inputs):
        X[k,:,0,0] = targets[k,0]
        X[k,:,0,1] = targets[k,1]
        X[k,:,0,2] = 0
        X[k,:,0,3] = 0
        test_xc = torch.from_numpy(X[k,:,0,:])
        test_xc = torch.reshape(test_xc,[20,1,4])
        outputm1 = torch.zeros(1,time_steps,1,2)
        outputppc = torch.zeros(1,time_steps,1,2)
        outputs = torch.zeros(1,time_steps,1,2)
        outputp = torch.zeros(1,time_steps,1,2)
        outpute = torch.zeros(1,time_steps,1,2)
        theta = 15; thresh = 4
    #    plt.figure()
    #for i in range(num_inputs):
        i = 0
        outputp[i,:,0,0] = test_xc[:,0,2]
        outputp[i,:,0,1] = test_xc[:,0,3]
        for t in range(time_steps):
            if t < thresh:
                outpute[0,t,0,0] = outputp[0,t-1,0,0]
                outpute[0,t,0,1] = outputp[0,t-1,0,1]
            else:
                outpute[0,t,0,0] = outputs[0,t-1,0,0]
                outpute[0,t,0,1] = outputs[0,t-1,0,1]
#            outpute[0,t,0,0] = outputp[0,t-1,0,0]
#            outpute[0,t,0,1] = outputp[0,t-1,0,1]
    #        i=0
            # PPC
            xppc = torch.zeros(1,1,4)
            yppc = torch.zeros(1,1,4)
            xppc = torch.cat((test_xc[t,:,0],outputppc[i,t-1,:,0],outpute[i,t-1,:,0],outpute[i,t,:,0]),0)
            yppc = torch.cat((test_xc[t,:,1],outputppc[i,t-1,:,1],outpute[i,t-1,:,1],outpute[i,t,:,1]),0)        
    #        if m == 0:
    #            oppc = ppc_model(xppc,yppc)
    #        else:
            oppc = ppc_model(xppc,yppc)
            outputppc[i,t,0,:] = oppc 
            # M1
            xm = torch.zeros(1,1,4)
            ym = torch.zeros(1,1,4)
            xm = torch.cat((outputppc[i,t,:,0],outputm1[i,t-1,:,0],outpute[i,t-1,:,0],outpute[i,t,:,0]),0)
            ym = torch.cat((outputppc[i,t,:,1],outputm1[i,t-1,:,1],outpute[i,t-1,:,1],outpute[i,t,:,1]),0)        
            om = m1_model(xm,ym)
            outputm1[i,t,0,:] = om 
            # Plant
            xp = torch.zeros(1,1,3)
            xp = torch.cat((outputm1[i,t,:,0],outputp[i,t-2,:,0],outputp[i,t-1,:,0]),0)
            fpx = plant_model(xp)        
            yp = torch.zeros(1,1,3)
            yp = torch.cat((outputm1[i,t,:,1],outputp[i,t-2,:,1],outputp[i,t-1,:,1]),0)
            fpy = plant_model(yp)
            
            outputp[:,t,0,0] = fpx#torch.reshape(opx,(batch,))
            outputp[:,t,0,1] = fpy#torch.reshape(opy,(batch,))
            spx = (math.cos(math.radians(theta)))*fpx - (math.sin(math.radians(theta)))*fpy            
            spy = (math.sin(math.radians(theta)))*fpx + (math.cos(math.radians(theta)))*fpy
            outputs[:,t,0,0] = spx#torch.reshape(opx,(batch,))
            outputs[:,t,0,1] = spy#torch.reshape(opy,(batch,))
            
        outputppc = outputppc.detach().numpy()
        outputppc = np.reshape(outputppc,[1,time_steps,2])
        outputm1 = outputm1.detach().numpy()
        outputm1 = np.reshape(outputm1,[1,time_steps,2])
        outputp = outputp.detach().numpy()
        outputp = np.reshape(outputp,[1,time_steps,2])
        outputs = outputs.detach().numpy()
        outputs = np.reshape(outputs,[1,time_steps,2])
        test_xc = test_xc.detach().numpy()
        test_xc = np.reshape(test_xc,[1,time_steps,4])
    #    for j in range(20):
    #        angles[m,e,j,k] = angle(targets[k,:],outputs[0,j,:])
    #e = e+1
    #plt.figure()
    #    for j in range(num_inputs):
        plt.scatter(targets[k,0],targets[k,1],color='r')
        plt.plot(outputp[i,:,0],outputp[i,:,1],color='b',label='hand')
        plt.plot(outputs[i,:,0],outputs[i,:,1],color='r',label='cursor')
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.title('Right Arm')
    plt.xlabel('x')
    plt.ylabel('y')      
