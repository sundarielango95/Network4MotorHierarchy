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
    #num_epochs = np.arange(1,151)
    #num_epochs = np.array((10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,
    #                       210,220,230,240,250,260,270,280,290,300,310,320))
    #num_epochs = np.array((40,80,160,320))
    #angles = np.zeros((2,len(num_epochs),20,8))
    #for m in range(2):
    #    e = 0
    #    for epoch in num_epochs:
    ##        epoch = 5
    #        # PPC model
    #        
    #        if m == 0:
    #            ppc_model = model_architecture.ctrlr_ppc()
    #            ppc_model.load_state_dict(torch.load("models/controller/rot_train/Left arm/L_PPC_v2_200_"+str(epoch)+"_epochs_L_arm"))
    #        else:
    #            ppc_model = model_architecture.ctrlr_ppc()
    #            ppc_model.load_state_dict(torch.load("models/controller/rot_train/Left arm/L_PPC_v2_200_"+str(epoch)+"_epochs_L_arm"))
    num_inputs = 8
    time_steps = 20
    # right arm
#    targets = np.array(([2,0],[2,1],[1,1],[0,1],[0,0],[0,-1],[1,-1],[2,-1])) # T1
#    targets = np.array(([1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1])) # T2
#    targets = np.array(([1,0],[1,0.5],[0.5,0.5],[0,0.5],[0,0],[0,-0.5],[0.5,-0.5],[1,-0.5])) # T3
    # left arm
#    targets = np.array(([0,0],[0,1],[-1,1],[-2,1],[-2,0],[-2,-1],[-1,-1],[0,-1])) # T1
#    targets = np.array(([1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1])) # T2
#    targets = np.array(([0,0],[0,0.5],[-0.5,0.5],[-1,0.5],[-1,0],[-1,-0.5],[-0.5,-0.5],[0,-0.5])) # T3
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
#    plt.savefig('paper/figures/rotated/right/'+str(mm+1)+'_new_targ_30.png')
#        plt.figure()
#        plt.plot(outputppc[0,:,0],c='b',label='PPC')
#        plt.plot(outputm1[0,:,0],c='g',label='M1')
#        plt.plot(outputp[0,:,0],c='r',label='plant')
#        plt.legend()

#vel = np.zeros((19,3))
#for i in range(19):
#    vel[i,0] = outputp[0,i,0]-outputp[0,i+1,0]
#    vel[i,1] = outputp[0,i,1]-outputp[0,i+1,1]
#    vel[i,2] = np.sqrt(vel[i,0]**2+vel[i,1]**2)
#plt.figure()
#plt.plot(vel[:,0],c='b',label='x vel')
#plt.plot(vel[:,1],c='r',label='y vel')
#plt.plot(vel[:,2],c='g',label='res vel')
#plt.legend()

#angle_final = np.zeros((2,len(num_epochs)+1,8)) # test + 320 training epochs
#angle_final[:,0,:] = 9.9831
#angle_vel = np.zeros((2,len(num_epochs)+1,8)) # test + 320 training epochs
#angle_vel[:,0,:] = 12.9256#5.10099
#for l in range(2):
#    for i in range(len(num_epochs)):
#        for j in range(8):
#            angle_final[l,i+1,j] = -1*angles[l,i,19,j]
#            angle_vel[l,i+1,j] = -1*angles[l,i,2,j]
#
###plt.figure()
#x = np.zeros((len(num_epochs)+1,1))
#x[1:,0] = num_epochs
##for i in range(8):
##    plt.plot(x,angle_final[0,:,i],label=str(i))
##    plt.plot(x,angle_final[1,:,i],label=str(i),linestyle='--')
##plt.title('Angle across epochs - Right Arm')
##plt.xlabel('Learning Trial Number')
##plt.ylabel('Directional Error (deg)')
###plt.savefig('figures/stimulation/v18/right_final_angles.png')
##plt.figure()
##for i in range(8):
##    plt.plot(x,angle_vel[0,:,i],label=str(i))
##    plt.plot(x,angle_vel[1,:,i],label=str(i),linestyle='--')
##plt.legend()
##plt.title('Angle across epochs - Right Arm')
##plt.xlabel('Learning Trial Number')
##plt.ylabel('Directional Error (deg)')
###plt.savefig('figures/stimulation/v18/right_pv_angles.png')
#
## mean across targets
#mean_angle = np.zeros((2,len(num_epochs)+1,2))
#for l in range(2):
#    for i in range(len(num_epochs)+1):
#        for j in range(8):
#            mean_angle[l,i,0] = np.sum(angle_final[l,i,:])/8
#            mean_angle[l,i,1] = np.sum(angle_vel[l,i,:])/8
## mean final angle
#plt.figure()
#plt.plot(x,mean_angle[0,:,0],c='b',label='R PPC Stim')
#plt.plot(x,mean_angle[1,:,0],c='r',label='R PPC Sham')
#plt.legend()
#plt.title('Angle across epochs')
#plt.xlabel('Learning Trial Number')
#plt.ylabel('Directional Error (deg)')
#plt.savefig('figures/stimulation/left_mean_final_angles.png')
## mean angle at peak velocity
#plt.figure()
#plt.plot(x,mean_angle[0,:,1],c='b',label='R PPC Stim')
#plt.plot(x,mean_angle[1,:,1],c='r',label='R PPC Sham')
#plt.legend()
#plt.title('Angle across epochs')
#plt.xlabel('Learning Trial Number')
#plt.ylabel('Directional Error (deg)')
#plt.savefig('figures/stimulation/left_mean_pv_angles.png')
#
##labels = ['Naive','40','80','160','320']
##xpos = np.arange(len(labels))
##fig=plt.figure()
##plt.bar(xpos,mean_angle[0,:,0],0.2,color='b')
##plt.title('Right Arm - Generalisation')
##plt.xticks(xpos,labels)
##plt.xlabel('Number of trials of Left arm')
##plt.ylabel('Directional Error')
##plt.savefig('figures/transfer/final_angles_avg_r_bar_v2.png')
##
##labels = ['Naive','40','80','160','320']
##xpos = np.arange(len(labels))
##fig=plt.figure()
##plt.bar(xpos,mean_angle[0,:,1],0.2,color='b')
##plt.title('Right Arm - Generalisation')
##plt.xticks(xpos,labels)
##plt.xlabel('Number of trials of Left arm')
##plt.ylabel('Directional Error')
##plt.savefig('figures/transfer/pv_angles_avg_r_bar_v2.png')