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
    return (math.degrees(math.acos(inner_product/(len1*len2))))

# Plant model
plant_model = model_architecture.plant_net_xandy()
plant_model.load_state_dict(torch.load("models/plant/plant_xandy_3_data_v1_random"))
#num_epochs = np.arange(1,151)
num_trials = 150
num_subjects = 5
angles = np.zeros((2,num_subjects,num_trials,2,8))
trial_groups = 2
for trial_group in range(trial_groups):
    for subject in range(num_subjects):
        for trial in range(num_trials):
            # M1 model
            m1_model = model_architecture.ctrlr_m1()
            m1_model.load_state_dict(torch.load("models/controller/with_xandy_plant/normal/lmc_rarm/mc_"+str(subject+1)+"_plant_xandy_3_data_random"))
            # PPC model
            if trial_group == 0:
                ppc_model = model_architecture.ctrlr_ppc()
                ppc_model.load_state_dict(torch.load("models/controller/with_xandy_plant/rot_train/left/320/ppc_"+str(subject+1)+"_mc_"+str(subject+1)+"_plant_xandy_3_data_random_"+str(trial+1)+"_epochs"))
            else:
                ppc_model = model_architecture.ctrlr_ppc_stim()
                ppc_model.load_state_dict(torch.load("models/controller/with_xandy_plant/stimulation/ppc_"+str(subject+1)+"_mc_"+str(subject+1)+"_plant_xandy_3_data_random_"+str(trial+1)+"_epochs"))
            num_inputs = 8
            time_steps = 20
            targets = np.array(([1,0],[0.7,0.8],[0,1],[-0.7,0.8],[-1,0],[-0.7,-0.8],[0,-1],[0.7,-0.8])) # T2
            X = np.ones((num_inputs,time_steps,1,4),dtype='float32')
    #        plt.figure()
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
                theta = 15; thresh = 8
                i = 0
                outputp[i,:,0,0] = test_xc[:,0,2]
                outputp[i,:,0,1] = test_xc[:,0,3]
                for t in range(time_steps):
                    outpute[0,t,0,0] = outputs[0,t-1,0,0]
                    outpute[0,t,0,1] = outputs[0,t-1,0,1]
                    # PPC
                    xppc = torch.zeros(1,1,4)
                    yppc = torch.zeros(1,1,4)
                    xppc = torch.cat((test_xc[t,:,0],outputppc[i,t-1,:,0],outpute[i,t-1,:,0],outpute[i,t,:,0]),0)
                    yppc = torch.cat((test_xc[t,:,1],outputppc[i,t-1,:,1],outpute[i,t-1,:,1],outpute[i,t,:,1]),0)        
                    if trial_group == 0:
                        oppc = ppc_model(xppc,yppc)
                        outputppc[i,t,0,:] = oppc 
                    else:
                        oppc = ppc_model(xppc,yppc,1)
                        outputppc[i,t,0,:] = oppc 
                    # M1
                    xm = torch.zeros(1,1,4)
                    ym = torch.zeros(1,1,4)
                    xm = torch.cat((outputppc[i,t,:,0],outputm1[i,t-1,:,0],outpute[i,t-1,:,0],outpute[i,t,:,0]),0)
                    ym = torch.cat((outputppc[i,t,:,1],outputm1[i,t-1,:,1],outpute[i,t-1,:,1],outpute[i,t,:,1]),0)        
                    om = m1_model(xm,ym)
                    outputm1[i,t,0,:] = om 
                    # Plant
                    xp = torch.zeros(1,1,6)
                    xp = torch.cat((outputm1[i,t,:,0],outputp[i,t-2,:,0],outputp[i,t-1,:,0],outputm1[i,t,:,1],outputp[i,t-2,:,1],outputp[i,t-1,:,1]),0)
                    fpx = plant_model(xp)        
        #            yp = torch.zeros(1,1,3)
        #            yp = torch.cat((outputm1[i,t,:,1],outputp[i,t-2,:,1],outputp[i,t-1,:,1]),0)
        #            fpy = plant_model(yp)
                    
                    outputp[:,t,0,:] = fpx#torch.reshape(opx,(batch,))
        #            outputp[:,t,0,1] = fpy#torch.reshape(opy,(batch,))
                    spx = (math.cos(math.radians(theta)))*fpx[0] - (math.sin(math.radians(theta)))*fpx[1]            
                    spy = (math.sin(math.radians(theta)))*fpx[0] + (math.cos(math.radians(theta)))*fpx[1]
                    outputs[:,t,0,0] = spx#torch.reshape(opx,(batch,))
                    outputs[:,t,0,1] = spy#torch.reshape(opy,(batch,))
                
                outputs = outputs.detach().numpy()
                outputs = np.reshape(outputs,[1,time_steps,2])
                outputp = outputp.detach().numpy()
                outputp = np.reshape(outputp,[1,time_steps,2])
                angles[trial_group,subject,trial,0,k] = angle(targets[k,:],outputs[0,19,:])
                angles[trial_group,subject,trial,1,k] = angle(targets[k,:],outputs[0,5,:])
#            plt.scatter(targets[k,0],targets[k,1],color='r')
#            plt.plot(outputp[0,:,0],outputp[0,:,1],color='b',label='hand')
#            plt.plot(outputs[0,:,0],outputs[0,:,1],color='r',label='cursor')
#        plt.xlim([-1.5,1.5])
#        plt.ylim([-1.5,1.5])
#        plt.title('Right Arm')
#        plt.xlabel('x')
#        plt.ylabel('y')      
#        plt.savefig('figures/with_xandy_plant/trained/right/'+str(nEp+1)+'_'+str(num_epochs[epoch])+'_epoch.png')

# mean across targets
mean_angle = np.zeros((trial_groups,num_trials,2))        
for i in range(trial_groups):
    for j in range(num_trials):
        a = np.zeros((num_subjects,2))
        for k in range(num_subjects):
            a[k,0] = np.sum(angles[i,k,j,0,:])/8
            a[k,1] = np.sum(angles[i,k,j,1,:])/8
        mean_angle[i,j,0] = np.sum(a[:,0])/num_subjects
        mean_angle[i,j,1] = np.sum(a[:,1])/num_subjects
        
## mean across targets
#mean_angle = np.zeros((320,2))
#for i in range(320):
#    mean_angle[i,0] = np.sum(angles[i,0,:])/8
#    mean_angle[i,1] = np.sum(angles[i,1,:])/8
#
#x = np.arange(1,321)
#plt.figure()
#plt.plot(x,angles[:,0,2],color='r')
#plt.title('Angle across epochs - Right Arm')
#plt.xlabel('Learning Trial Number')
#plt.ylabel('Directional Error (deg)')
#plt.savefig('figures/ppc_1_rot_trained_final_transfer.png')
#
#plt.figure()
#plt.plot(x,angles[:,0,2],color='r')
#plt.title('Angle across epochs - Right Arm')
#plt.xlabel('Learning Trial Number')
#plt.ylabel('Directional Error (deg)')
#plt.savefig('figures/ppc_1_rot_trained_peak_transfer.png')

x = np.arange(1,151)
plt.figure()
plt.plot(x,mean_angle[0,:,0],color='k',label='Left Sham')
plt.plot(x,mean_angle[1,:,0],color='b',label='Left PPC')
plt.legend()
plt.title('Angle across epochs - Right Arm')
plt.xlabel('Learning Trial Number')
plt.ylabel('Directional Error (deg)')
plt.savefig('figures/with_xandy_plant/stimulated/right/final_angles.png')

plt.figure()
plt.plot(x,mean_angle[0,:,1],color='k',label='Left Sham')
plt.plot(x,mean_angle[1,:,1],color='b',label='Left PPC')
plt.legend()
plt.title('Angle across epochs - Right Arm')
plt.xlabel('Learning Trial Number')
plt.ylabel('Directional Error (deg)')
plt.savefig('figures/with_xandy_plant/stimulated/right/pv_angles.png')
