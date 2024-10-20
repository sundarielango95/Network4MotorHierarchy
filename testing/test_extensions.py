import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import matplotlib.pyplot as plt
import torch
import brain #type: ignore
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
plant_model = brain.plant_net_xandy()
plant_model.load_state_dict(torch.load("models/plant/plant_xandy_2_data_v1_random"))

num_subjects = 20

num_criteria = 6

targets = np.array(([1,0],[0.7,0.8],[0,1],[-0.7,0.8],[-1,0],[-0.7,-0.8],[0,-1],[0.7,-0.8]))
num_inputs = targets.shape[0]
time_steps = 20             
uk = np.zeros((num_inputs,time_steps,4),dtype='float32') # reference signal [Txl Tyl HSl Txr Tyr HSr] 
yk = np.zeros((num_inputs,time_steps,2),dtype='float32') # next plant output o(t)
for j in range(num_inputs):
    # common inputs
    uk[j,:,0] = targets[j,0]#data[i*time_steps:(i+1)*time_steps,0]#-data[i*time_steps:(i+1)*time_steps,2] # T(t)
    uk[j,:,1] = targets[j,1]#data[i*time_steps:(i+1)*time_steps,1]#-data[i*time_steps:(i+1)*time_steps,3] # T(t)

    test_xc = torch.from_numpy(uk)
test_xc = torch.reshape(test_xc,[num_inputs,time_steps,1,4])

re = np.zeros((num_subjects,num_criteria,num_inputs))
de = np.zeros((num_subjects,num_criteria,num_inputs))
peak_vel = np.zeros((num_criteria,num_inputs))
vel = np.zeros((num_criteria,time_steps-1,num_inputs))

outputp = torch.zeros((num_subjects,num_criteria,num_inputs,time_steps,1,2))
# outputm1 = torch.zeros((num_criteria,num_inputs,time_steps,1,2))
# outputlppc = torch.zeros((num_criteria,num_inputs,time_steps,1,3))
# outputrppc = torch.zeros((num_criteria,num_inputs,time_steps,1,3))

for sub_num in range(num_subjects):

    # m1 model parameters
    m1_model = brain.ctrlr_m1()
    m1_model.load_state_dict(torch.load("models/rppc_lppc/m1/m1_dist_"+str(sub_num+1)+"_200"))

    # LPPC and RPPC
    lppc_model = brain.lppc()
    lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/normal/lppc_v1_"+str(sub_num+1)+"_500_epochs"))
    rppc_model = brain.rppc()
    rppc_model.load_state_dict(torch.load("models/rppc_lppc/rppc/PM/normal/rppc_v1_"+str(sub_num+1)+"_500_epochs"))    

    outputm1 = torch.zeros((num_criteria,num_inputs,time_steps,1,2))
    outputlppc = torch.zeros((num_criteria,num_inputs,time_steps,1,3))
    outputrppc = torch.zeros((num_criteria,num_inputs,time_steps,1,3))

    for num in range(num_criteria):
        for i in range(num_inputs):        
            for t in range(time_steps):
                # PPC - theta       
                ppc_t = torch.zeros((1,1,4))
                ppc_t = torch.cat((test_xc[i,t,:,0],test_xc[i,t,:,1],outputp[sub_num,num,i,t,:,0],outputp[sub_num,num,i,t,:,1]),0)#,batch_x[:,t,0],batch_x[:,t,1]),0)
                # PPC - distance
                ppc_d = torch.zeros((1,1,4))
                ppc_d = torch.cat((outputp[sub_num,num,i,t-1,:,0],outputp[sub_num,num,i,t-1,:,1],outputp[sub_num,num,i,t,:,0],outputp[sub_num,num,i,t,:,1]),0)#outpute[i,t-1,:,1],outpute[i,t,:,0],outpute[i,t,:,1]),0) 
                # PPC - left
                xppc_l = torch.zeros(1,1,4)
                yppc_l = torch.zeros(1,1,4)
                xppc_l = torch.cat((test_xc[i,t,:,0],outputlppc[num,i,t-1,:,0],outputp[sub_num,num,i,t-1,:,0],outputp[sub_num,num,i,t,:,0]),0)
                yppc_l = torch.cat((test_xc[i,t,:,1],outputlppc[num,i,t-1,:,1],outputp[sub_num,num,i,t-1,:,1],outputp[sub_num,num,i,t,:,1]),0)        
                oppcl = lppc_model(xppc_l,yppc_l,ppc_t,num)
                outputlppc[num,i,t,0,:] = torch.reshape(oppcl,(1,1,1,3))
                # PPC - right
                xppc_r = torch.zeros(1,1,4)
                yppc_r = torch.zeros(1,1,4)
                xppc_r = torch.cat((test_xc[i,t,:,0],outputrppc[num,i,t-1,:,0],outputp[sub_num,num,i,t-1,:,0],outputp[sub_num,num,i,t,:,0]),0)
                yppc_r = torch.cat((test_xc[i,t,:,1],outputrppc[num,i,t-1,:,1],outputp[sub_num,num,i,t-1,:,1],outputp[sub_num,num,i,t,:,1]),0)        
                oppcr = rppc_model(xppc_r,yppc_r,ppc_d,num)#
                outputrppc[num,i,t,0,:] = torch.reshape(oppcr,(1,1,1,3))
                # MCM
                if num < 3:# LEFT ARM
                    m1_x = torch.zeros((1,1,4))
                    m1_x = torch.cat((outputlppc[num,i,t,:,0],outputlppc[num,i,t,:,2],outputrppc[num,i,t,:,2],outputm1[num,i,t-1,:,0],outputp[sub_num,num,i,t-1,:,0],outputp[sub_num,num,i,t,:,0]),0)
                    m1_y = torch.zeros((1,1,4))
                    m1_y = torch.cat((outputlppc[num,i,t,:,1],outputlppc[num,i,t,:,2],outputrppc[num,i,t,:,2],outputm1[num,i,t-1,:,1],outputp[sub_num,num,i,t-1,:,1],outputp[sub_num,num,i,t,:,1]),0)
                    outputc = m1_model(m1_x,m1_y)
                    outputm1[num,i,t,0,:] = torch.reshape(outputc,(1,1,1,2))

                else: # RIGHT ARM
                    m1_x = torch.zeros((1,1,4))
                    m1_x = torch.cat((outputrppc[num,i,t,:,0],outputlppc[num,i,t,:,2],outputrppc[num,i,t,:,2],outputm1[num,i,t-1,:,0],outputp[sub_num,num,i,t-1,:,0],outputp[sub_num,num,i,t,:,0]),0)
                    m1_y = torch.zeros((1,1,4))
                    m1_y = torch.cat((outputrppc[num,i,t,:,1],outputlppc[num,i,t,:,2],outputrppc[num,i,t,:,2],outputm1[num,i,t-1,:,1],outputp[sub_num,num,i,t-1,:,1],outputp[sub_num,num,i,t,:,1]),0)
                    outputc = m1_model(m1_x,m1_y)
                    outputm1[num,i,t,0,:] = torch.reshape(outputc,(1,1,2))
                
                # plant model for x- and y-coordinate
                plant_x = torch.zeros(1,1,6) # [uk, zk, z(k-1)] [T(t), o(t-1), o(t-2) ]
                plant_x = torch.cat((outputm1[num,i,t,:,0],outputp[sub_num,num,i,t-2,:,0],outputp[sub_num,num,i,t-1,:,0],outputm1[num,i,t,:,1],outputp[sub_num,num,i,t-2,:,1],outputp[sub_num,num,i,t-1,:,1]),0)
                op = plant_model(plant_x)
                outputp[num,i,t,0,:] = op
    
outputp = outputp.detach().numpy()
outputp = np.reshape(outputp,[num_subjects,num_criteria,num_inputs,time_steps,1,2])
targets = np.reshape(targets,[num_inputs,2])

for sub_num in range(num_subjects):
    for num in range(num_criteria):
        for kk in range(num_inputs):
            re[sub_num,num,kk] = (np.sqrt((targets[kk,0]-outputp[sub_num,num,kk,19,0,0])**2+(targets[kk,1]-outputp[sub_num,num,kk,19,0,1])**2))#/num_inputs
            de[sub_num,num,kk] = -1*angle(targets[kk,:],outputp[sub_num,num,kk,2,0,:])
            for tt in range(time_steps-1):
                vel[sub_num,num,tt,kk] = np.sqrt((outputp[sub_num,num,kk,tt+1,0,0]-outputp[sub_num,num,kk,tt,0,0])**2+(outputp[sub_num,num,kk,tt+1,0,1]-outputp[sub_num,num,kk,tt,0,1])**2)
            peak_vel[sub_num,num,kk] = np.max(vel[num,:,kk])
            
for sub_num in range(num_subjects):
    for i in range(num_criteria):
        plt.figure()
        for k in range(num_inputs):
            plt.scatter(targets[k,0],targets[k,1],color='r')
            plt.plot(outputp[sub_num,i,k,:,0,0],outputp[sub_num,i,k,:,0,1],'black')

re_mean = np.mean(re,axis=1)
de_mean = np.mean(de,axis=1)

re_mean_mean = np.mean(re_mean,axis=0)
de_mean_mean = np.mean(de_mean,axis=0)

# x = np.arange(6)
# plt.figure()
# plt.scatter(x,re_mean,color='blue')
# plt.xticks(x,['N_LH', 'DLPPC_LH','DRPPC_LH','N_RH', 'DLPPC_RH','DRPPC_RH'])
# plt.title('Reaching Error')    
# plt.figure()
# plt.scatter(x,de_mean,color='red')
# plt.xticks(x,['N_LH', 'DLPPC_LH','DRPPC_LH','N_RH', 'DLPPC_RH','DRPPC_RH'])
# plt.title('Directional Error')    

re_plot = np.zeros((4,1))
de_plot = np.zeros((4,1))

re_plot[0,0] = re_mean[0]
re_plot[1,0] = re_mean[2]
re_plot[2,0] = re_mean[5]
re_plot[3,0] = re_mean[3]

de_plot[0,0] = de_mean[3]
de_plot[1,0] = de_mean[4]
de_plot[2,0] = de_mean[1]
de_plot[3,0] = de_mean[0]

x = np.arange(4)
plt.figure()
plt.scatter(x,re_plot,color='blue')
plt.xticks(x,['N_LH', 'P_LH','NP_RH','N_RH'])
plt.title('Reaching Error')    
plt.figure()
plt.scatter(x,de_plot,color='red')
plt.xticks(x,['N_RH', 'P_RH','NP_LH','N_LH'])
plt.title('Directional Error')    
