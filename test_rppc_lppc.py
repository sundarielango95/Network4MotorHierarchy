# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 09:36:05 2022

@author: Sundari Elango
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
plant_model.load_state_dict(torch.load("models/plant/plant_xandy_2_data_v1_random"))
num_criteria = 4
num_subjects = 5
num_epochs = 150
num_targets = 8
time_steps = 20 
re = np.zeros((num_criteria,num_subjects,num_epochs,num_targets))
de = np.zeros((num_criteria,num_subjects,num_epochs,num_targets))
peak_vel = np.zeros((num_subjects,num_epochs,num_targets))
vel = np.zeros((num_epochs,num_subjects,time_steps-1,num_targets))
outputp_all = np.zeros((num_criteria,num_subjects,num_epochs,num_targets,time_steps,2))
dirs = np.array(([320,320,320,320,320,
                  160,160,160,160,160,
                  80,80,80,80,80,
                  40,40,40,40,40]))

for num_crit in range(num_criteria):
    for sub_num in range(1,num_subjects+1):
        # sub_num = 4
        m1_model = model_architecture.ctrlr_m1()
        m1_model.load_state_dict(torch.load("models/rppc_lppc/m1/m1_dist_"+str(sub_num)+"_200"))
        epoch_dir = num_epochs#dirs[sub_num-1]
        e = 0
        for epoch in range(1,epoch_dir+1):
            print(sub_num,epoch,num_crit)
            # if num_crit == 0: # Naive/Sham group
            # lppc_model = model_architecture.lppc()
            # lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/rotated/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_rotated_v2"))
            # if num_crit == 0:
            #     lppc_model = model_architecture.lppc()
            #     lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/stimulated/new_new/v1_only_dir_train_w_gf_in_dir_0_25/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_v1"))  
            #     stim_gain_b = 1
            #     stim_gain_c = 0.25
            # elif num_crit == 1:
            #     lppc_model = model_architecture.lppc()
            #     lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/stimulated/new_new/v2_only_dir_train_w_gf_in_dir_0_5/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_v2"))          
            #     stim_gain_b = 1
            #     stim_gain_c = 0.5
            # elif num_crit == 2:
            #     lppc_model = model_architecture.lppc()
            #     lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/stimulated/new_new/v3_only_dir_train_w_gf_in_dir_0_75/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_v3"))          
            #     stim_gain_b = 1
            #     stim_gain_c = 0.75
            # elif num_crit == 3:
            #     lppc_model = model_architecture.lppc()
            #     lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/stimulated/new_new/v4_only_dir_train_w_gf_in_dir_0_025/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_v4"))          
            #     stim_gain_b = 1
            #     stim_gain_c = 0.025
            # elif num_crit == 4:
            #     lppc_model = model_architecture.lppc()
            #     lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/stimulated/new_new/v5_only_dir_train_w_gf_in_dir_0_05/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_v5"))          
            #     stim_gain_b = 1
            #     stim_gain_c = 0.05
            # elif num_crit == 5:
            #     lppc_model = model_architecture.lppc()
            #     lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/stimulated/new_new/v6_only_dir_train_w_gf_in_dir_0_075/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_v6"))          
            #     stim_gain_b = 1
            #     stim_gain_c = 0.075
            # if num_crit == 0:
            #     lppc_model = model_architecture.lppc()
            #     lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/stimulated/new_new/v16_only_dir_train_w_gf_in_all_0_25/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_v16"))          
            #     stim_gain_b = 0.25
            #     stim_gain_c = 0.25
            # elif num_crit == 1:
            #     lppc_model = model_architecture.lppc()
            #     lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/stimulated/new_new/v17_only_dir_train_w_gf_in_all_0_5/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_v17"))          
            #     stim_gain_b = 0.5
            #     stim_gain_c = 0.5
            # lppc_model = model_architecture.lppc()
            # lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/rotated/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_rotated_v2"))
            if num_crit == 0 or num_crit == 2: # normal/ sham stim
                lppc_model = model_architecture.lppc()
                lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/rotated/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_rotated_v2"))
                rppc_model = model_architecture.rppc()
                rppc_model.load_state_dict(torch.load("models/rppc_lppc/rppc/PM/normal/rppc_v1_"+str(sub_num)+"_200_epochs"))                          
                stim_gain_b = 1 # LPPC GF
                stim_gain_c = 1 # RPPC GF
            elif num_crit == 1: # LPPC stim
                lppc_model = model_architecture.lppc()
                lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/stimulated/new_new/v18_only_dir_train_w_gf_in_all_0_75/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_v18"))          
                rppc_model = model_architecture.rppc()
                rppc_model.load_state_dict(torch.load("models/rppc_lppc/rppc/PM/normal/rppc_v1_"+str(sub_num)+"_200_epochs"))          
                stim_gain_b = 0.75
                stim_gain_c = 1
            else: # RPPC STIM
                lppc_model = model_architecture.lppc()
                lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/rotated/lppc_v1_"+str(sub_num)+"_"+str(epoch)+"_epochs_rotated_v2"))
                rppc_model = model_architecture.rppc()
                rppc_model.load_state_dict(torch.load("models/rppc_lppc/rppc/PM/normal/rppc_v1_"+str(sub_num)+"_200_epochs"))                          
                stim_gain_b = 1
                stim_gain_c = 0.75
            # else: # Transfer/Stim group
            #     lppc_model = model_architecture.lppc()
            #     lppc_model.load_state_dict(torch.load("models/rppc_lppc/lppc/PM/stimulated/new/lppc_v1_"+str(6)+"_"+str(epoch)+"_epochs_stimulated_v1"))
            #     rppc_model = model_architecture.rppc()
            #     rppc_model.load_state_dict(torch.load("models/rppc_lppc/rppc/PM/normal/rppc_v1_"+str(6)+"_200_epochs"))          
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
                    # if num_crit == 0:
                    #     if t < 8:
                    #         outpute[i,t,:,0] = outputp[i,t-1,:,0]
                    #         outpute[i,t,:,1] = outputp[i,t-1,:,1]
                    #     else:
                    #         outpute[i,t,:,0] = outputs[i,t-1,:,0]
                    #         outpute[i,t,:,1] = outputs[i,t-1,:,1]
                    # else:
                    outpute[i,t,:,0] = outputs[i,t-1,:,0]
                    outpute[i,t,:,1] = outputs[i,t-1,:,1] 
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
                    oppcl = lppc_model(xppc_l,yppc_l,ppc_t,stim_gain_b)
                    outputlppc[i,t,0,:] = torch.reshape(oppcl,(1,1,3))
                    # PPC - right
                    xppc_r = torch.zeros(1,1,4)
                    yppc_r = torch.zeros(1,1,4)
                    xppc_r = torch.cat((test_xc[i,t,:,0],outputrppc[i,t-1,:,0],outpute[i,t-1,:,0],outpute[i,t,:,0]),0)
                    yppc_r = torch.cat((test_xc[i,t,:,1],outputrppc[i,t-1,:,1],outpute[i,t-1,:,1],outpute[i,t,:,1]),0)        
                    oppcr = rppc_model(xppc_r,yppc_r,ppc_d,stim_gain_c)#
                    outputrppc[i,t,0,:] = torch.reshape(oppcr,(1,1,3))
                    # MCM
                    # if num_crit == 0  or num_crit == 2 or num_crit == 4: # LEFT ARM
                    m1_x = torch.zeros((1,1,4))
                    m1_x = torch.cat((outputlppc[i,t,:,0],outputlppc[i,t,:,2],outputrppc[i,t,:,2],outputm1[i,t-1,:,0],outpute[i,t-1,:,0],outpute[i,t,:,0]),0)
                    m1_y = torch.zeros((1,1,4))
                    m1_y = torch.cat((outputlppc[i,t,:,1],outputlppc[i,t,:,2],outputrppc[i,t,:,2],outputm1[i,t-1,:,1],outpute[i,t-1,:,1],outpute[i,t,:,1]),0)
                    outputc = m1_model(m1_x,m1_y)
                    outputm1[i,t,0,:] = torch.reshape(outputc,(1,1,2))
                    # else: # RIGHT ARM
                    #     m1_x = torch.zeros((1,1,4))
                    #     m1_x = torch.cat((outputlppc[i,t,:,0],outputlppc[i,t,:,2],outputrppc[i,t,:,2],outputm1[i,t-1,:,0],outpute[i,t-1,:,0],outpute[i,t,:,0]),0)
                    #     m1_y = torch.zeros((1,1,4))
                    #     m1_y = torch.cat((outputlppc[i,t,:,1],outputlppc[i,t,:,2],outputrppc[i,t,:,2],outputm1[i,t-1,:,1],outpute[i,t-1,:,1],outpute[i,t,:,1]),0)
                    #     outputc = m1_model(m1_x,m1_y)
                    #     outputm1[i,t,0,:] = torch.reshape(outputc,(1,1,2))
                    
                    # plant model for x- and y-coordinate
                    plant_x = torch.zeros(1,1,6) # [uk, zk, z(k-1)] [T(t), o(t-1), o(t-2) ]
                    plant_x = torch.cat((outputm1[i,t,:,0],outputp[i,t-2,:,0],outputp[i,t-1,:,0],outputm1[i,t,:,1],outputp[i,t-2,:,1],outputp[i,t-1,:,1]),0)
                    op = plant_model(plant_x)
                    outputp[i,t,0,:] = op
                    spx = ((((math.cos(math.radians(15)))*op[0]) - (math.sin(math.radians(15)))*op[1]))#*(1/2)
                    spy = ((((math.sin(math.radians(15)))*op[0]) + (math.cos(math.radians(15)))*op[1]))#*(1/2)
                    outputs[i,t,0,0] = spx#torch.reshape(opx,(batch,))
                    outputs[i,t,0,1] = spy#torch.reshape(opy,(batch,))
                    
            outputs = outputs.detach().numpy()
            outputs = np.reshape(outputs,[num_inputs,time_steps,2])
            outputp = outputp.detach().numpy()
            outputp = np.reshape(outputp,[num_inputs,time_steps,2])
            targets = np.reshape(targets,[num_inputs,2])
            # outputp_all[num_crit,sub_num-1,epoch-1,:,:,:] = outputp           
            for kk in range(num_inputs):
                re[num_crit,sub_num-1,e,kk] = (np.sqrt((targets[kk,0]-outputs[kk,19,0])**2+(targets[kk,1]-outputs[kk,19,1])**2))#/num_inputs
                de[num_crit,sub_num-1,e,kk] = -1*angle(targets[kk,:],outputs[kk,2,:])
                for tt in range(time_steps-1):
                    vel[e,sub_num-1,tt,kk] = np.sqrt((outputp[kk,tt+1,0]-outputp[kk,tt,0])**2+(outputp[kk,tt+1,1]-outputp[kk,tt,1])**2)
                peak_vel[sub_num-1,e,kk] = np.max(vel[e,sub_num-1,:,kk])
        
            e = e+1
            # # plotting
            # # plot trajectory
            # plt.figure()
            # for i in range(num_inputs):
            #     # plt.figure()
            #     plt.scatter(targets[i,0],targets[i,1],color='r')
            #     plt.plot(outputp[i,:,0],outputp[i,:,1],'black')
            #     plt.plot(outputs[i,:,0],outputs[i,:,1],'blue')
            #     # plt.savefig("models/rppc_lppc/hypothesis_plots/v6_all_training_w_dir_error/RH/epoch_"+str(epoch)+".png")
# #
# =============================================================================
#                               HYPOTHESIS PLOTS
# =============================================================================

# re_mean = np.zeros((num_criteria,num_epochs,2))
# de_mean = np.zeros((num_criteria,num_epochs,2))
# for j in range(num_criteria):
#     for i in range(num_epochs):
#         a = np.zeros((5,8))
#         a = re[j,:,i,:]
#         b = np.mean(a,axis=1)
#         re_mean[j,i,0] = np.mean(b)
#         re_mean[j,i,1] = np.std(b)
#         c = np.zeros((5,8))
#         c = de[j,:,i,:]
#         d = np.mean(c,axis=1)
#         de_mean[j,i,0] = np.mean(d)
#         de_mean[j,i,1] = np.std(d)

# plt.figure()
# plt.plot(de_mean[0,:,0],color='red',label='LPPC_SHAM')
# plt.plot(de_mean[1,:,0],color='blue',label='LPPC_STIM')
# plt.plot(de_mean[2,:,0],color='green',label='RPPC_SHAM')
# plt.plot(de_mean[3,:,0],color='orange',label='RPPC_STIM')
# # # plt.plot(de_mean[4,:,0],color='cyan',label='v5_0_05')
# # # plt.plot(de_mean[5,:,0],color='yellow',label='v6_0_075')
# # # # plt.plot(de_mean[6,:,0],color='orange',label='v7_0_025')
# # # # plt.plot(de_mean[7,:,0],color='cyan',label='v8_0_05')
# # # # plt.plot(de_mean[8,:,0],color='yellow',label='v9_0_075')
# # # plt.plot(de_mean[9,:,0],color='black',label='normal')
# # # plt.ylim([0,15])
# plt.legend()
# plt.savefig("models/rppc_lppc/lppc/PM/stimulated/new_new/de_compare_final_epochs_all.png")
# plt.figure()
# plt.plot(de_mean[0,:,0],color='red',label='LH')
# plt.plot(de_mean[1,:,0],color='blue',label='RH')
# plt.legend()
# plt.savefig("models/rppc_lppc/hypothesis_plots/v6_all_training_w_dir_error/de_compare.png")

# re_mean = np.zeros((num_epochs,2))
# de_mean = np.zeros((num_epochs,2))
# for i in range(num_epochs):
#     a = np.zeros((5,8))
#     a = re[0,:,i,:]
#     b = np.mean(a,axis=1)
#     re_mean[i,0] = np.mean(b)
#     re_mean[i,1] = np.std(b)
#     c = np.zeros((5,8))
#     c = de[0,:,i,:]
#     d = np.mean(c,axis=1)
#     de_mean[i,0] = np.mean(d)
#     de_mean[i,1] = np.std(d)

# plt.figure()
# plt.plot(de_mean[:,0],color='blue',label='DE')
# plt.savefig("models/rppc_lppc/hypothesis_plots/v6_all_training_w_dir_error/RH/de_mean.png")
# plt.figure()
# plt.plot(re_mean[:,0],color='red',label='RE')
# plt.savefig("models/rppc_lppc/hypothesis_plots/v6_all_training_w_dir_error/RH/re_mean.png")
# # plt.legend()

# #
# # =============================================================================
# #                             MUTHA ET AL PLOTS
# # =============================================================================
# #
# # 1. plot trajectory across epochs for all subjects 
# for i in range(num_subjects):
#     for j in range(num_epochs):
#         plt.figure()
#         for k in range(num_targets):
#             plt.scatter(targets[k,0],targets[k,1],color='r')
#             plt.plot(outputp_all[i,j,k,:,0],outputp_all[i,j,k,:,1],'black')
#             # plt.plot(outputs[i,:,0],outputs[i,:,1],'blue')
#             plt.savefig("figures/mutha_et_al/normal/rh/trajectory/"+str(i+1)+"_epoch_"+str(j+1)+".png")
# # mean trajectpry across subjects for each epoch
# mean_traj = np.zeros((num_epochs,num_targets,time_steps,2))
# for i in range(num_epochs):
#     for j in range(num_targets):
#         for k in range(time_steps):
#             mean_traj[i,j,k,0] = np.sum(outputp_all[:,i,j,k,0])/num_subjects
#             mean_traj[i,j,k,1] = np.sum(outputp_all[:,i,j,k,1])/num_subjects
# for j in range(num_epochs):
#     plt.figure()
#     for k in range(num_targets):
#         plt.scatter(targets[k,0],targets[k,1],color='r')
#         plt.plot(mean_traj[j,k,:,0],mean_traj[j,k,:,1],'black')
#         # plt.plot(outputs[i,:,0],outputs[i,:,1],'blue')
#         plt.savefig("figures/mutha_et_al/normal/rh/trajectory/avg_across_subjects/epoch_"+str(j+1)+".png")

# # 2. a. plot velocity for 200th epoch for all subjects
# mean_vel = np.zeros((num_subjects,time_steps-1))
# for i in range(num_subjects):
#     for j in range(time_steps-1):
#         mean_vel[i,j] = np.sum(vel[199,i,j,:])/8
# x = np.arange(time_steps-1)
# for i in range(num_subjects):
#     plt.figure()
#     plt.plot(x,mean_vel[i,:])
#     plt.xlabel('Timesteps')
#     plt.ylabel('Velocity across targets')
#     plt.savefig('figures/mutha_et_al/normal/rh/velocity/'+str(i+1)+'.png')

# # 2. b. plot velocity for 200th epoch across all subjects 
# mean_mean_vel = np.zeros((time_steps-1))
# for i in range(time_steps-1):
#     mean_mean_vel[i] = np.sum(mean_vel[:,i])/20
# x = np.arange(time_steps-1)
# plt.figure()
# plt.plot(x,mean_mean_vel[:])
# plt.xlabel('Timesteps')
# plt.ylabel('Velocity across targets')
# plt.savefig('figures/mutha_et_al/normal/rh/velocity/avg_across_subjects.png')

# # 2. c. plot velocity for 200th epoch across all subjects with SD 
# mean_mean_vel = np.zeros((time_steps-1,2))
# for i in range(time_steps-1):
#     mean_mean_vel[i,0] = np.sum(mean_vel[:,i])/20
#     mean_mean_vel[i,1] = np.std(mean_vel[:,i])
# x = np.arange(time_steps-1)
# plt.figure()
# plt.errorbar(x,mean_mean_vel[:,0],yerr=mean_mean_vel[:,1],marker='o',color='blue',capsize=5)
# plt.xlabel('Timesteps')
# plt.ylabel('Velocity across targets')
# plt.savefig('figures/mutha_et_al/normal/rh/velocity/avg_across_subjects_with_sd.png')

# # 3. Generalisation/Transfer plots 
# de_mean = np.zeros((2,20))
# de_mean[0,:] = np.average(de[0,:,0,:],axis=1)
# de_mean[1,:] = np.average(de[1,:,0,:],axis=1)
# de_mean_mean = np.zeros((2,4,2))
# for i in range(2):
#     de_mean_mean[i,0,0] = np.sum(de_mean[i,:5])/5
#     de_mean_mean[i,1,0] = np.sum(de_mean[i,5:10])/5
#     de_mean_mean[i,2,0] = np.sum(de_mean[i,10:15])/5
#     de_mean_mean[i,3,0] = np.sum(de_mean[i,15:])/5#de_mean_mean[i,2,0]#
#     de_mean_mean[i,0,1] = np.std(de_mean[i,:5])
#     de_mean_mean[i,1,1] = np.std(de_mean[i,5:10])
#     de_mean_mean[i,2,1] = np.std(de_mean[i,10:15])
#     de_mean_mean[i,3,1] = de_mean_mean[i,2,1]#np.std(de_mean[i,15:])#

# # set width of bars
# barWidth = 0.125
 
# # set heights of bars
# bars1 = (de_mean_mean[:,3,0]*0)+18.3#[12, 30, 1, 8, 22]
# bars2 = de_mean_mean[:,2,0]
# bars3 = de_mean_mean[:,1,0]
# bars4 = de_mean_mean[:,0,0]
 
# # Set position of bar on X axis
# r1 = np.arange(len(bars1))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3] 
# # Make the plot
# plt.bar(r1, bars1, yerr=[np.zeros(len(de_mean_mean[:,3,1])),de_mean_mean[:,3,1]], color='red', width=barWidth, edgecolor='white', label='40')
# plt.bar(r2, bars2, yerr=[np.zeros(len(de_mean_mean[:,2,1])),de_mean_mean[:,2,1]],color='blue', width=barWidth, edgecolor='white', label='80')
# plt.bar(r3, bars3, yerr=[np.zeros(len(de_mean_mean[:,1,1])),de_mean_mean[:,1,1]],color='green', width=barWidth, edgecolor='white', label='160')
# plt.bar(r4, bars4, yerr=[np.zeros(len(de_mean_mean[:,0,1])),de_mean_mean[:,0,1]],color='yellow', width=barWidth, edgecolor='white', label='320')
 
# # Add xticks on the middle of the group bars
# plt.xlabel('Group', fontweight='bold')
# plt.xticks([r + 0.2 for r in range(len(bars1))], ['Naive Learning','Generalization (after Right arm learning)'])
# plt.ylabel('Directional Error (deg)') 
# # Create legend & Show graphic
# plt.legend(title='Transfer group')
# plt.savefig("figures/mutha_et_al/generalisation/Left_Arm.png")

# de_mean = np.average(de,axis=3)
# plt.figure()
# plt.plot(de_mean[0,0,:],color='blue')
# plt.plot(de_mean[1,0,:],color='red')

# 4. Stimulation plots - sham vs stim
re_mean = np.zeros((num_criteria,num_epochs,2))
de_mean = np.zeros((num_criteria,num_epochs,2))
for j in range(num_criteria):
    for i in range(num_epochs):
        a = np.zeros((5,8))
        a = re[j,:,i,:]
        b = np.mean(a,axis=1)
        re_mean[j,i,0] = np.mean(b)
        re_mean[j,i,1] = np.std(b)
        c = np.zeros((5,8))
        c = de[j,:,i,:]
        d = np.mean(c,axis=1)
        de_mean[j,i,0] = np.mean(d)
        de_mean[j,i,1] = np.std(d)

plt.figure()
plt.plot(de_mean[0,:,0],color='red',label='LPPC_SHAM')
plt.plot(de_mean[1,:,0],color='blue',label='LPPC_STIM')
# plt.plot(de_mean[2,:,0],color='green',label='RPPC_SHAM')
# plt.plot(de_mean[3,:,0],color='orange',label='RPPC_STIM')
plt.legend()
plt.savefig("figures/mutha_et_al/stimulated/Right_Arm_lppc_across_trials.png")

plt.figure()
plt.plot(de_mean[2,:,0],color='orange',label='RPPC_SHAM')
plt.plot(de_mean[3,:,0],color='green',label='RPPC_STIM')
plt.legend()
plt.savefig("figures/mutha_et_al/stimulated/Left_Arm_rppc_across_trials.png")

# mean across targets; first 8 trials, next 8 trials; sham vs stim
de_mean = np.zeros((4,2))
de_std = np.zeros((4,2))
for i in range(4):
    # across targts; across trials; across subjects
    a = np.average(de[i,:,:8,:],axis=2)
    b = np.average(de[i,:,142:,:],axis=2)
    c = np.average(a,axis=1)
    d = np.average(b,axis=1)
    e = np.average(c)
    f = np.average(d)
    de_mean[i,0] = e#np.average(np.average((np.average(de[i,:,:8,:],axis=2)),axis=1))
    de_mean[i,1] = f#np.average(np.average((np.average(de[i,:,142:,:],axis=2)),axis=1))
    de_std[i,0] = np.std(np.std((np.std(de[i,:,:8,:],axis=2)),axis=1))
    de_std[i,1] = np.std(np.std((np.std(de[i,:,142:,:],axis=2)),axis=1))

# set width of bars
barWidth = 0.25
 
# set heights of bars
bars2 = de_mean[2,:2]
bars1 = de_mean[3,:2]
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

# Make the plot
plt.bar(r1, bars1, yerr=[np.zeros(len(de_std[3,:2])),de_std[3,:2]], color='green', width=barWidth, edgecolor='white', label='Stim')
plt.bar(r2, bars2, yerr=[np.zeros(len(de_std[2,:2])),de_std[2,:2]],color='orange', width=barWidth, edgecolor='white', label='Sham')
# Add xticks on the middle of the group bars
plt.ylim([7,11])
plt.xlabel('Group', fontweight='bold')
plt.xticks([r + 0.13 for r in range(len(bars1))], ['First 8 learning trials','Last 8 learning trials'])
plt.ylabel('Directional Error (deg)') 
# Create legend & Show graphic
plt.legend(title='Stimulation group')
plt.savefig("figures/mutha_et_al/stimulated/Left_Arm_rppc.png")
        
# mean_de = np.zeros((2,20,320))
# mean_mean_de = np.zeros((2,4,320))
# for i in range(2):
#     for j in range(20):
#         for k in range(320):
#             mean_de[i,j,k] = np.sum(de[i,j,k,:])/8
# for i in range(2):
#     for j in range(320):
#         mean_mean_de[i,0,j] = np.sum(mean_de[i,:5,j])/5
#         mean_mean_de[i,1,j] = np.sum(mean_de[i,5:10,j])/5
#         mean_mean_de[i,2,j] = np.sum(mean_de[i,10:15,j])/5
#         mean_mean_de[i,3,j] = np.sum(mean_de[i,15:,j])/5

# x=np.arange(320)
# plt.figure()
# plt.plot(x,mean_mean_de[0,0,:],color='blue')
# plt.plot(x[:160],mean_mean_de[0,1,:160],color='green')
# plt.plot(x[:80],mean_mean_de[0,2,:80],color='red')
# plt.plot(x[:40],mean_mean_de[0,3,:40],color='yellow')   

# x=np.arange(320)
# plt.figure()
# plt.plot(x,mean_mean_de[1,0,:],color='blue')
# plt.plot(x[:160],mean_mean_de[1,1,:160],color='green')
# plt.plot(x[:80],mean_mean_de[1,2,:80],color='red')
# plt.plot(x[:40],mean_mean_de[1,3,:40],color='yellow')        

# mean_vel = np.zeros((num_criteria,time_steps-1))
# for k in range(3):
#     for i in range(time_steps-1):
#         a = np.zeros((num_subjects))
#         for j in range(num_subjects):
#             a[j] = np.sum(vel[k,j,i,:])/8
#         mean_vel[k,i] = np.sum(a)/20
# x = np.arange(time_steps-1)
# plt.figure()
# plt.plot(x,mean_vel[0,:],color='red',label='normal')
# plt.plot(x,mean_vel[1,:],color='blue',label='lppc_lesion')
# # plt.plot(x,mean_vel[2,:],color='green',label='rppc_lesion')
# plt.legend()
# plt.xlabel('Timesteps')
# plt.ylabel('Velocity')
# plt.title('Velocity curve')

# x = np.arange(num_epochs)
# re_mean = np.zeros((2,num_epochs))
# re_std = np.zeros((2,num_epochs))
# for i in range(2):
#     for j in range(num_epochs):
#         re_m = np.zeros((3,1))
#         for k in range(3):
#             re_m[k,0] = np.sum(re[i,k,j,:])/8
#         re_mean[i,j] = np.sum(re_m[:,0])/3
#         re_std[i,j] = np.std(re_m[:,0])

# plt.figure()
# # plt.ylim([0,1])
# plt.plot(x,re_mean[0,:],color='red',marker='o',label='LR')
# plt.plot(x,re_mean[1,:],color='blue',marker='o',label='RL')
# # plt.errorbar(x,re_mean[0,:],color='red',label='LR')
# # plt.errorbar(x,re_mean[0,:],yerr=re_std[0,:],marker='o',color='red',label='RL',capsize=5)
# # plt.errorbar(x,re_mean[1,:],yerr=re_std[1,:],marker='o',color='blue',label='LR',capsize=5)
# plt.legend()
# plt.xlabel('Number of Epochs')
# plt.ylabel('Reaching Error')
# plt.title('Final position error - Exposure')
# plt.savefig("models/rppc_lppc/sainburg/Left/12_epochs/RE_exp.png")

# x = np.arange(num_epochs)
# de_mean = np.zeros((2,num_epochs))
# de_std = np.zeros((2,num_epochs))
# for i in range(2):
#     for j in range(num_epochs):
#         de_m = np.zeros((3,1))
#         for k in range(3):
#             de_m[k,0] = np.sum(de[i,k,j,:])/8
#         de_mean[i,j] = np.sum(de_m[:,0])/3
#         de_std[i,j] = np.std(de_m[:,0])
# # de_mean[:,0] = 15
# plt.figure()
# # plt.ylim([0,15])
# plt.plot(x,de_mean[0,:],color='red',marker='o',label='LR')
# plt.plot(x,de_mean[1,:],color='blue',marker='o',label='RL')
# # plt.errorbar(x,de_mean[0,:num_epochs],yerr=de_std[0,:num_epochs],marker='o',color='red',label='RL',capsize=5)
# # plt.errorbar(x,de_mean[1,:num_epochs],yerr=de_std[1,:num_epochs],marker='o',color='blue',label='LR',capsize=5)
# plt.legend()
# plt.xlabel('Number of Epochs')
# plt.ylabel('Directional Error')
# plt.title('Directional Error at Vmax - Exposure')
# plt.savefig("models/rppc_lppc/sainburg/Left/12_epochs/DE_at_Vmax_exp.png")

# x = np.arange(151)
# de_l_mean = np.zeros((2,151))
# de_l_mean[:,0] = 15
# for i in range(150):
#     de_norm = np.zeros((5,1))
#     de_stim = np.zeros((5,1))
#     for j in range(5):
#         de_norm[j,0] = np.sum(de_l[0,j,i,:])/8
#         de_stim[j,0] = np.sum(de_l[1,j,i,:])/8
#     de_l_mean[0,i+1] = np.sum(de_norm)/5
#     de_l_mean[1,i+1] = np.sum(de_stim)/5

# plt.figure()
# plt.plot(x,de_l_mean[0,:],color='red',label='normal')
# plt.plot(x,de_l_mean[1,:],color='blue',label='stim')
# plt.legend()

# x = np.arange(8)

# plt.figure()
# plt.plot(x,de[0,99,:],color='red',marker='o',label='45 degree')
# plt.plot(x,de[1,99,:],color='blue',marker='o',label='135 degree')
# plt.plot(x,de[2,99,:],color='green',marker='o',label='225 degree')
# plt.plot(x,de[3,99,:],color='yellow',marker='o',label='315 degree')
# plt.legend()
# labels=np.array((["-90","-45","-22.5","0","22.5","45","90","180"]))
# plt.xticks(x,labels)
# plt.xlabel('Target Direction')
# plt.ylabel('Reaching Error')
# plt.savefig("models/rppc_lppc/krakauer_new/figures/direction_adaptation/dir_gen.png")

# x = np.arange(4)
# re_mean = np.zeros((num_criteria,num_epochs,4))
# for i in range(num_criteria):
#     for j in range(num_epochs):
#         re_mean[i,j,0] = (re[i,j,0]+re[i,j,4])/2
#         re_mean[i,j,1] = (re[i,j,1]+re[i,j,5])/2
#         re_mean[i,j,2] = (re[i,j,2]+re[i,j,6])/2
#         re_mean[i,j,3] = (re[i,j,3]+re[i,j,7])/2
# re_final_mean = np.zeros((2,4))
# re_final_mean[0,0] = np.sum(re_mean[:,59,0])/4
# re_final_mean[0,1] = np.sum(re_mean[:,59,1])/4
# re_final_mean[0,2] = np.sum(re_mean[:,59,2])/4
# re_final_mean[0,3] = np.sum(re_mean[:,59,3])/4
# re_final_mean[1,0] = np.std(re_mean[:,59,0])#/4
# re_final_mean[1,1] = np.std(re_mean[:,59,1])#/4
# re_final_mean[1,2] = np.std(re_mean[:,59,2])#/4
# re_final_mean[1,3] = np.std(re_mean[:,59,3])#/4

# plt.figure()
# plt.plot(x,re_mean[0,59,:],color='red',marker='o',label='first dist 45')
# plt.plot(x,re_mean[1,59,:],color='blue',marker='o',label='second dist 45')
# plt.plot(x,re_mean[2,59,:],color='green',marker='o',label='first dist 135')
# plt.plot(x,re_mean[3,59,:],color='yellow',marker='o',label='second dist 135')
# plt.legend()
# labels=np.array((["2","4","6","8"]))
# plt.xticks(x,labels)
# plt.xlabel('Target Extent')
# plt.ylabel('Reaching Error')
# plt.savefig("models/rppc_lppc/krakauer_new/figures/distance_generalisation_of_gain/dist_gen.png")

# plt.figure()
# plt.errorbar(x,re_final_mean[0,:],yerr=re_final_mean[1,:],color='blue',marker='o',capsize=5)
# labels=np.array((["2","4","6","8"]))
# plt.xticks(x,labels)
# plt.xlabel('Target Extent')
# plt.ylabel('Reaching Error')
# plt.savefig("models/rppc_lppc/krakauer_new/figures/distance_generalisation_of_gain/dist_gen_mean.png")

# x = np.arange(5)
# re_mean = np.zeros((num_criteria,num_epochs,5))
# for i in range(num_criteria):
#     for j in range(num_epochs):
#         re_mean[i,j,0] = re[i,j,0] # 0 degree
#         re_mean[i,j,1] = (re[i,j,1]+re[i,j,2])/2 # 22.5 degree
#         re_mean[i,j,2] = (re[i,j,3]+re[i,j,4])/2 # 45 degree
#         re_mean[i,j,3] = (re[i,j,5]+re[i,j,6])/2 # 90 degree
#         re_mean[i,j,4] = re[i,j,7] # 180 degree
# re_final_mean = np.zeros((2,5))
# re_final_mean[0,0] = np.sum(re_mean[:,59,0])/4
# re_final_mean[0,1] = np.sum(re_mean[:,59,1])/4
# re_final_mean[0,2] = np.sum(re_mean[:,59,2])/4
# re_final_mean[0,3] = np.sum(re_mean[:,59,3])/4
# re_final_mean[0,4] = np.sum(re_mean[:,59,4])/4
# re_final_mean[1,0] = np.std(re_mean[:,59,0])#/4
# re_final_mean[1,1] = np.std(re_mean[:,59,1])#/4
# re_final_mean[1,2] = np.std(re_mean[:,59,2])#/4
# re_final_mean[1,3] = np.std(re_mean[:,59,3])#/4
# re_final_mean[1,4] = np.std(re_mean[:,59,4])#/4

# plt.figure()
# plt.plot(x,re_mean[0,59,:],color='red',marker='o',label='45 degree')
# plt.plot(x,re_mean[1,59,:],color='blue',marker='o',label='135 degree')
# plt.plot(x,re_mean[2,59,:],color='green',marker='o',label='225 degree')
# plt.plot(x,re_mean[3,59,:],color='yellow',marker='o',label='315 degree')
# plt.legend()
# labels=np.array((["0","22.5","45","90","180"]))
# plt.xticks(x,labels)
# plt.xlabel('Target Direction')
# plt.ylabel('Reaching Error')
# plt.savefig("models/rppc_lppc/krakauer_new/figures/direction_generalisation_of_gain/dir_gen.png")

# plt.figure()
# plt.errorbar(x,re_final_mean[0,:],yerr=re_final_mean[1,:],color='blue',marker='o',capsize=5)
# labels=np.array((["0","22.5","45","90","180"]))
# plt.xticks(x,labels)
# plt.xlabel('Target Direction')
# plt.ylabel('Reaching Error')
# plt.savefig("models/rppc_lppc/krakauer_new/figures/direction_generalisation_of_gain/dir_gen_mean.png")

# x = np.arange(num_epochs)
# de_targ_mean = np.zeros((num_criteria,num_epochs))
# de_mean = np.zeros((num_epochs))
# de_std = np.zeros((num_epochs))
# for i in range(num_epochs):
#     for j in range(num_criteria):
#         de_targ_mean[j,i] = np.sum(de[j,i,:])/num_targets
#     de_mean[i] = np.sum(de_targ_mean[:,i])/num_criteria
#     de_std[i] = np.std(de_targ_mean[:,i])

# # for i in range(5):
# #     plt.figure()
# #     plt.plot(x,de_targ_mean[i,:],color='blue')
# #     plt.xlabel("Number of Epochs")
# #     plt.ylabel("Directional Error (deg)")
# #     # plt.savefig("models/rppc_lppc/figures/extent_plus_rot/new/rh/"+str(i+1)+"/de.png")

# plt.figure()
# plt.plot(x,de_mean,color='blue')
# plt.xlabel("Number of Epochs")
# plt.ylabel("Directional Error (deg)")
# # plt.savefig("models/rppc_lppc/figures/extent_plus_rot/new/rh/de.png")

# x = np.arange(num_epochs)
# re_targ_mean = np.zeros((num_criteria,num_epochs))
# re_mean = np.zeros((num_epochs))
# re_std = np.zeros((num_epochs))
# for i in range(num_epochs):
#     for j in range(num_criteria):
#         re_targ_mean[j,i] = np.sum(re[j,i,:])/num_targets
#     re_mean[i] = np.sum(re_targ_mean[:,i])/num_criteria
#     re_std[i] = np.std(re_targ_mean[:,i])
    
# pv = np.zeros((num_epochs,1))
# for i in range(num_epochs):
#     pv[i,0] = np.sum(peak_vel[0,i,:])/2
    
# # for i in range(5):
# #     plt.figure()
# #     plt.plot(x,re_targ_mean[i,:],color='blue')
# #     plt.xlabel("Number of Epochs")
# #     plt.ylabel("Reaching Error")
# #     # plt.savefig("models/rppc_lppc/figures/extent_plus_rot/new/rh/"+str(i+1)+"/re.png")

# plt.figure()
# plt.plot(x,re_targ_mean[0,:],color='blue')
# plt.xlabel("Number of Epochs")
# plt.ylabel("Reaching Error")
# plt.savefig("models/rppc_lppc/krakauer_new/figures/direction_generalisation_of_gain/315_degree/lh_re.png")

# plt.figure()
# plt.plot(x,pv[:,0],color='blue')
# plt.xlabel("Number of Epochs")
# plt.ylabel("Peak Velocity")
# plt.savefig("models/rppc_lppc/krakauer_new/figures/direction_generalisation_of_gain/45_degree/lh_adapted_pv_lppc_100.png")
# # =============================================================================
# #                                   NORMAL
# # =============================================================================
# de_l_mean = np.zeros((4,5,321))
# de_l_norm = np.zeros((4,321,2))
# de_l_mean[:,:,0] = 15
# for i in range(4):
#     for j in range(5):
#         for e in range(dirs[i]):
#             de_l_mean[i,j,e+1] = np.sum(de_l[i,j,e,:])/8
#     x = np.arange(dirs[i]+1)
#     plt.figure()
#     plt.plot(x,de_l_mean[i,0,:(dirs[i]+1)],color='red',label='sub_1')
#     plt.plot(x,de_l_mean[i,1,:(dirs[i]+1)],color='blue',label='sub_2')
#     plt.plot(x,de_l_mean[i,2,:(dirs[i]+1)],color='green',label='sub_3')
#     plt.plot(x,de_l_mean[i,3,:(dirs[i]+1)],color='yellow',label='sub_4')
#     plt.plot(x,de_l_mean[i,4,:(dirs[i]+1)],color='black',label='sub_5')
#     plt.legend()
#     plt.xlabel("Number of Epochs")
#     plt.ylabel("Directional Error")
#     plt.title("Directional Error vs Epochs - Left Hand")
#     # plt.savefig("models/rppc_lppc/DE_rot_train_v2_RH.png")
# de_l_mean[3,2,:] = 0
# # de_l_mean[3,4,:] = 0
# for i in range(4):
#     for e in range(321):
#         if i == 3:
#             de_l_norm[i,e,0] = np.sum(de_l_mean[i,:,e])/4
#             de_l_norm[i,e,1] = np.std(de_l_mean[i,:,e])
#         else:
#             de_l_norm[i,e,0] = np.sum(de_l_mean[i,:,e])/5
#             de_l_norm[i,e,1] = np.std(de_l_mean[i,:,e])

# plt.figure()
# plt.plot(de_l_norm[0,:,0],color='blue',label='320')
# plt.plot(de_l_norm[1,:160,0],color='green',label='160')
# plt.plot(de_l_norm[2,:80,0],color='red',label='80')
# plt.plot(de_l_norm[3,:40,0],color='yellow',label='40')
# plt.legend()
# ## plt.xticks(x,labels)
# plt.xlabel("Number of Epochs")
# plt.ylabel("Directional Error")
# plt.title("Directional Error vs Epochs - Right Hand")
# # plt.savefig("models/rppc_lppc/DE_rot_train_v2_RH.png")

# bar = np.array(([de_l_norm[3,40,0],de_l_norm[2,80,0],de_l_norm[1,160,0],de_l_norm[0,320,0]]))
# bar_err = np.array(([de_l_norm[3,40,1],de_l_norm[2,80,1],de_l_norm[1,160,1],de_l_norm[0,320,1]]))
# x = np.arange(4)
# labels=np.array((['40','80','160','320']))
# plt.figure()
# plt.bar(x,bar,yerr=bar_err,color='blue',capsize=10)
# plt.xticks(x,labels)
# plt.xlabel("Number of Epochs")
# plt.ylabel("Directional Error")
# plt.title("Directional Error vs Epochs - Right Hand")

# =============================================================================
#                                  OTHER ARM
# =============================================================================
#de_mean_r = np.zeros((4,5,321))
#de_norm_r = np.zeros((4,321))
#de_mean_r[:,:,0] = 15
#k = 0
#for i in range(4):
#    for j in range(5):
#        for e in range(epochs[i]):
#            if k == 17:
#                break
#            else:
#                de_mean_r[i,j,e+1] = -1*np.sum(de[k,e,3,:])/8
#        k = k+1
#for t in range(4):
#    for e in range(321):
#        if t == 3:
#            de_norm_r[t,e] = np.sum(de_mean_r[t,:,e])/4
#        else:
#            de_norm_r[t,e] = np.sum(de_mean_r[t,:,e])/5

#plt.figure()
#plt.plot(de_norm[0,:],color='blue',label='320')
#plt.plot(de_norm[1,:160],color='green',label='160')
#plt.plot(de_norm[2,:80],color='red',label='80')
#plt.plot(de_norm[3,:40],color='yellow',label='40')
#plt.plot(de_norm_r[0,:],color='blue',label='320',linestyle='--')
#plt.plot(de_norm_r[1,:160],color='green',label='160',linestyle='--')
#plt.plot(de_norm_r[2,:80],color='red',label='80',linestyle='--')
#plt.plot(de_norm_r[3,:40],color='yellow',label='40',linestyle='--')
#plt.legend()
#plt.xlabel("Number of Epochs")
#plt.ylabel("Directional Error")
#plt.title("Directional Error vs Epochs - Generalization")
#plt.savefig("models/rppc_lppc/generalization_RH.png")
#
#bar = np.zeros((5))
#bar[0] = 15
#bar[4] = de_norm_r[0,319]
#bar[3] = de_norm_r[1,159]
#bar[2] = de_norm_r[2,79]
#bar[1] = de_norm_r[3,39]
#x = np.arange(5)
#labels = np.array(['Naive','40','80','160','320'])
#plt.figure()
#plt.bar(x,bar)#,width=0.4)
#plt.xticks(x,labels)
#plt.xlabel('Number of Epochs')
#plt.ylabel('Directional Error')
#plt.title('Generalization - Right Arm after Left Arm learning')
#plt.savefig('models/rppc_lppc/bar_rh.png')
#
#bar = np.zeros((5))
#bar[0] = 15
#bar[4] = 15
#bar[3] = 15
#bar[2] = 15
#bar[1] = 15
#x = np.arange(5)
#labels = np.array(['Naive','40','80','160','320'])
#plt.figure()
#plt.bar(x,bar)#,width=0.4)
#plt.xticks(x,labels)
#plt.xlabel('Number of Epochs')
#plt.ylabel('Directional Error')
#plt.title('Generalization - Left Arm after Right Arm learning')
#plt.savefig('models/rppc_lppc/bar_lh.png')

# =============================================================================
#                                 STIMULATION
# =============================================================================
#de_mean_s = np.zeros((5,151))
#de_norm_s = np.zeros((1,151))
#de_mean_s[:,0] = 15
#for j in range(5):
#    for e in range(150):
#        de_mean_s[j,e+1] = -1*np.sum(de[j,e,3,:])/8
#
#for e in range(151):
#        de_norm_s[0,e] = np.sum(de_mean_s[:,e])/5
#
#plt.figure()
#plt.plot(de_norm[0,:150],color='blue',label='SHAM')
#plt.plot(de_norm_s[0,:],color='red',label='STIM')
#plt.legend()
#plt.xlabel("Number of Epochs")
#plt.ylabel("Directional Error")
#plt.title("Directional Error vs Epochs")
#plt.savefig("models/rppc_lppc/DE_stim_v2_v2_RH.png")

#plt.figure()
#plt.plot(de_norm[0,:150],color='blue',label='SHAM')
#plt.plot(de_norm[0,:150],color='red',label='STIM')
#plt.legend()
#plt.xlabel("Number of Epochs")
#plt.ylabel("Directional Error")
#plt.title("Directional Error vs Epochs")
#plt.savefig("models/rppc_lppc/DE_stim_LH.png")
