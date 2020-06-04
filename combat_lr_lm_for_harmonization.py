#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:27:25 2019

@author: fenqiang
"""
import glob
import numpy as np
import scipy.io as sio 
import matplotlib.pyplot as plt
from neuroCombat import neuroCombat
import pandas as pd


#################################################################################################
################################ combat for harmonization ###########################
data = np.concatenate((realA, realB), axis=0)
batch = np.concatenate((np.zeros(len(realA)), np.zeros(len(realB))+1), 0) + 1
age =  np.concatenate((realA_age, realB_age), 0)
covars = pd.DataFrame(data={'batch': batch, 'age': age})
continuous_cols = ['age']
batch_col = 'batch'
data_combat = neuroCombat(data=data,
                          covars=covars,
                          batch_col=batch_col,
                          continuous_cols=continuous_cols)

fakeB = data_combat[0:len(realA),:]
fakeA = data_combat[len(realA):,:]

for i in range(len(fakeA)):
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/generated/combat/' + realB_files[i].split('/')[-1].split('.')[0] + '_combat.txt', fakeA[i,:])
for i in range(len(fakeB)):
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/generated/combat/' + realA_files[i].split('/')[-1].split('.')[0] + '_combat.txt', fakeB[i,:])
############################################################################################################
    

#################################################################################################
################################ histogram matching for harmonization ###########################
realA = np.concatenate((realA_train, realA_test), 0)
realA_files = realA_train_files + realA_test_files
realA_age = np.concatenate((realA_train_age, realA_test_age), 0)
realB = np.concatenate((realB_train, realB_test), 0)
realB_files = realB_train_files + realB_test_files
realB_age = np.concatenate((realB_train_age, realB_test_age), 0)
realA_mean =realA.mean(0)
realA_std =realA.std(0)
realB_mean = realB.mean(0)
realB_std = realB.std(0)
fakeB = (realA -realA_mean) / realA_std * realB_std + realB_mean
fakeA = (realB - realB_mean) / realB_std * realA_std + realA_mean


for i in range(len(fakeA)):
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/generated/hs/' + realB_files[i].split('/')[-1].split('.')[0] + '_hs.txt', fakeA[i,:])
for i in range(len(fakeB)):
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/generated/hs/' + realA_files[i].split('/')[-1].split('.')[0] + '_hs.txt', fakeB[i,:])
###############################################################################################################



############################################################################################
################# linear regression for harmonization #####################################
realA = np.concatenate((realA_train, realA_test), 0)
realA_files = realA_train_files + realA_test_files
realB = np.concatenate((realB_train, realB_test), 0)
realB_files = realB_train_files + realB_test_files
realA_mean = realA.mean(0)
realB_mean = realB.mean(0)
fakeB = (realB_mean - realA_mean) + realA
fakeA = (realA_mean - realB_mean) + realB

for i in range(len(fakeA)):
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/lr/' + realB_files[i].split('/')[-1].split('.')[0] + '_lr.txt', fakeA[i,:])
for i in range(len(fakeB)):
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/lr/' + realA_files[i].split('/')[-1].split('.')[0] + '_lr.txt', fakeB[i,:])
###############################################################################################
###################################################################################################
    
    
############################################################################################
################# linear mapping for harmonization #####################################
realA = np.concatenate((realA_train, realA_test), 0)
realA_files = realA_train_files + realA_test_files
realB = np.concatenate((realB_train, realB_test), 0)
realB_files = realB_train_files + realB_test_files
realA_mean = realA.mean(0)
realB_mean = realB.mean(0)
fakeB = (realB_mean / realA_mean) * realA
fakeA = (realA_mean / realB_mean) * realB

for i in range(len(fakeA)):
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/lm/' + realB_files[i].split('/')[-1].split('.')[0] + '_lm.txt', fakeA[i,:])
for i in range(len(fakeB)):
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/lm/' + realA_files[i].split('/')[-1].split('.')[0] + '_lm.txt', fakeB[i,:])
###############################################################################################
###################################################################################################


# read atlas template to get 36 roi feature
atlas = sio.loadmat('/media/fenqiang/DATA/unc/Data/harmonization/atlas_par40962_36.mat')
atlas = atlas['par40962_36'][:,0]
roi_label = np.unique(atlas)
n_roi = len(roi_label)
roi_list = []
for i in range(n_roi):
    roi_list.append(np.argwhere(atlas == roi_label[i]))


n_roi = 36
realA_roi = np.zeros((len(realA), n_roi))
realB_roi = np.zeros((len(realB), n_roi))
fakeA_roi = np.zeros((len(fakeA), n_roi))
fakeB_roi = np.zeros((len(fakeB), n_roi))

for i in range(len(realA_roi)):
    data = realA[i]
    for j in range(n_roi):
        realA_roi[i,j] = np.mean(data[roi_list[j]])
    
for i in range(len(realB_roi)):
    data = realB[i]
    for j in range(n_roi):
        realB_roi[i,j] = np.mean(data[roi_list[j]])
    
for i in range(len(fakeA_roi)):
    data = fakeA[i]
    for j in range(n_roi):
        fakeA_roi[i,j] = np.mean(data[roi_list[j]])
    
for i in range(len(fakeB_roi)):
    data = fakeB[i]
    for j in range(n_roi):
        fakeB_roi[i,j] = np.mean(data[roi_list[j]])
    
     
   
realA_roi_m = realA_roi.mean(0)
realB_roi_m = realB_roi.mean(0)
fakeA_roi_m = fakeA_roi.mean(0)
fakeB_roi_m = fakeB_roi.mean(0)
