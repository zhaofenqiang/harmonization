#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 20:15:12 2019

@author: fenqiang
"""


import torch
import torchvision
import torch.nn as nn
import scipy.io as sio 
import numpy as np
import glob
import random
import os

from utils import ImagePool
from model import Unet


""" hyper-parameters """
cuda = torch.device('cuda:1')
batch_size = 1
train_fold = '/media/fenqiang/DATA/unc/Data/harmonization/paired_data/data_in_mat/train'
test_fold = '/media/fenqiang/DATA/unc/Data/harmonization/paired_data/data_in_mat/test'
in_ch = 1
out_ch = 1


"""Initialize models """
# define networks (both Generators and discriminators)
# The naming is different from those used in the paper.
# Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
netG_A = Unet(in_ch, out_ch).cuda(cuda)
netG_B = Unet(out_ch, in_ch).cuda(cuda)
netG_A.load_state_dict(torch.load('/home/fenqiang/harmonization/model/paired/netG_A_299.pkl'))
netG_B.load_state_dict(torch.load('/home/fenqiang/harmonization/model/paired/netG_B_299.pkl'))


realA_mean = np.load( "/home/fenqiang/harmonization/realA_mean.npy")
realA_std = np.load( "/home/fenqiang/harmonization/realA_std.npy")
realB_mean = np.load( "/home/fenqiang/harmonization/pairB_mean.npy")
realB_std = np.load( "/home/fenqiang/harmonization/pairB_std.npy")
class BrainSphere(torch.utils.data.Dataset):
    """BrainSphere datset
    
    """
    def __init__(self, root):
        """init root --path for data  """
        self.B_files = sorted(glob.glob(root + '/*BCP*_lh_111.InnerSurf.mat'))
        self.B_len = len(self.B_files)

    def __getitem__(self, index):
        B_file = self.B_files[index]
        A_file = '/media/fenqiang/DATA/unc/Data/harmonization/data_in_mat/train/' + B_file.split('/')[-1].split('_')[0] + '_' + B_file.split('/')[-1].split('_')[1] + '_lh.InnerSurf.mat'
        if not os.path.exists(A_file):
            A_file = A_file.replace('train','test')
        
        A = sio.loadmat(A_file)
        A = A['data'][:,0] # 0:thickness, 1: sulc
        B = sio.loadmat(B_file)
        B = B['data'][:,0] # 0:thickness, 1: sulc
        
        A = (A - realA_mean) / realA_std
        B = (B - realB_mean) / realB_std
        
        A = A[:, np.newaxis]
        B = B[:, np.newaxis]
        
        return {'A': A.astype(np.float32), 'B': B.astype(np.float32), 'A_path': A_file, 'B_path': B_file}

    def __len__(self):
        return self.B_len
    

train_dataset = BrainSphere(train_fold)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_dataset = BrainSphere(test_fold)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


#    dataiter = iter(train_dataloader)
#    data = dataiter.next()

for batch_idx, data in enumerate(train_dataloader):

    real_A = data['A'].squeeze(0).cuda(cuda)   # 40962*1
    real_B = data['B'].squeeze(0).cuda(cuda)   # 40962*1
    path_A = data['A_path']
    path_B = data['B_path']
    
    netG_A.eval()
    netG_B.eval()
    
    fake_B = netG_A(real_A)   # G_A(A)         # 40962*1
    rec_A = netG_B(fake_B)    # G_B(G_A(A))    # 40962*1
    fake_A = netG_B(real_B)   # G_B(B)         # 40962*1
    rec_B = netG_A(fake_A)    # G_A(G_B(B))    # 40962*1
    
    # save for normalization input data
    fake_B = (fake_B.detach().squeeze().cpu().numpy() * realB_std) + realB_mean
    rec_A =  (rec_A.detach().squeeze().cpu().numpy() * realA_std) + realA_mean
    fake_A = (fake_A.detach().squeeze().cpu().numpy() * realA_std) + realA_mean
    rec_B = (rec_B.detach().squeeze().cpu().numpy() * realB_std) + realB_mean
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/gan/train_299/' + path_A[0].split('/')[-1].split('.')[0] + '_toB.txt', fake_B)
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/gan/train_299/' + path_A[0].split('/')[-1].split('.')[0] + '_toB_toA.txt', rec_A)
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/gan/train_299/' + path_B[0].split('/')[-1].split('.')[0] + '_toA.txt', fake_A)
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/gan/train_299/' + path_B[0].split('/')[-1].split('.')[0] + '_toA_toB.txt', rec_B)
    
    
#   old style save function, when not normalization for input data
#    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/generated/mseloss_normal_input/train_164/' + path_A[0].split('/')[-1].split('.')[0] + '_toB.txt', fake_B.detach().squeeze().cpu().numpy())
#    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/generated/mseloss_normal_input/train_164/' + path_A[0].split('/')[-1].split('.')[0] + '_toB_toA.txt', rec_A.detach().squeeze().cpu().numpy())
#    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/generated/mseloss_normal_input/train_164/' + path_B[0].split('/')[-1].split('.')[0] + '_toA.txt', fake_A.detach().squeeze().cpu().numpy())
#    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/generated/mseloss_normal_input/train_164/' + path_B[0].split('/')[-1].split('.')[0] + '_toA_toB.txt', rec_B.detach().squeeze().cpu().numpy())