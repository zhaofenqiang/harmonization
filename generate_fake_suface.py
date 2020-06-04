#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  15 11:29:28 2018

@author: fenqiang
"""

import torch
import torchvision
import scipy.io as sio 
import numpy as np
import glob
import random
import csv

from utils_for_ML import Stratified_split_files
from Common.model import Unet

###########################################################
""" hyper-parameters """
device = torch.device('cuda:0')
batch_size = 1
in_ch = 1
out_ch = 1
age = 720
data_for_test =  0.3
AGEINFO = True
###########################################################

""" split files  """
map_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/journal/data/M0*lh.*.npy'))
bcp_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/journal/data/*BCP*lh.*.npy'))
map_age_data = csv.reader(open('/media/fenqiang/DATA/unc/Data/harmonization/MapDayInfo.csv', 'r'))
map_age_dic = {}
for row in map_age_data:
    map_age_dic[row[0]] = float(row[1])
map_files = [x for x in map_files if map_age_dic[x.split('/')[-1].split('.')[0][:-3]] <= age ]
bcp_files = [x for x in bcp_files if float(x.split('/')[-1].split('_')[1].split('.')[0]) <= age ]
train_files = bcp_files + map_files


"""Initialize models """
# define networks (both Generators and discriminators)
# The naming is different from those used in the paper.
# Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
netG_A = Unet(in_ch, out_ch, AGEINFO).cuda(device)
netG_B = Unet(in_ch, out_ch, AGEINFO).cuda(device)
if AGEINFO:
    netG_A.load_state_dict(torch.load('/home/fenqiang/harmonization/model/journal/age_netG_A.pkl'))
    netG_B.load_state_dict(torch.load('/home/fenqiang/harmonization/model/journal/age_netG_B.pkl'))
else:
    netG_A.load_state_dict(torch.load('/home/fenqiang/harmonization/model/journal/orig_netG_A.pkl'))
    netG_B.load_state_dict(torch.load('/home/fenqiang/harmonization/model/journal/orig_netG_B.pkl'))

realA_mean = np.load('/media/fenqiang/DATA/unc/Data/harmonization/realA_mean.npy')
realA_std = np.load('/media/fenqiang/DATA/unc/Data/harmonization/realA_std.npy')
realB_mean = np.load('/media/fenqiang/DATA/unc/Data/harmonization/realB_mean.npy')
realB_std = np.load('/media/fenqiang/DATA/unc/Data/harmonization/realB_std.npy')

class BrainSphere(torch.utils.data.Dataset):
    """BrainSphere datset
    
    """
    def __init__(self, files, map_age_dic, AGEINFO=True):
        self.A_files = [x for x in files if 'BCP' in x ]
        self.B_files = [x for x in files if 'M0' in x]
        assert set(self.A_files + self.B_files) == set(files), "MAP files and BCP files not split successfully!"
        self.A_len = len(self.A_files)
        self.B_len = len(self.B_files)
        self.AGEINFO = AGEINFO
        self.map_age_dic = map_age_dic

    def __getitem__(self, index):
        if index < min(self.A_len, self.B_len):
            A_file = self.A_files[index]
            B_file = self.B_files[index]
        else:
            if self.A_len > self.B_len:
                A_file = self.A_files[index]
                B_file = self.B_files[random.randint(0, self.B_len-1)]
            else:
                A_file = self.A_files[random.randint(0, self.A_len-1)]
                B_file = self.B_files[index]
                
        A = np.load(A_file)
        B = np.load(B_file)
        
        A = (A - realA_mean) / realA_std
        B = (B - realB_mean) / realB_std
        
        A = A[:, np.newaxis]
        B = B[:, np.newaxis]
        
        if self.AGEINFO:
            A_age = float(A_file.split('/')[-1].split('_')[1].split('.')[0])
            B_age = self.map_age_dic[B_file.split('/')[-1].split('.')[0][:-3]] 
        
        return {'A': A.astype(np.float32), 'B': B.astype(np.float32), 'A_path': A_file, 'B_path': B_file, 'A_age': A_age/720.0, 'B_age': B_age/720.0}

    def __len__(self):
        return max(self.A_len, self.B_len)

train_dataset = BrainSphere(train_files, map_age_dic, AGEINFO=AGEINFO)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

#    dataiter = iter(train_dataloader)
#    data = dataiter.next()

for batch_idx, data in enumerate(train_dataloader):

    real_A = data['A'].squeeze(0).cuda(device)   # 40962*1
    real_B = data['B'].squeeze(0).cuda(device)   # 40962*1
    realA_age = data['A_age'].float().cuda(device)
    realB_age = data['B_age'].float().cuda(device)
    path_A = data['A_path']
    path_B = data['B_path']
    
    netG_A.eval()
    netG_B.eval()
    
    fake_B = netG_A(real_A, realA_age)  # G_A(A)         # 40962*1
    rec_A = netG_B(fake_B, realA_age)   # G_B(G_A(A))    # 40962*1
    fake_A = netG_B(real_B, realB_age)  # G_B(B)         # 40962*1
    rec_B = netG_A(fake_A, realB_age)   # G_A(G_B(B))    # 40962*1
    
    fake_B = (fake_B.detach().squeeze().cpu().numpy() * realB_std) + realB_mean
    rec_A =  (rec_A.detach().squeeze().cpu().numpy() * realA_std) + realA_mean
    fake_A = (fake_A.detach().squeeze().cpu().numpy() * realA_std) + realA_mean
    rec_B = (rec_B.detach().squeeze().cpu().numpy() * realB_std) + realB_mean
    np.save('/media/fenqiang/DATA/unc/Data/harmonization/journal/generated/age/' + path_A[0].split('/')[-1].replace('.npy', '.toB.npy'), fake_B)
    np.save('/media/fenqiang/DATA/unc/Data/harmonization/journal/generated/age/' + path_A[0].split('/')[-1].replace('.npy', '.toB_toA.npy'), rec_A)
    np.save('/media/fenqiang/DATA/unc/Data/harmonization/journal/generated/age/' + path_B[0].split('/')[-1].replace('.npy', '.toA.npy'), fake_A)
    np.save('/media/fenqiang/DATA/unc/Data/harmonization/journal/generated/age/' + path_B[0].split('/')[-1].replace('.npy', '.toA_toB.npy'), rec_B)
    