#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  15 11:29:28 2018

@author: fenqiang
"""

import torch
import torchvision
import numpy as np
import glob
import random
import csv

from utils_for_ML import Stratified_split_files
from Common.utils import ImagePool
from Common.model import Unet, Discriminator_MSE
from tensorboardX import SummaryWriter
writer = SummaryWriter('log/a')

###########################################################
""" hyper-parameters """
device = torch.device('cuda:0')
learning_rate = 0.0001
batch_size = 1
lambda_identity = 0.05
lambda_cycle = 10
lambda_cc = 0.5
pooling_type = "mean"  # "max" or "mean" 
out_ch = 1
pool_size = 20
age = 720
data_for_test = 0.3
AGEINFO = True
in_ch = 1
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
bcp_age = [float(x.split('/')[-1].split('_')[1].split('.')[0]) for x in bcp_files]
map_age = [map_age_dic[x.split('/')[-1].split('.')[0][:-3]] for x in map_files]
realA_train_files, realA_test_files = Stratified_split_files(bcp_files, bcp_age, data_for_test)
realB_train_files, realB_test_files = Stratified_split_files(map_files, map_age, data_for_test)
train_files = realA_train_files + realB_train_files
test_files = realA_test_files + realB_test_files

"""Initialize models """
# define networks (both Generators and discriminators)
# The naming is different from those used in the paper.
# Code (vs. paper): G_A (G_X), G_B (G_Y), D_A (D_Y), D_B (D_X)
netG_A = Unet(in_ch, out_ch, AGEINFO).cuda(device)
netG_B = Unet(in_ch, out_ch, AGEINFO).cuda(device) 
netD_A = Discriminator_MSE(in_ch, AGEINFO=AGEINFO).cuda(device)
netD_B = Discriminator_MSE(in_ch, AGEINFO=AGEINFO).cuda(device)

fake_A_pool = ImagePool(pool_size)  # create image buffer to store previously generated images
fake_B_pool = ImagePool(pool_size)  # create image buffer to store previously generated images
# define loss functions
criterionGAN = torch.nn.MSELoss()  # define GAN loss. CrossEntropyLoss or MSELoss
criterionCycle = torch.nn.L1Loss()
criterionIdt = torch.nn.L1Loss()
# initialize optimizers
optimizer_G_A = torch.optim.Adam(netG_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_G_B = torch.optim.Adam(netG_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=learning_rate, betas=(0.5, 0.999))

optimizers = [optimizer_G_A, optimizer_G_B, optimizer_D_A, optimizer_D_B]

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
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
test_dataset = BrainSphere(test_files, map_age_dic, AGEINFO=AGEINFO)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)


def get_learning_rate(epoch):
    limits = [5, 10, 30, 100, 180]
    lrs = [1,  0.5, 0.1, 0.05, 0.01, 0.001]
    assert len(lrs) == len(limits) + 1
    for lim, lr in zip(limits, lrs):
        if epoch < lim:
            return lr * learning_rate
    return lrs[-1] * learning_rate


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def backward_D_basic(netD, real, fake, real_age, fake_age, lamda):
    """Calculate GAN loss for the discriminator

    Parameters:
        netD (network)      -- the discriminator D
        real (tensor array) -- real images
        fake (tensor array) -- images generated by a generator

    Return the discriminator loss.
    We also call loss_D.backward() to calculate the gradients.
    """
    target_real = torch.tensor(1.0).cuda(device)
    target_false = torch.tensor(0.0).cuda(device)

    # Real
    loss_D_real = criterionGAN(netD(real, real_age).squeeze(), target_real)
    # Fake
    loss_D_fake = criterionGAN(netD(fake.detach(), fake_age).squeeze(), target_false)
    # Combined loss and calculate gradients
    loss_D = (loss_D_real + loss_D_fake)/2 * lamda
    loss_D.backward()
    return loss_D.item()



realA_mean_cuda = torch.from_numpy(realA_mean[:,np.newaxis].astype(np.float32)).cuda(device)
realA_std_cuda = torch.from_numpy(realA_std[:,np.newaxis].astype(np.float32)).cuda(device)
realB_mean_cuda = torch.from_numpy(realB_mean[:,np.newaxis].astype(np.float32)).cuda(device)
realB_std_cuda = torch.from_numpy(realB_std[:,np.newaxis].astype(np.float32)).cuda(device)

for epoch in range(300):
    
    lr = get_learning_rate(epoch)
    for optimizer in optimizers:
        optimizer.param_groups[0]['lr'] = lr
    print("learning rate = {}".format(lr))    
    
#    dataiter = iter(train_dataloader)
#    data = dataiter.next()
    
    for batch_idx, data in enumerate(train_dataloader):

        real_A = data['A'].squeeze(0).cuda(device)   # 40962*1
        real_B = data['B'].squeeze(0).cuda(device)   # 40962*1
        realA_age = data['A_age'].float().cuda(device)
        realB_age = data['B_age'].float().cuda(device)
        target_real = torch.tensor(1.0).cuda(device)
        target_false = torch.tensor(0.0).cuda(device)       

        netG_A.train()
        netG_B.train()
        netD_A.train()
        netD_B.train()
        
        """Run forward pass; called by both functions"""
        fake_B = netG_A(real_A, realA_age)  # G_A(A)         # 40962*1
        rec_A = netG_B(fake_B, realA_age)   # G_B(G_A(A))    # 40962*1
        fake_A = netG_B(real_B, realB_age)  # G_B(B)         # 40962*1
        rec_B = netG_A(fake_A, realB_age)   # G_A(G_B(B))    # 40962*1

        """ train G_A and G_B"""
        set_requires_grad([netD_A, netD_B], False)  # Ds require no gradients when optimizing Gs
        optimizer_G_A.zero_grad()  # set G_A and G_B's gradients to zero
        optimizer_G_B.zero_grad()  # set G_A and G_B's gradients to zero
            
        """Calculate the loss for generators G_A and G_B"""
        # Identity loss
        if lambda_identity > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            idt_A = netG_A(real_B, realB_age)   # 40962*1
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            idt_B = netG_B(real_A, realA_age)
            loss_idt_A = criterionIdt(idt_A, real_B) * lambda_cycle * lambda_identity
            loss_idt_B = criterionIdt(idt_B, real_A) * lambda_cycle * lambda_identity
        else:
            loss_idt_A = 0
            loss_idt_B = 0
        
        # GAN loss D_A(G_A(A))
        loss_G_A = criterionGAN(netD_A(fake_B, realA_age).squeeze(), target_real)
        # GAN loss D_B(G_B(B))
        loss_G_B = criterionGAN(netD_B(fake_A, realB_age).squeeze(), target_real)
        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = criterionCycle(rec_A, real_A) * lambda_cycle
        # Backward cycle loss || G_A(G_B(B)) - B||
        loss_cycle_B = criterionCycle(rec_B, real_B) * lambda_cycle 
        # combined loss and calculate gradients
            
        # Correlation coeffecient loss
        real_A_unnorm = real_A * realA_std_cuda + realA_mean_cuda
        real_B_unnorm = real_B * realB_std_cuda + realB_mean_cuda
        fake_A_unnorm = fake_A * realA_std_cuda + realA_mean_cuda
        fake_B_unnorm = fake_B * realB_std_cuda + realB_mean_cuda
        cc_A = ((fake_B_unnorm - fake_B_unnorm.mean()) * (real_A_unnorm - real_A_unnorm.mean())).mean() / fake_B_unnorm.std() / real_A_unnorm.std()   # compute correlation between gan(A) and A
        cc_B = ((fake_A_unnorm - fake_A_unnorm.mean()) * (real_B_unnorm - real_B_unnorm.mean())).mean() / fake_A_unnorm.std() / real_B_unnorm.std()   # compute correlation between gan(B) and B
        
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B - cc_A * lambda_cc - cc_B * lambda_cc
       
        """ calculate gradients for G_A and G_B """
        loss_G.backward()
        
        optimizer_G_A.step()       # update G_A and G_B's weights
        optimizer_G_B.step()       # update G_A and G_B's weights
        
        
        # train D_A and D_B
        set_requires_grad([netD_A, netD_B], True)
        optimizer_D_A.zero_grad()   # set D_A and D_B's gradients to zero
        optimizer_D_B.zero_grad()   # set D_A and D_B's gradients to zero
      
        """Calculate GAN loss for discriminator D_A"""
        fake_B = fake_B_pool.query(fake_B)
        loss_D_A = backward_D_basic(netD_A, real_B, fake_B, realB_age, realA_age, 0.2)

        """Calculate GAN loss for discriminator D_B"""
        fake_A = fake_A_pool.query(fake_A)
        loss_D_B = backward_D_basic(netD_B, real_A, fake_A, realA_age, realB_age, 0.2)
        
        optimizer_D_A.step()  # update D_A and D_B's weights
        optimizer_D_B.step()  # update D_A and D_B's weights
        
        print("[{}:{}/{}] IDT_A= {:5.4f}, IDT_B={:5.4f}, G_A={:5.4f}, G_B={:5.4f}, CYCLE_A={:5.4f}, CYCLE_B={:5.4f}, D_A={:5.4f}, D_B={:5.4f}, CC_A={:5.4f}, CC_B={:5.4f}".format(epoch, 
              batch_idx, len(train_dataloader), loss_idt_A, loss_idt_B,
              loss_G_A, loss_G_B, loss_cycle_A, loss_cycle_B, loss_D_A, loss_D_B, cc_A, cc_B))
        
        writer.add_scalars('Train/IDT_loss', {'A': loss_idt_A.item(), 'B': loss_idt_B.item()}, epoch*len(train_dataloader) + batch_idx)
        writer.add_scalars('Train/GAN_loss', {'A': loss_G_A.item(), 'B': loss_G_B.item()}, epoch*len(train_dataloader) + batch_idx)
        writer.add_scalars('Train/CYCLE_loss', {'A': loss_cycle_A.item(), 'B': loss_cycle_B.item()}, epoch*len(train_dataloader) + batch_idx)
        writer.add_scalars('Train/D_loss', {'A': loss_D_A, 'B': loss_D_B}, epoch*len(train_dataloader) + batch_idx)
        writer.add_scalars('Train/CC_loss', {'A': cc_A.item(), 'B': cc_B.item()}, epoch*len(train_dataloader) + batch_idx)
    
 
    torch.save(netG_A.state_dict(), "model/journal/age_netG_A.pkl")
    torch.save(netG_B.state_dict(), "model/journal/age_netG_B.pkl")
    torch.save(netD_A.state_dict(), "model/journal/age_netD_A.pkl")
    torch.save(netD_B.state_dict(), "model/journal/age_netD_B.pkl")
