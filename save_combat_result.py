#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 11:53:30 2019

@author: fenqiang
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:26:36 2019

@author: fenqiang
"""

import glob
import numpy as np
import scipy.io as sio 
import matplotlib.pyplot as plt
from neuroCombat import neuroCombat
import pandas as pd


realA_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/data_in_mat/test' + '/*BCP*lh.*')) + \
              sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/data_in_mat/train' + '/*BCP*lh.*'))
realB_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/data_in_mat/test' + '/M0*lh.*'))  + \
              sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/data_in_mat/train' + '/M0*lh.*'))
all_files = realA_files + realB_files



realA_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/data_in_mat_unsmoothed' + '/*BCP*lh.*'))
realB_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/data_in_mat_unsmoothed' + '/M0*lh.*'))
all_files = realA_files + realB_files

realA = np.zeros((len(realA_files), 40962))
realB = np.zeros((len(realB_files), 40962))
ageA = np.zeros(len(realA_files))
ageB = np.zeros(len(realB_files))


for i in range(len(realA_files)):
     data = sio.loadmat(realA_files[i])
     realA[i,:] = np.squeeze(data['data'][:,[0]]) # 0:thickness, 1: sulc
     ageA[i] = int(realA_files[i].split('/')[-1].split('_')[1])

for i in range(len(realB_files)):
     data = sio.loadmat(realB_files[i])
     realB[i,:] = np.squeeze(data['data'][:,[0]]) # 0:thickness, 1: sulc
     ageB[i] = int(realB_files[i].split('/')[-1].split('_')[1]) * 30


data = np.concatenate((realA, realB), axis=0)

# using combat
batch = np.concatenate((np.zeros(len(realA_files)), np.zeros(len(realB_files))+1), 0) + 1
age =  np.concatenate((ageA, ageB), 0)
covars = pd.DataFrame(data={'batch': batch})
continuous_cols = ['age']
batch_col = 'batch'
data_combat = neuroCombat(data=data,
                          covars=covars,
                          batch_col=batch_col)

for i in range(len(data_combat)):
    np.savetxt('/media/fenqiang/DATA/unc/Data/harmonization/generated/Combat/unsmoothed_without_age/' + all_files[i].split('/')[-1].split('.')[0] + '_combat.txt', data_combat[i,:])

