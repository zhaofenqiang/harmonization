#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 09:26:21 2019

@author: fenqiang
"""
import numpy as np
import random
import glob
import scipy.io as sio 
import os
import csv

def group(data, target, *grp_range):
    """
    extract data in 'data' to group, according to their target range
    """
    n_grp = len(grp_range)
    grp = []
    for i in range(n_grp):
        lower = grp_range[i][0]
        upper = grp_range[i][1]
        index1 = np.where(target >= lower)
        index2 = np.where(target <= upper)
        index = np.intersect1d(index1, index2)
        grp.append(data[index])
     
    return grp


def cohensd(grp1, grp2):
    """
    compute cohen's d for two groups
    grp1, grp2: [n_samples, n_features]
    """
    n_g1 = len(grp1)
    n_g2 = len(grp2)
    g1_mean = grp1.mean(0)
    g2_mean = grp2.mean(0)
    g1_var = grp1.var(0)
    g2_var = grp2.var(0)
    spooled = np.sqrt(((n_g1-1)*g1_var + (n_g2-1)*g2_var)/(n_g1 + n_g2 - 2))
    
    d = (g1_mean - g2_mean) / spooled
    
    return d


def compute_cd_for_all_grp(grp):
    n_grp = len(grp)
#    cd = np.zeros((n_grp, n_grp, np.shape(grp[0])[1]))   for detail debug
    cd = np.zeros((n_grp, n_grp))
    for i in range(n_grp):
        for j in range(n_grp):
            if i==j:
                continue
            else:
                #cd[i,j,:] = cohensd(grp[i], grp[j])
                cd[i,j] = np.abs(cohensd(grp[i], grp[j])).mean()
   
    return cd
    

    
def Stratified_split_files(files, age, test_size=0.25, split=10):
    """
    return split files according to age info
    
    files: list
    ages: list
    """
    age = np.asarray(age)
    threshold = np.linspace(np.min(age), np.max(age), num=split+1)
    split_files = []
    for i in range(10):
        lower = threshold[i]
        upper = threshold[i+1]
        index1 = np.where(age >= lower)
        index2 = np.where(age < upper)
        if i == 9:
            index2 = np.where(age <= upper)
        index = np.intersect1d(index1, index2)
        split_files.append([files[x] for x in index])
        
    test = []
    train = []
    for i in range(split):
        test_len = int(len(split_files[i]) * test_size)
        test = test + [ split_files[i][x] for x in range(test_len) ]
        train = train + [ split_files[i][x] for x in range(test_len, len(split_files[i])) ]
        
    return train, test


    
def Stratified_split(data, target, test_size=0.3, split=10):
    """
    return split data and target using stratified target
    """
    threshold = np.linspace(np.min(target), np.max(target), num=split+1)
    split_data = []
    split_target = []
    for i in range(10):
        lower = threshold[i]
        upper = threshold[i+1]
        index1 = np.where(target >= lower)
        index2 = np.where(target < upper)
        if i == 9:
            index2 = np.where(target <= upper)
        index = np.intersect1d(index1, index2)
        split_data.append(data[index])
        split_target.append(target[index])
        
    for i in range(split):
        test_len = int(len(split_data[i]) * test_size)
        if i == 0:
            X_train = split_data[i][test_len:,:]
            X_test = split_data[i][0:test_len,:]
            y_train = split_target[i][test_len:]
            y_test = split_target[i][0:test_len]
        else:
            X_train = np.concatenate((X_train, split_data[i][test_len:,:]),0)
            X_test = np.concatenate((X_test, split_data[i][0:test_len,:]), 0)
            y_train = np.concatenate((y_train, split_target[i][test_len:]), 0)
            y_test = np.concatenate((y_test, split_target[i][0:test_len]), 0)
            
    return X_train, X_test, y_train, y_test
            
    
    
class Stratified_split_cv():
    """
    using target age as stratified variable, return cross validation index
    """
    def __init__(self, n_splits=5, test_size=0.4, target=None):
        
        self.test_size = test_size
        self.n_splits = n_splits
        self.n_stratification = 10
        threshold = np.linspace(np.min(target), np.max(target), num=self.n_stratification+1)
        self.split_index = []
        for i in range(10):
            lower = threshold[i]
            upper = threshold[i+1]
            index1 = np.where(target >= lower)
            index2 = np.where(target < upper)
            if i == 9:
                index2 = np.where(target <= upper)
            self.split_index.append(np.intersect1d(index1, index2))
            

    
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
    
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
    
        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.
    
        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
    
        Yields
        ------
        train : ndarray
            The training set indices for that split.
    
        test : ndarray
            The testing set indices for that split.
        """
        for i in range(self.n_splits):
            for j in range(self.n_stratification):
                n_test = int(len(self.split_index[j]) * self.test_size)
                if j == 0:
                    ind_test = np.asarray(random.sample(list(self.split_index[j]), n_test), dtype=np.int64)
                    ind_train = np.setdiff1d(self.split_index[j], ind_test)
                else:
                    tmp = np.asarray(random.sample(list(self.split_index[j]), n_test), dtype=np.int64)
                    ind_train = np.concatenate((ind_train, np.setdiff1d(self.split_index[j], tmp)), 0)
                    ind_test = np.concatenate((ind_test, tmp), 0)

            yield ind_train, ind_test
            
            
    def get_n_splits(self, X=None, y=None, groups=None):
        """Returns the number of splitting iterations in the cross-validator

        Parameters
        ----------
        X : object
            Always ignored, exists for compatibility.

        y : object
            Always ignored, exists for compatibility.

        groups : object
            Always ignored, exists for compatibility.

        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


def get_data_from_files(realA_train_files,realB_train_files,realA_test_files,realB_test_files,fakeA_train_files,fakeB_train_files,fakeA_test_files,fakeB_test_files,vertex,paired):
    map_data = csv.reader(open('/media/fenqiang/DATA/unc/Data/harmonization/MapDayInfo.csv', 'r'))
    map_age_dic = {}
    for row in map_data:
        map_age_dic[row[0]] = int(row[1])
    
    if vertex:
        n_roi = 40962
    else:
        # read atlas template to get roi feature
        atlas = sio.loadmat('/media/fenqiang/DATA/unc/Data/harmonization/atlas_par40962_36.mat')
        atlas = atlas['par40962_36'][:,0]
            
        par_1 = sio.loadmat('/media/fenqiang/DATA/unc/zhengwang/scripts/par_FS_to_par_vec.mat')
        par_1 = par_1['pa'].astype(np.int64)
        par_2 = np.loadtxt('/media/fenqiang/DATA/unc/zhengwang/scripts/FScolortable.txt')
        par_2 = par_2[:,0:4].astype(np.int64)
        
        roi_label = []
        for i in range(len(par_2)):
            par_vec = par_2[i,1:]
            for j in range(len(par_1)):
                if (par_vec == par_1[j,0:3]).all():
                    roi_label.append(par_1[j,3])
                    
        roi_label = np.asarray(roi_label)
        n_roi = len(roi_label)
        roi_list = []
        for i in range(n_roi):
            roi_list.append(np.argwhere(atlas == roi_label[i]))
        
    realA_train = np.zeros((len(realA_train_files), n_roi))
    realB_train = np.zeros((len(realB_train_files), n_roi))
    fakeA_train = np.zeros((len(fakeA_train_files), n_roi))
    fakeB_train = np.zeros((len(fakeB_train_files), n_roi))
    realA_test = np.zeros((len(realA_test_files), n_roi))
    realB_test = np.zeros((len(realB_test_files), n_roi))
    fakeA_test = np.zeros((len(fakeA_test_files), n_roi))
    fakeB_test = np.zeros((len(fakeB_test_files), n_roi))
    
    realA_train_age = np.zeros(len(realA_train))
    realB_train_age = np.zeros(len(realB_train))
    fakeA_train_age = np.zeros(len(fakeA_train))
    fakeB_train_age = np.zeros(len(fakeB_train))
    realA_test_age = np.zeros(len(realA_test))
    realB_test_age = np.zeros(len(realB_test))
    fakeA_test_age = np.zeros(len(fakeA_test))
    fakeB_test_age = np.zeros(len(fakeB_test))
        
    if vertex:
        for i in range(len(realA_train_files)):
            data = sio.loadmat(realA_train_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            realA_train[i,:] = data
            realA_train_age[i] = int(realA_train_files[i].split('/')[-1].split('_')[1])
            
            
        for i in range(len(realB_train_files)):
            data = sio.loadmat(realB_train_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            realB_train[i,:] = data
            if paired:
                realB_train_age[i] = int(realB_train_files[i].split('/')[-1].split('_')[1])
            else:
                realB_train_age[i] = map_age_dic[realB_train_files[i].split('/')[-1].split('_')[0] + '_' + realB_train_files[i].split('/')[-1].split('_')[1]]
        
        for i in range(len(fakeA_train_files)):
            data = sio.loadmat(fakeA_train_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            fakeA_train[i,:] = data
            if paired:
                fakeA_train_age[i] = int(fakeA_train_files[i].split('/')[-1].split('_')[1])
            else:
                fakeA_train_age[i] = map_age_dic[fakeA_train_files[i].split('/')[-1].split('_')[0] + '_' + fakeA_train_files[i].split('/')[-1].split('_')[1]]
        
        for i in range(len(fakeB_train_files)):
            data = sio.loadmat(fakeB_train_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            fakeB_train[i,:] = data
            fakeB_train_age[i] = int(fakeB_train_files[i].split('/')[-1].split('_')[1])
        
        for i in range(len(realA_test_files)):
            data = sio.loadmat(realA_test_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            realA_test[i,:] = data
            realA_test_age[i] = int(realA_test_files[i].split('/')[-1].split('_')[1])
        
        for i in range(len(realB_test_files)):
            data = sio.loadmat(realB_test_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            realB_test[i,:] = data
            if paired:
                realB_test_age[i] = int(realB_test_files[i].split('/')[-1].split('_')[1])
            else:
                realB_test_age[i] = map_age_dic[realB_test_files[i].split('/')[-1].split('_')[0] + '_' + realB_test_files[i].split('/')[-1].split('_')[1]]
        
        for i in range(len(fakeA_test_files)):
            data = sio.loadmat(fakeA_test_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            fakeA_test[i,:] = data
            if paired:
                fakeA_test_age[i] = int(fakeA_test_files[i].split('/')[-1].split('_')[1])
            else:
                fakeA_test_age[i] = map_age_dic[fakeA_test_files[i].split('/')[-1].split('_')[0] + '_' + fakeA_test_files[i].split('/')[-1].split('_')[1]]
        
        for i in range(len(fakeB_test_files)):
            data = sio.loadmat(fakeB_test_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            fakeB_test[i,:] = data
            fakeB_test_age[i] = int(fakeB_test_files[i].split('/')[-1].split('_')[1])
            
    else:
        for i in range(len(realA_train_files)):
            data = sio.loadmat(realA_train_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            for j in range(n_roi):
                realA_train[i,j] = np.mean(data[roi_list[j]])
            realA_train_age[i] = int(realA_train_files[i].split('/')[-1].split('_')[1])
            
        for i in range(len(realB_train_files)):
            data = sio.loadmat(realB_train_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            for j in range(n_roi):
                realB_train[i,j] = np.mean(data[roi_list[j]])
            if paired:
                realB_train_age[i] = int(realB_train_files[i].split('/')[-1].split('_')[1])
            else:
                realB_train_age[i] = map_age_dic[realB_train_files[i].split('/')[-1].split('_')[0] + '_' + realB_train_files[i].split('/')[-1].split('_')[1]]
        
        for i in range(len(fakeA_train_files)):
            data = sio.loadmat(fakeA_train_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            for j in range(n_roi):
                fakeA_train[i,j] = np.mean(data[roi_list[j]])
            if paired:
                fakeA_train_age[i] = int(fakeA_train_files[i].split('/')[-1].split('_')[1])
            else:
                fakeA_train_age[i] = map_age_dic[fakeA_train_files[i].split('/')[-1].split('_')[0] + '_' + fakeA_train_files[i].split('/')[-1].split('_')[1]]
        
        for i in range(len(fakeB_train_files)):
            data = sio.loadmat(fakeB_train_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            for j in range(n_roi):
                fakeB_train[i,j] = np.mean(data[roi_list[j]])
            fakeB_train_age[i] = int(fakeB_train_files[i].split('/')[-1].split('_')[1])
        
        for i in range(len(realA_test_files)):
            data = sio.loadmat(realA_test_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            for j in range(n_roi):
                realA_test[i,j] = np.mean(data[roi_list[j]])
            realA_test_age[i] = int(realA_test_files[i].split('/')[-1].split('_')[1])
        
        for i in range(len(realB_test_files)):
            data = sio.loadmat(realB_test_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            for j in range(n_roi):
                realB_test[i,j] = np.mean(data[roi_list[j]])
            if paired:
                realB_test_age[i] = int(realB_test_files[i].split('/')[-1].split('_')[1])
            else:
                realB_test_age[i] = map_age_dic[realB_test_files[i].split('/')[-1].split('_')[0] + '_' + realB_test_files[i].split('/')[-1].split('_')[1]]
        
        for i in range(len(fakeA_test_files)):
            data = sio.loadmat(fakeA_test_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            for j in range(n_roi):
                fakeA_test[i,j] = np.mean(data[roi_list[j]])
            if paired:
                fakeA_test_age[i] = int(fakeA_test_files[i].split('/')[-1].split('_')[1])
            else:
                fakeA_test_age[i] = map_age_dic[fakeA_test_files[i].split('/')[-1].split('_')[0] + '_' + fakeA_test_files[i].split('/')[-1].split('_')[1]]
        
        for i in range(len(fakeB_test_files)):
            data = sio.loadmat(fakeB_test_files[i])
            data = data['data'][:,0]   # 0:thickness, 1: sulc
            for j in range(n_roi):
                fakeB_test[i,j] = np.mean(data[roi_list[j]])
            fakeB_test_age[i] = int(fakeB_test_files[i].split('/')[-1].split('_')[1])
        
    return  realA_train, realB_train, fakeA_train, fakeB_train, realA_test, realB_test, fakeA_test, fakeB_test, realA_train_age, \
           realB_train_age, fakeA_train_age, fakeB_train_age, realA_test_age, realB_test_age, fakeA_test_age, fakeB_test_age


def get_pair_data(method='gan', vertex=True, epoch='300'):
    realB_train_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/data_in_mat/train' + '/*BCP*_lh_111.InnerSurf.mat'))
    realB_test_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/data_in_mat/test' + '/*BCP*_lh_111.InnerSurf.mat'))
    realA_train_files = [a.replace('/paired_data/', '/').replace('_111.', '.') for a in realB_train_files]
    realA_test_files = [a.replace('/paired_data/', '/').replace('_111.', '.') for a in realB_test_files]
    for i in range(len(realA_train_files)):
        if not os.path.exists(realA_train_files[i]):
            realA_train_files[i] = realA_train_files[i].replace('train', 'test')
    for i in range(len(realA_test_files)):
        if not os.path.exists(realA_test_files[i]):
            realA_test_files[i] = realA_test_files[i].replace('test', 'train')
    
    realA_train_subjects = []
    for i in range(len(realA_train_files)):
        realA_train_subjects.append(realA_train_files[i].split('/')[-1].split('.')[0][:-3])

    if method == 'gan':
        fakeA_train_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/gan/train_' + epoch + '/*_lh_111_toA.mat'))
        fakeB_train_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/gan/train_' + epoch + '/*BCP*_lh_toB.mat'))
        fakeA_test_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/gan/test_' + epoch + '/*_lh_111_toA.mat'))
        fakeB_test_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/gan/test_' + epoch + '/*BCP*_lh_toB.mat'))
        
    else:
        fakeA_train_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/' + method + '/*_lh_111_' + method + '.mat'))
        fakeB_train_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/paired_data/generated/' + method + '/*_lh_' + method + '.mat'))

        fakeB_test_files = []
        fakeB_train_files_2 = []
        for i in range(len(fakeB_train_files)):
            if fakeB_train_files[i].split('/')[-1].split('_')[0]+'_'+fakeB_train_files[i].split('/')[-1].split('_')[1] in realA_train_subjects:
                fakeB_train_files_2.append(fakeB_train_files[i])
            else:
                fakeB_test_files.append(fakeB_train_files[i])
        fakeB_train_files = fakeB_train_files_2        
                
        fakeA_test_files = []
        fakeA_train_files_2 = []
        for i in range(len(fakeA_train_files)):
            if  fakeA_train_files[i].split('/')[-1].split('_')[0]+'_'+fakeA_train_files[i].split('/')[-1].split('_')[1] in realA_train_subjects:
                fakeA_train_files_2.append(fakeA_train_files[i])
            else:
                fakeA_test_files.append(fakeA_train_files[i])
        fakeA_train_files = fakeA_train_files_2
    
        fakeA_train_files = sorted(fakeA_train_files)
        fakeA_test_files = sorted(fakeA_test_files)
        fakeB_train_files = sorted(fakeB_train_files)
        fakeB_test_files = sorted(fakeB_test_files)

    return get_data_from_files(realA_train_files,realB_train_files,realA_test_files,realB_test_files,
                               fakeA_train_files,fakeB_train_files,fakeA_test_files,fakeB_test_files, vertex=vertex, paired=True)    

def get_data(method='gan', vertex=True, epoch='300'):
    realA_train_files =  sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/data_in_mat/train' + '/*BCP*lh.*'))
    realB_train_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/data_in_mat/train' + '/M0*lh.*'))
    realA_test_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/data_in_mat/test' + '/*BCP*lh.*'))
    realB_test_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/data_in_mat/test' + '/M0*lh.*'))
    
    realA_train_subjects = []
    for i in range(len(realA_train_files)):
        realA_train_subjects.append(realA_train_files[i].split('/')[-1].split('.')[0][:-3])
    realB_train_subjects = []
    for i in range(len(realB_train_files)):
        realB_train_subjects.append(realB_train_files[i].split('/')[-1].split('.')[0][:-3])
    
    if method == 'gan':
        fakeA_train_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/generated/only_young/train_' + epoch + '/M0*_lh_toA.mat'))
        fakeB_train_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/generated/only_young/train_' + epoch + '/*BCP*_lh_toB.mat'))
        fakeA_test_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/generated/only_young/test_' + epoch + '/M0*_lh_toA.mat'))
        fakeB_test_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/generated/only_young/test_' + epoch + '/*BCP*_lh_toB.mat'))
  
    else:
        fakeA_train_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/generated/' + method + '/M0*_lh_' + method + '.mat'))
        fakeB_train_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/generated/' + method + '/*BCP*_lh_' + method + '.mat'))

        fakeB_test_files = []
        fakeB_train_files_2 = []
        for i in range(len(fakeB_train_files)):
            if fakeB_train_files[i].split('/')[-1].split('_')[0]+'_'+fakeB_train_files[i].split('/')[-1].split('_')[1] in realA_train_subjects:
                fakeB_train_files_2.append(fakeB_train_files[i])
            else:
                fakeB_test_files.append(fakeB_train_files[i])
        fakeB_train_files = fakeB_train_files_2        
                
        fakeA_test_files = []
        fakeA_train_files_2 = []
        for i in range(len(fakeA_train_files)):
            if  fakeA_train_files[i].split('/')[-1].split('_')[0]+'_'+fakeA_train_files[i].split('/')[-1].split('_')[1] in realB_train_subjects:
                fakeA_train_files_2.append(fakeA_train_files[i])
            else:
                fakeA_test_files.append(fakeA_train_files[i])
        fakeA_train_files = fakeA_train_files_2
    
        fakeA_train_files = sorted(fakeA_train_files)
        fakeA_test_files = sorted(fakeA_test_files)
        fakeB_train_files = sorted(fakeB_train_files)
        fakeB_test_files = sorted(fakeB_test_files)

    return get_data_from_files(realA_train_files,realB_train_files,realA_test_files,realB_test_files,
                               fakeA_train_files,fakeB_train_files,fakeA_test_files,fakeB_test_files, vertex=vertex, paired=False)    



def get_npy_data(info='age', n_roi=40962):
    data_for_test = 0.3
    age = 720
    
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
    
    fakeA_train_files = [x.replace('data', 'generated/'+info).replace('.npy', '.toA.npy') for x in realB_train_files]
    fakeA_test_files = [x.replace('data', 'generated/'+info).replace('.npy', '.toA.npy') for x in realB_test_files]
    fakeB_train_files = [x.replace('data', 'generated/'+info).replace('.npy', '.toB.npy') for x in realA_train_files]
    fakeB_test_files = [x.replace('data', 'generated/'+info).replace('.npy', '.toB.npy') for x in realA_test_files]

    return get_npy_data_from_files(realA_train_files,realB_train_files,realA_test_files,realB_test_files,
                               fakeA_train_files,fakeB_train_files,fakeA_test_files,fakeB_test_files, n_roi=n_roi)    
    
def get_npy_data_from_files(realA_train_files,realB_train_files,realA_test_files,realB_test_files,fakeA_train_files,fakeB_train_files,fakeA_test_files,fakeB_test_files, n_roi=40962):       
    map_data = csv.reader(open('/media/fenqiang/DATA/unc/Data/harmonization/MapDayInfo.csv', 'r'))
    map_age_dic = {}
    for row in map_data:
        map_age_dic[row[0]] = int(row[1])
        
    if n_roi == 36:
        # read atlas template to get roi feature
        atlas = sio.loadmat('/media/fenqiang/DATA/unc/Data/harmonization/atlas_par40962_36.mat')
        atlas = atlas['par40962_36'][:,0]
            
        par_1 = sio.loadmat('/media/fenqiang/DATA/unc/Data/Template/par_FS_to_par_vec.mat')
        par_1 = par_1['pa'].astype(np.int64)
        par_2 = np.loadtxt('/media/fenqiang/DATA/unc/Data/Template/FScolortable.txt')
        par_2 = par_2[:,0:4].astype(np.int64)
        
        roi_label = []
        for i in range(len(par_2)):
            par_vec = par_2[i,1:]
            for j in range(len(par_1)):
                if (par_vec == par_1[j,0:3]).all():
                    roi_label.append(par_1[j,3])
        roi_label = np.asarray(roi_label)
        n_roi = len(roi_label)
        roi_list = []
        for i in range(n_roi):
            roi_list.append(np.argwhere(atlas == roi_label[i]))
    elif n_roi == 181:
        # read atlas template to get roi feature
        atlas = sio.loadmat('/media/fenqiang/DATA/unc/Data/harmonization/atlas_par40962_181.mat')
        atlas = atlas['par40962'][:,0]
        roi_label = np.unique(atlas)
        roi_list = []
        for i in range(len(roi_label)):
            roi_list.append(np.argwhere(atlas == roi_label[i])) 
    
    realA_train = np.zeros((len(realA_train_files), n_roi))
    realB_train = np.zeros((len(realB_train_files), n_roi))
    fakeA_train = np.zeros((len(fakeA_train_files), n_roi))
    fakeB_train = np.zeros((len(fakeB_train_files), n_roi))
    realA_test = np.zeros((len(realA_test_files), n_roi))
    realB_test = np.zeros((len(realB_test_files), n_roi))
    fakeA_test = np.zeros((len(fakeA_test_files), n_roi))
    fakeB_test = np.zeros((len(fakeB_test_files), n_roi))
    
    realA_train_age = np.zeros(len(realA_train))
    realB_train_age = np.zeros(len(realB_train))
    fakeA_train_age = np.zeros(len(fakeA_train))
    fakeB_train_age = np.zeros(len(fakeB_train))
    realA_test_age = np.zeros(len(realA_test))
    realB_test_age = np.zeros(len(realB_test))
    fakeA_test_age = np.zeros(len(fakeA_test))
    fakeB_test_age = np.zeros(len(fakeB_test))
    
    if n_roi == 40962:
        for i in range(len(realA_train_files)):
            data = np.load(realA_train_files[i])
            realA_train[i,:] = data
            realA_train_age[i] = int(realA_train_files[i].split('/')[-1].split('_')[1].split('.')[0])
            
        for i in range(len(realB_train_files)):
            data = np.load(realB_train_files[i])
            realB_train[i,:] = data
            realB_train_age[i] = map_age_dic[realB_train_files[i].split('/')[-1].split('.')[0][:-3]]
        
        for i in range(len(realA_test_files)):
            data = np.load(realA_test_files[i])
            realA_test[i,:] = data
            realA_test_age[i] = int(realA_test_files[i].split('/')[-1].split('_')[1].split('.')[0])
        
        for i in range(len(realB_test_files)):
            data = np.load(realB_test_files[i])
            realB_test[i,:] = data
            realB_test_age[i] = map_age_dic[realB_test_files[i].split('/')[-1].split('.')[0][:-3]]
            
        for i in range(len(fakeA_train_files)):
            data = np.load(fakeA_train_files[i])
            fakeA_train[i,:] = data
            fakeA_train_age[i] = map_age_dic[fakeA_train_files[i].split('/')[-1].split('.')[0][:-3]]
            
        for i in range(len(fakeB_train_files)):
            data = np.load(fakeB_train_files[i])
            fakeB_train[i,:] = data
            fakeB_train_age[i] = int(fakeB_train_files[i].split('/')[-1].split('_')[1].split('.')[0])
        
        for i in range(len(fakeA_test_files)):
            data = np.load(fakeA_test_files[i])
            fakeA_test[i,:] = data
            fakeA_test_age[i] = map_age_dic[fakeA_test_files[i].split('/')[-1].split('.')[0][:-3]]
        
        for i in range(len(fakeB_test_files)):
            data = np.load(fakeB_test_files[i])
            fakeB_test[i,:] = data
            fakeB_test_age[i] = int(fakeB_test_files[i].split('/')[-1].split('_')[1].split('.')[0])
    else:
        for i in range(len(realA_train_files)):
            data = np.load(realA_train_files[i])
            for j in range(n_roi):
                realA_train[i,j] = np.mean(data[roi_list[j]])
            realA_train_age[i] = int(realA_train_files[i].split('/')[-1].split('_')[1].split('.')[0])
            
        for i in range(len(realB_train_files)):
            data = np.load(realB_train_files[i])
            for j in range(n_roi):
                realB_train[i,j] = np.mean(data[roi_list[j]])
            realB_train_age[i] = map_age_dic[realB_train_files[i].split('/')[-1].split('.')[0][:-3]]
        
        for i in range(len(realA_test_files)):
            data = np.load(realA_test_files[i])
            for j in range(n_roi):
                realA_test[i,j] = np.mean(data[roi_list[j]])
            realA_test_age[i] = int(realA_test_files[i].split('/')[-1].split('_')[1].split('.')[0])
        
        for i in range(len(realB_test_files)):
            data = np.load(realB_test_files[i])
            for j in range(n_roi):
                realB_test[i,j] = np.mean(data[roi_list[j]])
            realB_test_age[i] = map_age_dic[realB_test_files[i].split('/')[-1].split('.')[0][:-3]]
            
        for i in range(len(fakeA_train_files)):
            data = np.load(fakeA_train_files[i])
            for j in range(n_roi):
                fakeA_train[i,j] = np.mean(data[roi_list[j]])
            fakeA_train_age[i] = map_age_dic[fakeA_train_files[i].split('/')[-1].split('.')[0][:-3]]
            
        for i in range(len(fakeB_train_files)):
            data = np.load(fakeB_train_files[i])
            for j in range(n_roi):
                fakeB_train[i,j] = np.mean(data[roi_list[j]])
            fakeB_train_age[i] = int(fakeB_train_files[i].split('/')[-1].split('_')[1].split('.')[0])
        
        for i in range(len(fakeA_test_files)):
            data = np.load(fakeA_test_files[i])
            for j in range(n_roi):
                fakeA_test[i,j] = np.mean(data[roi_list[j]])
            fakeA_test_age[i] = map_age_dic[fakeA_test_files[i].split('/')[-1].split('.')[0][:-3]]
        
        for i in range(len(fakeB_test_files)):
            data = np.load(fakeB_test_files[i])
            for j in range(n_roi):
                fakeB_test[i,j] = np.mean(data[roi_list[j]])
            fakeB_test_age[i] = int(fakeB_test_files[i].split('/')[-1].split('_')[1].split('.')[0])
    
    return  realA_train, realB_train, fakeA_train, fakeB_train, realA_test, realB_test, fakeA_test, fakeB_test, realA_train_age, \
           realB_train_age, fakeA_train_age, fakeB_train_age, realA_test_age, realB_test_age, fakeA_test_age, fakeB_test_age


def get_data_for_raw(realA_train_files,realB_train_files,realA_test_files,realB_test_files,n_roi=40962,level=40962):
    map_data = csv.reader(open('/media/fenqiang/DATA/unc/Data/harmonization/MapDayInfo.csv', 'r'))
    map_age_dic = {}
    for row in map_data:
        map_age_dic[row[0]] = int(row[1])
    
    if n_roi == 36:
        # read atlas template to get roi feature
        atlas = sio.loadmat('/media/fenqiang/DATA/unc/Data/harmonization/atlas_par40962_36.mat')
        atlas = atlas['par40962_36'][:,0]
            
        par_1 = sio.loadmat('/media/fenqiang/DATA/unc/Data/Template/par_FS_to_par_vec.mat')
        par_1 = par_1['pa'].astype(np.int64)
        par_2 = np.loadtxt('/media/fenqiang/DATA/unc/Data/Template/FScolortable.txt')
        par_2 = par_2[:,0:4].astype(np.int64)
        
        roi_label = []
        for i in range(len(par_2)):
            par_vec = par_2[i,1:]
            for j in range(len(par_1)):
                if (par_vec == par_1[j,0:3]).all():
                    roi_label.append(par_1[j,3])
                    
        roi_label = np.asarray(roi_label)
        n_roi = len(roi_label)
        roi_list = []
        for i in range(n_roi):
            roi_list.append(np.argwhere(atlas == roi_label[i]))
    elif n_roi == 181:
        # read atlas template to get roi feature
        atlas = sio.loadmat('/media/fenqiang/DATA/unc/Data/harmonization/atlas_par40962_181.mat')
        atlas = atlas['par40962'][:,0]
        roi_label = np.unique(atlas)
        roi_list = []
        for i in range(len(roi_label)):
            roi_list.append(np.argwhere(atlas == roi_label[i])) 
        
    realA_train = np.zeros((len(realA_train_files), n_roi))
    realB_train = np.zeros((len(realB_train_files), n_roi))
    realA_test = np.zeros((len(realA_test_files), n_roi))
    realB_test = np.zeros((len(realB_test_files), n_roi))
    
    realA_train_age = np.zeros(len(realA_train))
    realB_train_age = np.zeros(len(realB_train))
    realA_test_age = np.zeros(len(realA_test))
    realB_test_age = np.zeros(len(realB_test))
        
    if n_roi == 40962:
        for i in range(len(realA_train_files)):
            data = np.load(realA_train_files[i])
            realA_train[i,:] = data
            realA_train_age[i] = int(realA_train_files[i].split('/')[-1].split('_')[1].split('.')[0])
            
        for i in range(len(realB_train_files)):
            data = np.load(realB_train_files[i])
            realB_train[i,:] = data
            realB_train_age[i] = map_age_dic[realB_train_files[i].split('/')[-1].split('.')[0][:-3]]
        
        for i in range(len(realA_test_files)):
            data = np.load(realA_test_files[i])
            realA_test[i,:] = data
            realA_test_age[i] = int(realA_test_files[i].split('/')[-1].split('_')[1].split('.')[0])
        
        for i in range(len(realB_test_files)):
            data = np.load(realB_test_files[i])
            realB_test[i,:] = data
            realB_test_age[i] = map_age_dic[realB_test_files[i].split('/')[-1].split('.')[0][:-3]]
        
    else:
        for i in range(len(realA_train_files)):
            data = np.load(realA_train_files[i])
            for j in range(n_roi):
                realA_train[i,j] = np.mean(data[roi_list[j]])
            realA_train_age[i] = int(realA_train_files[i].split('/')[-1].split('_')[1].split('.')[0])
            
        for i in range(len(realB_train_files)):
            data = np.load(realB_train_files[i])
            for j in range(n_roi):
                realB_train[i,j] = np.mean(data[roi_list[j]])
            realB_train_age = map_age_dic[realB_train_files[i].split('/')[-1].split('.')[0][:-3]]
        
        for i in range(len(realA_test_files)):
            data = np.load(realA_test_files[i])
            for j in range(n_roi):
                realA_test[i,j] = np.mean(data[roi_list[j]])
            realA_test_age[i] = int(realA_test_files[i].split('/')[-1].split('_')[1].split('.')[0])
        
        for i in range(len(realB_test_files)):
            data = np.load(realB_test_files[i])
            for j in range(n_roi):
                realB_test[i,j] = np.mean(data[roi_list[j]])
            realB_test_age = map_age_dic[realB_test_files[i].split('/')[-1].split('.')[0][:-3]]
            
    return  realA_train, realB_train, realA_test, realB_test, \
            realA_train_age, realB_train_age, realA_test_age, realB_test_age
        