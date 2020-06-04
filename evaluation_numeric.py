#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:01:40 2019

@author: fenqiang
"""


import glob
import numpy as np
import scipy.io as sio 
from scipy import stats
import xlwt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import csv

from utils_for_ML import cohensd, group, compute_cd_for_all_grp, get_pair_data, get_data, Stratified_split, Stratified_split_cv, get_npy_data


#%% prepare data
# unpaired data
data_list = get_npy_data('age', n_roi=40962)  ## 'age' or 'combat', 'lr'. 'lm'

#paired data
# data_list = get_pair_data('gan', vertex=True, epoch='300')  ## 'gan' or 'combat', 'lr'. 'lm'

realA_train = data_list[0]
realB_train = data_list[1]
fakeA_train = data_list[2]
fakeB_train = data_list[3]
realA_test = data_list[4]
realB_test = data_list[5]
fakeA_test = data_list[6]
fakeB_test = data_list[7]

realA_train_age = data_list[8]
realB_train_age = data_list[9]
fakeA_train_age =data_list[10]
fakeB_train_age =data_list[11]
realA_test_age = data_list[12]
realB_test_age = data_list[13]
fakeA_test_age = data_list[14]
fakeB_test_age = data_list[15]
    
realA = np.concatenate((realA_train, realA_test), 0)
realB = np.concatenate((realB_train, realB_test), 0)
fakeA = np.concatenate((fakeA_train, fakeA_test), 0)
fakeB = np.concatenate((fakeB_train, fakeB_test), 0)
realA_age = np.concatenate((realA_train_age, realA_test_age), 0)
realB_age = np.concatenate((realB_train_age, realB_test_age), 0)
fakeA_age = np.concatenate((fakeA_train_age, fakeA_test_age), 0)
fakeB_age = np.concatenate((fakeB_train_age, fakeB_test_age), 0)


#%%##################################
""" paired t-test """

realA_m = np.mean(realA, 0)
realB_m = np.mean(realB, 0)
fakeA_m = np.mean(fakeA, 0)
fakeB_m = np.mean(fakeB, 0)

realA_train_m = np.mean(realA_train, 0)
realB_train_m = np.mean(realB_train, 0)
fakeA_train_m = np.mean(fakeA_train, 0)
fakeB_train_m = np.mean(fakeB_train, 0)

realA_test_m = np.mean(realA_test, 0)
realB_test_m = np.mean(realB_test, 0)
fakeA_test_m = np.mean(fakeA_test, 0)
fakeB_test_m = np.mean(fakeB_test, 0)


stats.ttest_rel(fakeA_test_m, fakeB_test_m)


#%%

def write_to_xls(data, name):
    file = xlwt.Workbook(encoding = 'utf-8')
    sheet = file.add_sheet('sheet1')
    n_row, n_col = np.shape(data)
    for i in range(n_row):
        for j in range(n_col):
            sheet.write(i,j, data[i,j])
    file.save('/home/fenqiang/harmonization/figures/sheets/' + name)
    
    

data = np.concatenate((realA_test_m[:,np.newaxis], realB_test_m[:,np.newaxis], fakeA_test_m[:,np.newaxis], fakeB_test_m[:,np.newaxis]), 1)
write_to_xls(fakeB, 'fakeB_36ROI_feat.xls')


#%% compute vertex-wise/ roi-wise unpaired t-test
def vertex_ttest(data1, data2):
    """
        vertex-wise unpaird t-test between real and fake
    """    
    assert np.shape(data1)[1] == np.shape(data2)[1]
    p_value = np.zeros(np.shape(data1)[1])
    for i in range(len(p_value)):
        p_value[i] = stats.ttest_ind(data1[:,i], data2[:,i])[1]
    return p_value

p = vertex_ttest(fakeB, fakeA)
print(p.mean())

#%% correaltion coefficient
""" correaltion coefficient """

# compute correlation to observe how effective the pairing is
cc_before = ((realB_m - realB_m.mean()) * (realA_m - realA_m.mean())).mean() / realB_m.std() / realA_m.std()   # compute correlation between realA and realB
cc_after = ((realB_m - realB_m.mean()) * (fakeB_m - fakeB_m.mean())).mean() / realB_m.std() / fakeB_m.std()   # compute correlation between realB and fakeB
cc_after = ((realA_m - realA_m.mean()) * (fakeA_m - fakeA_m.mean())).mean() / realA_m.std() / fakeA_m.std()   # compute correlation between realB and fakeB


def compute_cc(data1,data2):
    """
    """
    return ((data1 - data1.mean()) * (data2 - data2.mean())).mean() / data1.std() / data2.std() 

def compute_cc_subject(data1, data2):
    assert np.shape(data1)[0] == np.shape(data2)[0]
    cc = np.zeros(np.shape(data1)[0])
    for i in range(np.shape(data1)[0]):
        cc[i] = compute_cc(data1[i,:], data2[i,:])
    return cc

def compute_cc_vertex(data1, data2):
    assert np.shape(data1)[1] == np.shape(data2)[1]
    cc = np.zeros(np.shape(data1)[1])
    for i in range(np.shape(data1)[1]):
        cc[i] = compute_cc(data1[:,i], data2[:,i])
    return cc
        
cc1 = compute_cc_subject(fakeA, realA)
print(cc1.mean())
cc1 = compute_cc_subject(fakeB, realB)
print(cc1.mean())


#%% psnr
""" psnr """
def compute_psnr(data1, data2, maxi):
    """
    """
    data1[np.argwhere(data1 > maxi)] = maxi
    data2[np.argwhere(data2 > maxi)] = maxi
    mse = np.mean((data1 - data2) ** 2)
    psnr = 20 * np.log10(maxi) - 10 * np.log10(mse)
    return psnr

def compute_psnr_subject(data1, data2):
    assert np.shape(data1)[0] == np.shape(data2)[0]
    psnr = np.zeros(np.shape(data1)[0])
    for i in range(np.shape(data1)[0]):
        psnr[i] = compute_psnr(data1[i,:], data2[i,:], 5.0)
    return psnr

psnr1 = compute_psnr_subject(fakeA_test, realA_test)
print(psnr1.mean())
print(psnr1.std())
psnr1 = compute_psnr_subject(fakeB, realB)
print(psnr1.mean())
print(psnr1.std())

#%%% vertex-wise mae
""" mae """
mae = np.abs(fakeA - realA)
print(mae.mean())
a = mae.mean(1)
print(a.std())  # std between subjects
mae = np.abs(fakeB - realB)
print(mae.mean())
a = mae.mean(1)
print(a.std())  # std between subjects

mae = np.abs(fakeB - fakeA)
print(mae.mean())
a = mae.mean(1)
print(a.std())  # std between subjects

#%% 
""" compute_distance_correlation """

def compute_cc(data1,data2):
    """
    """
    return ((data1 - data1.mean()) * (data2 - data2.mean())).mean() / data1.std() / data2.std()

def compute_distance_correlation(data1, data2):
    assert np.shape(data1)[0] == np.shape(data2)[0]
    dis1 = np.zeros(((np.shape(data1)[0]), (np.shape(data1)[0])))
    for i in range(len(dis1)):
        dis1[i,:] = np.sqrt(((data1 - data1[i,:]) ** 2).sum(1))
    dis2 = np.zeros(((np.shape(data2)[0]), (np.shape(data2)[0])))
    for i in range(len(dis2)):
        dis2[i,:] = np.sqrt(((data2 - data2[i,:]) ** 2).sum(1))
    
    return compute_cc(dis1.flatten(), dis2.flatten())
    
a = compute_distance_correlation(realA, fakeB)
print(a)
a = compute_distance_correlation(realB, fakeA)
print(a)


#%% compute cohen's d
""" compute cohen's d """

grp = group(realB, realB_age, [0,45],[46,135],[136,225],[226,315],[316,450],[451,750])
cd1 = compute_cd_for_all_grp(grp)

grp = group(fakeA, fakeA_age, [0,45],[46,135],[136,225],[226,315],[316,450],[451,750])
cd2 = compute_cd_for_all_grp(grp)

grp = group(np.concatenate((realA,realB), 0), np.concatenate((realA_age, realB_age), 0), [0,45],[46,135],[136,225],[226,315],[316,450],[451,630],[631,900],[901,1620],[1620,5000])
cd3 = compute_cd_for_all_grp(grp)

cd4 = (cd1 + cd2) / 2.0

a1 = np.abs(cd1-cd2)
c2 = np.zeros(0)
for i in range(5):
    c2 = np.append(c2, a1[i,i+1:6])

print(c2.mean())
print(c2.std())

#%% compute cohen's d
grp = group(realB, realB_age, [0,180],[360,451])
cd1 = compute_cd_for_all_grp(grp)

grp = group(fakeA, fakeA_age, [0,180],[360,451])
cd2 = compute_cd_for_all_grp(grp)

a1 = np.abs(cd1-cd2)
print(a1.mean())






#%% compute R^2 for linear regression
X_train, X_test, y_train, y_test = Stratified_split(fakeB, fakeB_age, test_size=0.3)
X_train, X_test, y_train, y_test = Stratified_split(fakeA, fakeA_age, test_size=0.3)
X_train, X_test, y_train, y_test = Stratified_split(np.concatenate((realA, fakeA),0), np.concatenate((realA_age, fakeA_age), 0), test_size=0.2)
X_train, X_test, y_train, y_test = Stratified_split(np.concatenate((realB, fakeB),0), np.concatenate((realB_age, fakeB_age), 0), test_size=0.2)
X_train, X_test, y_train, y_test = Stratified_split(np.concatenate((fakeA, fakeB),0), np.concatenate((fakeA_age, fakeB_age), 0), test_size=0.3)

data = np.concatenate((fakeA, realA),0)
target = np.concatenate((fakeA_age, realA_age), 0)

cv = Stratified_split_cv(n_splits=100, test_size=0.3, target=target)
gen = cv.split(data)
R2 = np.zeros((2, cv.get_n_splits()))
for i in range(cv.get_n_splits()):
    ind = next(gen) 
    ind_train = ind[0]
    ind_test = ind[1]
    X_train = data[ind_train]
    X_test = data[ind_test]
    y_train = target[ind_train] 
    y_test = target[ind_test]
   
    reg = LinearRegression().fit(np.median(X_train, 1, keepdims=True), y_train)
    R2[0,i] = reg.score(np.median(X_train, 1, keepdims=True), y_train)
    R2[1,i] = reg.score(np.median(X_test, 1, keepdims=True), y_test)
    
#    reg = LinearRegression().fit(X_train, y_train)
#    R2[0,i] = reg.score(X_train, y_train)
#    R2[1,i] = reg.score(X_test, y_test)
    
    
a = R2.mean(1)
b = R2.std(1)
c = np.concatenate((a,b),0)
c = np.reshape(c, [2,2])
c= np.transpose(c)
print(c)

#%% plot linear regression
x_train = np.median(X_train, 1, keepdims=True)
y_train = y_train

x_test = np.median(X_test, 1, keepdims=True)
y_test = y_test

plt.scatter(x_train, y_train, s=10, color='black')
plt.plot(x_train, reg.predict(x_train), color='blue', linewidth=3)
plt.scatter(x_test, y_test, s=10, color='red')
#plt.plot(x_test, LinearRegression().fit(x_train,y_train).predict(x_train), color='blue', linewidth=3)
plt.show()