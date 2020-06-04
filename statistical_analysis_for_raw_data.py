#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 22:06:49 2019

@author: fenqiang
"""

import matplotlib.pyplot as plt
import numpy as np
import glob
import csv
from sklearn.svm import SVR
from sklearn import svm
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.datasets import make_regression
#from sklearn import preprocessing
from sklearn.model_selection import ShuffleSplit, cross_val_score, GridSearchCV, KFold
#from sklearn.pipeline import make_pipeline
#from sklearn.linear_model import LinearRegression
#from sklearn.gaussian_process import GaussianProcessRegressor

from utils_for_ML import Stratified_split_files, get_data_for_raw, Stratified_split, Stratified_split_cv, get_data
from Common.vtk_io import read_vtk

""" parameters """
age = 720  # discard files larger than this age
map_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/journal/data/M0*lh.*.npy'))
bcp_files = sorted(glob.glob('/media/fenqiang/DATA/unc/Data/harmonization/journal/data/*BCP*lh.*.npy'))

""" preprocessing """
map_age_data = csv.reader(open('/media/fenqiang/DATA/unc/Data/harmonization/MapDayInfo.csv', 'r'))
map_age_dic = {}
for row in map_age_data:
    map_age_dic[row[0]] = float(row[1])

map_files = [x for x in map_files if map_age_dic[x.split('/')[-1].split('.')[0][:-3]] <= age ]
bcp_files = [x for x in bcp_files if float(x.split('/')[-1].split('_')[1].split('.')[0]) <= age ]

bcp_age = [float(x.split('/')[-1].split('_')[1].split('.')[0]) for x in bcp_files]
map_age = [map_age_dic[x.split('/')[-1].split('.')[0][:-3]] for x in map_files]


""" prepare data """
realA_train_files, realA_test_files = Stratified_split_files(bcp_files, bcp_age, 0.3)
realB_train_files, realB_test_files = Stratified_split_files(map_files, map_age, 0.3)

data_list = get_data_for_raw(realA_train_files,realB_train_files,realA_test_files,realB_test_files, n_roi=40962)

realA_train = data_list[0]
realB_train = data_list[1]
realA_test = data_list[2]
realB_test = data_list[3]

realA_train_age = data_list[4]
realB_train_age = data_list[5]
realA_test_age = data_list[6]
realB_test_age = data_list[7]
    
realA = np.concatenate((realA_train, realA_test), 0)
realB = np.concatenate((realB_train, realB_test), 0)
realA_age = np.concatenate((realA_train_age, realA_test_age), 0)
realB_age = np.concatenate((realB_train_age, realB_test_age), 0)

#%%
realA_mean = np.mean(realA_train, 0)
np.save('/home/fenqiang/harmonization/realA_mean.npy', realA_mean)
realA_std = np.std(realA_train, 0)
np.save('/home/fenqiang/harmonization/realA_std.npy', realA_std)
realB_mean = np.mean(realB_train, 0)
np.save('/home/fenqiang/harmonization/realB_mean.npy', realB_mean)
realB_std = np.std(realB_train, 0)
np.save('/home/fenqiang/harmonization/realB_std.npy', realB_std)

#%% 
""" age prediction """

""" dimension reduction """

data = np.concatenate((realA_train, realA_test, realB_train, realB_test), axis=0)
data = PCA(n_components=10).fit_transform(data)

realA_train = data[0:len(realA_train), :]
realA_test = data[len(realA_train): len(realA_train) + len(realA_test), :]
realB_train = data[len(realA_train) + len(realA_test): len(realA_train) + len(realA_test) + len(realB_train),:]
realB_test = data[len(realA_train) + len(realA_test) + len(realB_train): len(realA_train) + len(realA_test) + len(realB_train) + len(realB_test), :]


#%% assign train data and test data
X_train = realA_train
X_test = realA_test
y_train = realA_train_age
y_test = realA_test_age

#%% svr
clf = svm.SVR()
clf.fit(X_train, y_train) 
mae = np.abs(y_train - clf.predict(X_train))
print("train_mae, std: ", mae.mean(), mae.std())
mae = np.abs(y_test - clf.predict(X_test))
print("test_mae, std: ", mae.mean(), mae.std())    

#%%  trained on A, apply to B and harmonized(B)
data = np.concatenate((realA_train, realA_test), 0)
target = np.concatenate((realA_train_age, realA_test_age), 0)
X_train, X_test, y_train, y_test = Stratified_split(data, target, test_size=0.3)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  



#%% grid search for age prediction + svr

C_range = np.logspace(1, 6, 10)
gamma_range = np.logspace(-6, 0, 10)
param_grid = dict(gamma=gamma_range, C=C_range, kernel=['rbf'])
cv = Stratified_split_cv(n_splits=20, test_size=0.5, target=y_train)
grid = GridSearchCV(SVR(), param_grid=param_grid, cv=cv, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

clf = grid.best_estimator_

mae = np.abs(y_train - clf.predict(X_train))
print("train_mae, std: ", mae.mean(), mae.std())
mae = np.abs(y_test - clf.predict(X_test))
print("test_mae, std: ", mae.mean(), mae.std())    

 
#X = np.arange(10)
#for train_index, test_index in cv.split(X_train):
#     print("%s %s" % (train_index, test_index))


#%% random forest

cv = Stratified_split_cv(n_splits=4, test_size=0.4, target=y_train)
max_depth = np.linspace(150, 250, 5).astype(np.int32)
n_estimators = np.linspace(150, 250, 5).astype(np.int32)
param_grid = dict(max_depth=max_depth, n_estimators=n_estimators)
grid = GridSearchCV(RandomForestRegressor(), param_grid=param_grid, cv=cv, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

clf = grid.best_estimator_

mae = np.abs(y_train - clf.predict(X_train))
print("train_mae, std: ", mae.mean(), mae.std())
mae = np.abs(y_test - clf.predict(X_test))
print("test_mae, std: ", mae.mean(), mae.std())    


#%% apply to realB

realB_train = scaler.transform(realB_train)
realB_test = scaler.transform(realB_test)
mae = np.abs(realB_train_age - clf.predict(realB_train))
print("train_mae, std: ", mae.mean(), mae.std())
mae = np.abs(realB_test_age - clf.predict(realB_test))
print("test_mae, std: ", mae.mean(), mae.std())  
mae = np.concatenate((np.abs(realB_train_age - clf.predict(realB_train)),  np.abs(realB_test_age - clf.predict(realB_test))), 0)
print("total mean, std: ", mae.mean(), mae.std())    

 
#apply to fakeA
fakeA_train = scaler.transform(fakeA_train)
fakeA_test = scaler.transform(fakeA_test)
mae = np.abs(fakeA_train_age - clf.predict(fakeA_train))
print("train_mae, std: ", mae.mean(), mae.std())
mae = np.abs(fakeA_test_age - clf.predict(fakeA_test))
print("test_mae, std: ", mae.mean(), mae.std())    
mae = np.concatenate((np.abs(fakeA_train_age - clf.predict(fakeA_train)),  np.abs(fakeA_test_age - clf.predict(fakeA_test))), 0)
print("total mean, std: ", mae.mean(), mae.std()) 




#%% trained on B as a classification task, apply to A and harmonized(A)

data = np.concatenate((realB_train, realB_test), 0)
target = np.concatenate((realB_train_age, realB_test_age), 0)
target = (target / 30).astype(np.int32)
X_train, X_test, y_train, y_test = Stratified_split(data, target, test_size=0.3)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

#%% svc

C_range = np.logspace(0, 5, 10)
gamma_range = np.logspace(-6, -1, 10)
param_grid = dict(gamma=gamma_range, C=C_range, kernel=['rbf', 'linear'], decision_function_shape=['ovo', 'ovr'])
cv = Stratified_split_cv(n_splits=10, test_size=0.4, target=y_train)
grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv)
grid.fit(X_train, y_train)

print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

clf = grid.best_estimator_

acc = np.sum(y_train == clf.predict(X_train))/len(X_train)
print("train acc: ", acc)
acc = np.sum(y_test == clf.predict(X_test))/len(X_test)
print("test acc: ", acc)    

mae = np.abs(y_train.astype(np.float32) - clf.predict(X_train))
print("train_mae, std: ", mae.mean(), mae.std())
mae = np.abs(y_test.astype(np.float32) - clf.predict(X_test))
print("test_mae, std: ", mae.mean(), mae.std())  

#%% apply to realA
realA_train = scaler.transform(realA_train)
realA_test = scaler.transform(realA_test)
mae = np.abs(realA_train_age - clf.predict(realA_train))
print("train_mae, std: ", mae.mean(), mae.std())
mae = np.abs(realA_test_age - clf.predict(realA_test))
print("test_mae, std: ", mae.mean(), mae.std())  
mae = np.concatenate((np.abs(realA_train_age - clf.predict(realA_train)),  np.abs(realA_test_age - clf.predict(realA_test))), 0)
print("total mean, std: ", mae.mean(), mae.std())    

 
#apply to fakeB
fakeB_train = scaler.transform(fakeB_train)
fakeB_test = scaler.transform(fakeB_test)
mae = np.abs(fakeB_train_age - clf.predict(fakeB_train))
print("train_mae, std: ", mae.mean(), mae.std())
mae = np.abs(fakeB_test_age - clf.predict(fakeB_test))
print("test_mae, std: ", mae.mean(), mae.std())    
mae = np.concatenate((np.abs(fakeB_train_age - clf.predict(fakeB_train)),  np.abs(fakeB_test_age - clf.predict(fakeB_test))), 0)
print("total mean, std: ", mae.mean(), mae.std()) 


#%% GaussianProcessRegressor


gpr = GaussianProcessRegressor().fit(X, y)
gpr.score(X, y) 

 gpr.predict(X[:2,:], return_std=True) 




#%% linear regression model
clf = LinearRegression().fit(X_train, y_train)

mae = np.abs(y_train - clf.predict(X_train))
print("train_mae, std: ", mae.mean(), mae.std())
mae = np.abs(y_test - clf.predict(X_test))
print("test_mae, std: ", mae.mean(), mae.std())    

#apply to realA
mae = np.abs(realA_train_age - clf.predict(realA_train))
print("train_mae, std: ", mae.mean(), mae.std())
mae = np.abs(realA_test_age - clf.predict(realA_test))
print("test_mae, std: ", mae.mean(), mae.std())    
 
#apply to fakeB
mae = np.abs(fakeB_train_age - clf.predict(fakeB_train))
print("train_mae, std: ", mae.mean(), mae.std())
mae = np.abs(fakeB_test_age - clf.predict(fakeB_test))
print("test_mae, std: ", mae.mean(), mae.std())   


#%%

fakeB = data_combat[0:416,:]
realB = data_combat[416:,:]
fakeB_train = fakeB[0:333,:]
fakeB_test = fakeB[333:,:]
realB_train = realB[0:267,:]
realB_test = realB[267:,:]

age_train = np.concatenate((ageA[0:333], ageB[0:267]),0)
data_train = np.concatenate((realA_train, realB_train),0)
age_test = np.concatenate((ageA[333:], ageB[267:]),0)
data_test = np.concatenate((fakeB_test, realB_test),0)



data_train = np.concatenate((realA_train, realB_train),0)
data_test = np.concatenate((realA_test, realB_test),0)
age_train = np.concatenate((realA_train_age, realB_train_age),0)
age_test = np.concatenate((realA_test_age, realB_test_age),0)

data_train = realB_train
data_test = realB_test
age_train = realB_train_age
age_test = realB_test_age


def svr(x_train, x_test, y_train, y_test):
    
    clf = SVR(kernel='rbf', gamma=1, C=0.01, epsilon=0.1)
    clf.fit(x_train, y_train) 
    
    mae = np.abs(y_train - clf.predict(x_train))
    mae_mean, mae_std = mae.mean(), mae.std()
    print("train_mae, std: ", mae_mean, mae_std)
    
    mae = np.abs(y_test - clf.predict(x_test))
    mae_mean, mae_std = mae.mean(), mae.std()
    print("test_mae, std: ", mae_mean, mae_std)     

svr(data_train, data_test, age_train, age_test)
    


    
data = np.concatenate((realB_train, realB_test), 0)
data = PCA(n_components=2).fit_transform(data)
data = data[:,1]
svr(data[0:len(realA_train),:], data[len(realA_train):,:])





def rf(train, test):
    x_train = train
    y_train = ageB_train
    x_test = test
    y_test = ageB_test
    
    regr = RandomForestRegressor(max_depth=50, random_state=0, n_estimators=100)
    regr.fit(x_train, y_train)
    train_mae = np.abs(y_train - regr.predict(x_train)).mean()
    test_mae = np.abs(y_test - regr.predict(x_test)).mean()

    print("train_mae: ", train_mae)
    print("test_mae: ", test_mae)     

    
data = np.concatenate((realA_train, realA_test), 0)
data = PCA(n_components=50).fit_transform(data)
svr(data[0:len(realA_train),:], data[len(realA_train):,:])
svr(realB_train, realB_test)



# rf trained by B, directly apply to A
regr = RandomForestRegressor(max_depth=10, n_estimators=30)
regr.fit(realB_train, ageB_train)
mae = np.abs(ageA_test - regr.predict(realA_test))
mae_mean, mae_std = mae.mean(), mae.std()
print("train_mae, std: ", mae_mean, mae_std)
mae = np.abs(age - regr.predict(data))
mae_mean, mae_std = mae.mean(), mae.std()
print("test_mae, std: ", mae_mean, mae_std) 



train_mae = np.abs(ageA_train - regr.predict(realA_train)).mean()
print("apply to BCP train_mae: ", train_mae)
test_mae = np.abs(ageA_test - regr.predict(realA_test)).mean()
print("apply to BCP test_mae: ", test_mae)
