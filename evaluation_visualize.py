#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 20:26:36 2019

@author: fenqiang
"""

import numpy as np
import scipy.io as sio 
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
from matplotlib import offsetbox
from time import time
import matplotlib.cm as cm
from neuroCombat import neuroCombat
import pandas as pd
from sklearn.manifold import TSNE

from utils_for_ML import cohensd, group, compute_cd_for_all_grp, get_data, Stratified_split, get_npy_data


#%% perpare data
data_list = get_npy_data('age', n_roi=40962)

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



#%%################################################
"""plot boxplot"""

#def plot_boxplot(A, B,C, A_age, B_age, C_age):
#
#    A_index = np.argsort(A_age)
#    B_index = np.argsort(B_age)
#    C_index = np.argsort(C_age)
#    
#    A_index_by_age = A[A_index,:]
#    B_index_by_age = B[B_index,:]
#    C_index_by_age = C[C_index,:]
#    
#    A_index_by_age = A_index_by_age[np.linspace(0, len(A_index_by_age)-1, num=100, dtype=np.int32),:]
#    B_index_by_age = B_index_by_age[np.linspace(0, len(B_index_by_age)-1, num=100, dtype=np.int32),:]
#    C_index_by_age = C_index_by_age[np.linspace(0, len(C_index_by_age)-1, num=100, dtype=np.int32),:]
#    
#    A = np.transpose(A_index_by_age) 
#    B = np.transpose(B_index_by_age)
#    C = np.transpose(C_index_by_age)
#    
#    
##    plt.plot([x1, x2], [y, y])
#    
#    fig, ax = plt.subplots()
#    
#    rect = matplotlib.patches.Rectangle(
#    xy=(0, 1.8),  
#    width=np.shape(A)[1]+np.shape(B)[1]+np.shape(C)[1]+1,  
#    height=0.6,
#    color='grey', alpha=0.5, ec='grey')
#    ax.add_patch(rect)
#    
#    c1 ="red"
##    plt.figure(figsize=(8,5))
#    bp1 = ax.boxplot(A, 0, '', widths=1,  patch_artist=True,
#                boxprops=dict(facecolor=c1, color=c1),
#                capprops=dict(color=c1),
#                whiskerprops=dict(color=c1),
#                flierprops=dict(color=c1, markeredgecolor=c1),
#                medianprops=dict(color=c1),)
#    
#    
#    c2 = "C0"
#    bp2 = ax.boxplot(B, 0, '', positions=np.arange(np.shape(A)[1]+1,np.shape(A)[1]+np.shape(B)[1]+1), widths=1, patch_artist=True,
#                    boxprops=dict(facecolor=c2, color=c2),
#                    capprops=dict(color=c2),
#                    whiskerprops=dict(color=c2),
#                    flierprops=dict(color=c2, markeredgecolor=c2),
#                    medianprops=dict(color=c2),)
#    
#    c3 = "green"
#    bp3 = ax.boxplot(C, 0, '', positions=np.arange(np.shape(A)[1]+np.shape(B)[1]+1, np.shape(A)[1]+np.shape(B)[1]+np.shape(C)[1]+1), widths=1, patch_artist=True,
#                    boxprops=dict(facecolor=c3, color=c3),
#                    capprops=dict(color=c3),
#                    whiskerprops=dict(color=c3),
#                    flierprops=dict(color=c3, markeredgecolor=c3),
#                    medianprops=dict(color=c3),)
#    
#    
#    
#    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['Site X', 'Site Y', 'Harmonized site Y'], loc='upper center',  ncol=3, fontsize=10)
#    
#    
#    ax.set_xlim(0,np.shape(A)[1]+np.shape(B)[1]+np.shape(C)[1]+1)
#    plt.xticks([])
#    plt.show()
#    
#
#
#plot_boxplot(realA, realB, fakeA, realA_age, realB_age, fakeA_age)







#%%################################################
#"""plot scatter of age and thickness"""
#plt.style.use('dark_background') # black background for slides! white for paper!
#
#realA_md = np.mean(realA, 1)
#realB_md = np.mean(realB, 1)
#fakeA_md = np.mean(fakeA, 1)
#fakeB_md = np.mean(fakeB, 1)
#
#fig = plt.figure(1)
#s1 = plt.scatter(realA_age, realA_md, marker='x', s=35, c=(1,0.3,0.3))
#s2 = plt.scatter(realB_age, realB_md, marker='v', s=35, c=(1,1,0))
#s3 = plt.scatter(fakeA_age, fakeA_md, marker='^', s=35, c=(0.3,1,0.3))
#
#plt.legend((s1, s2, s3), ['Site X (BCP)', 'Site Y (MAP)', 'Harmonized Site Y'], loc='upper right',  ncol=1, fontsize=22)
#
#plt.xlabel("Age (days)", fontsize=22)
#plt.ylabel("Cortical thickness (mm)", fontsize=22)
#
#ax = fig.add_subplot(111)
## We change the fontsize of minor ticks label 
#ax.tick_params(labelsize=20)
#ax.tick_params(labelsize=20)
#
#plt.show()


#%%##################################################################
""" scatter plot """

def cluster_pca(data):
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)
    return data

def cluster_tsne(data):
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    data = tsne.fit_transform(data)
    return data



data = np.concatenate((realA, realB, fakeA, fakeB), axis=0)
#age = np.concatenate((realA_age, realB_age, fakeA_age, fakeB_age))
#data = cluster_pca(data)
data = cluster_tsne(data)

realA = data[0:len(realA), :]
realB = data[len(realA): len(realA) + len(realB), :]
fakeA = data[len(realA) + len(realB):len(realA) + len(realB)+len(fakeA),:]
fakeB = data[len(realA) + len(realB)+len(fakeA):,:]

#plt.style.use('dark_background') # black background for slides! white for paper!

plt.figure(1)
# alpha control the transparent for age
a = plt.scatter(realA[:,0],realA[:,1], marker='x', s=35, c=(1,0.3,0.3))
b = plt.scatter(realB[:,0],realB[:,1], marker='v', s=35, c=(1,1,0))
c = plt.scatter(fakeA[:,0],fakeA[:,1], marker='^', s=35, c=(0.3,1,0.3))
#d = plt.scatter(fakeB[:,0],fakeB[:,1], marker='v', s=35, c='y')
plt.legend((a, b, c),
           ('Site X (BCP)', 'Site Y (MAP)', 'Harmonized Site Y'),
           scatterpoints=1,
           loc='upper right',
           ncol=1,
           fontsize=10)

#plot corresponding lines 
def draw_line(x, y):
    plt.plot(x, y, '--', color=(0.8,0.8,0.8), linewidth=0.6)
for i in range(len(fakeA)):
    A = fakeA[i,:]
    B = realB[i,:]
    x = [A[0], B[0]]
    y = [A[1], B[1]]
    draw_line(x, y)

#plot age beside the point
#for i, txt in enumerate(age):
#    plt.annotate(txt, (data[i,0], data[i,1]), fontsize=5)

#plt.xlim(-130, 180)
#plt.ylim(-40, 45)    
plt.xticks([])
plt.yticks([])
plt.show()




    
##plot 2 types of data in one figure
#data = np.concatenate((realA, realB), axis=0)
#pca = PCA(n_components=2)
#data = pca.fit_transform(data)
#
#realB = data[0: len(realA_files), :]
#fakeB = data[len(realA_files):,:]
#
#plt.figure(1)
#b = plt.scatter(realB[:,0],realB[:,1], marker='+', c='b')
#d = plt.scatter(fakeB[:,0],fakeB[:,1], marker='v', c='y')
#plt.legend((b, d),
#           ('real MAP', 'fake MAP'),
#           scatterpoints=1,
#           loc='lower right',
#           ncol=2,
#           fontsize=10)
#plt.show()



# t-sne for feature after pca
#data = TSNE(n_components=2).fit_transform(data)
#realA = data[0:len(realA), :]
#realB = data[len(realA): len(realA) + len(realB), :]
#fakeA = data[len(realA) + len(realB): len(realA) + len(realB) + len(fakeA),:]
#fakeB = data[len(realA) + len(realB) + len(fakeA):,:]
#
#plt.figure(2)
#a = plt.scatter(realA[:,0],realA[:,1], marker='x', c='r')
#b = plt.scatter(realB[:,0],realB[:,1], marker='+', c='b')
#c = plt.scatter(fakeA[:,0],fakeA[:,1], marker='^', c='g')
#d = plt.scatter(fakeB[:,0],fakeB[:,1], marker='v', c='y')
#plt.legend((a, b, c, d),
#           ('Site X', 'Site Y', 'Harmonized Y', 'Harmonized X'),
#           scatterpoints=1,
#           loc='lower right',
#           ncol=2,
#           fontsize=10)
#plt.xticks([])
#plt.yticks([])
#plt.show()

#
#
#
#
#X = data_gan
#y = batch.astype(np.long)
#
###----------------------------------------------------------------------
### Scale and visualize the embedding vectors
##def plot_embedding(X, title=None):
##    x_min, x_max = np.min(X, 0), np.max(X, 0)
##    X = (X - x_min) / (x_max - x_min)
##
##    plt.figure()
##    ax = plt.subplot(111)
##    for i in range(X.shape[0]):
##        plt.plot(X[i, 0], X[i, 1], legend[y[i]])
##
###    plt.xticks([]), plt.yticks([])
##    if title is not None:
##        plt.title(title)
#
##----------------------------------------------------------------------
## Scale and visualize the embedding vectors
#def plot_embedding(X, title=None):
#    x_min, x_max = np.min(X, 0), np.max(X, 0)
#    X = (X - x_min) / (x_max - x_min)
#
#    plt.figure()
#    ax = plt.subplot(111)
##    colors = cm.cool(np.linspace(0, 1, max(y)))
#    for i in range(X.shape[0]):
#        plt.text(X[i, 0], X[i, 1], 1 if i < 416 else 2,
#                 color=plt.cm.Set1(y[i] / 10.),
#                 fontdict={'weight': 'bold', 'size': 9})
#
##    if hasattr(offsetbox, 'AnnotationBbox'):
##        # only print thumbnails with matplotlib > 1.0
##        shown_images = np.array([[1., 1.]])  # just something big
##        for i in range(X.shape[0]):
##            dist = np.sum((X[i] - shown_images) ** 2, 1)
##            if np.min(dist) < 4e-3:
##                # don't show points that are too close
##                continue
##            shown_images = np.r_[shown_images, [X[i]]]
##            imagebox = offsetbox.AnnotationBbox(
##                offsetbox.OffsetImage(digits.images[i], cmap=plt.cm.gray_r),
##                X[i])
##            ax.add_artist(imagebox)
#    plt.xticks([]), plt.yticks([])
#    if title is not None:
#        plt.title(title)
#
##
##
###----------------------------------------------------------------------
### Random 2D projection using a random unitary matrix
##print("Computing random projection")
##rp = random_projection.SparseRandomProjection(n_components=2, random_state=42)
##X_projected = rp.fit_transform(X)
##plot_embedding(X_projected, "Random Projection of the digits")
##plt.show()
#
##----------------------------------------------------------------------
## Projection on to the first 2 principal components
#
#print("Computing PCA projection")
#t0 = time()
#X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
#plot_embedding(X_pca,
#               "Principal Components projection of the digits (time %.2fs)" %
#               (time() - t0))
#X_pca_2 = PCA(n_components=2).fit_transform(X)
#plot_embedding(X_pca_2,
#               "Principal Components projection of the digits (time %.2fs)" %
#               (time() - t0))
#
#
##----------------------------------------------------------------------
## Random Trees embedding of the digits dataset
#print("Computing Totally Random Trees embedding")
#hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
#                                       max_depth=5)
#t0 = time()
#X_transformed = hasher.fit_transform(X)
#pca = decomposition.TruncatedSVD(n_components=2)
#X_reduced = pca.fit_transform(X_transformed)
#
#plot_embedding(X_reduced,
#               "Random forest embedding of the digits (time %.2fs)" %
#               (time() - t0))
#
##----------------------------------------------------------------------
## Spectral embedding of the digits dataset
#print("Computing Spectral embedding")
#embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
#                                      eigen_solver="arpack")
#t0 = time()
#X_se = embedder.fit_transform(X)
#
#plot_embedding(X_se,
#               "Spectral embedding of the digits (time %.2fs)" %
#               (time() - t0))
#
##----------------------------------------------------------------------
## t-SNE embedding of the digits dataset
#print("Computing t-SNE embedding")
#tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
#t0 = time()
#X_tsne = tsne.fit_transform(X)
#
#plot_embedding(X_tsne,
#               "t-SNE embedding of the digits (time %.2fs)" %
#               (time() - t0))
#
#plt.show()
#
