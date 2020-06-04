#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:35:23 2019

@author: fenqiang
"""
import numpy as np
import SimpleITK as sitk
import os
import glob
#from utils import read_med_image, copy_sitk_information, getImageInformation_itk


def resample_image(input_name, output_name):
    cmd_resample = '''./ResampleImage 3 %s %s 1x1x1''' % (input_name, output_name)    
    os.system(cmd_resample)
   
    
def re_orientation_LPI(filename):
    '''
    Change origin and reorientation to RAI
    '''
   #origin_filename = filename.replace('.nii.gz', '-origin.hdr')
    lpi_filename = filename.replace('.nii.gz','-LPI.hdr')   
    cmd_LPI = '''./reorientation_RAI/build/orientLPI %s %s''' % (filename, lpi_filename)
    os.system(cmd_LPI)    
    return lpi_filename



root_folder = '/media/fenqiang/DATA/unc/Data/harmonization/intensity/intensity12'
subjects = sorted(glob.glob(root_folder + '/*'))

for i in range(len(subjects)):
    subject = subjects[i]
    ages = sorted(glob.glob(subject + '/*'))
    for j in range(len(ages)):
        root = ages[j]
        if not os.path.exists(root + '/' + root.split('/')[-2] + '_' + root.split('/')[-1] + '-label-111.hdr'):
    
            
            files = [root + '/' + root.split('/')[-2] + '_' + root.split('/')[-1] + '-T1-rmcere-n3-111-080808-h.hdr_preSub-CSF.nii.gz', \
            root + '/' + root.split('/')[-2] + '_' + root.split('/')[-1] + '-T1-rmcere-n3-111-080808-h.hdr_preSub-GM.nii.gz', \
            root + '/' + root.split('/')[-2] + '_' + root.split('/')[-1] + '-T1-rmcere-n3-111-080808-h.hdr_preSub-WM.nii.gz']
    
            # Set segresult to 0.8
            prob_CSF_WM_GM = 0
            for ind, filename in enumerate(files):    
                print(filename)
                img_itk = sitk.ReadImage(filename)
                img = sitk.GetArrayFromImage(img_itk)
                print (img.min(), img.max())
                prob_CSF_WM_GM += img
                img_itk.SetSpacing((0.8, 0.8, 0.8))
                output_filename = filename.replace('-T1-', '-')
                output_filename = output_filename.replace('.nii.gz', '-080808.nii.gz')
                print(output_filename)
                sitk.WriteImage(img_itk, output_filename)
                files[ind] = output_filename
                
            # write Background    
            prob_BG = 1.0 - prob_CSF_WM_GM
            prob_BG = prob_BG.astype(np.float32)
            prob_BG = sitk.GetImageFromArray(prob_BG)
            prob_BG.SetSpacing((0.8, 0.8, 0.8))
            BG_out08 = root + '/' + root.split('/')[-2] + '_' + root.split('/')[-1] + '-rmcere-n3-111-080808-h.hdr_preSub-BG-080808.nii.gz'
            sitk.WriteImage(prob_BG, BG_out08)
            
            #downsample to 1 1 1 
            files.insert(0, BG_out08)
            for file in files:
                up_out = file.replace('.nii.gz','-111.hdr')
                resample_image(file, up_out)
            
            #get tissue label image
            un_sampled_list =[]
            for ind, filename in enumerate(files):
                f_T1_out = filename.replace('.nii.gz','-111.hdr') 
                prob_i_itk =  sitk.ReadImage(f_T1_out)
                prob_i = sitk.GetArrayFromImage(prob_i_itk).astype(np.float32)
                prob_i = prob_i[None, ...]
                un_sampled_list.append(prob_i)
            
            un_sampled_list = np.vstack(un_sampled_list)
            unsampled = np.argmax(un_sampled_list, axis=0).astype(np.uint8)
            unsampled = sitk.GetImageFromArray(unsampled)
            unsampled.SetSpacing((1,1,1))
            filename_out = root + '/' + root.split('/')[-2] + '_' + root.split('/')[-1] + '-label-111.hdr'
            sitk.WriteImage(unsampled, filename_out)
                        