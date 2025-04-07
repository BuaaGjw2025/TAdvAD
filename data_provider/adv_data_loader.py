import torch
import os
import random
from PIL import Image
import collections
import numbers
import math
import pickle
import numpy as np
import pandas as pd
import glob
import re
from torch.utils.data import Dataset, DataLoader
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
warnings.filterwarnings('ignore')

class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="test", attack_method='FGSM',attack_target='Precision', cross_attack=False):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        if cross_attack:
            root_path = root_path + 'cross_AT_attack'
            print("loading from:", root_path)
        else:
            root_path = root_path + 'TCN_attack'
            print("loading from:", root_path)

        if os.path.exists(os.path.join(root_path, "SMAP_train_adv_"+attack_method+'_'+attack_target+".npy")):
            data = np.load(os.path.join(root_path, "SMAP_train_adv_"+attack_method+'_'+attack_target+".npy"))
            self.train = data
            print("SMAP adv train:", self.train.shape)
    
        if os.path.exists(os.path.join(root_path, "SMAP_test_adv_"+attack_method+'_'+attack_target+".npy")):
            test_data = np.load(os.path.join(root_path, "SMAP_test_adv_"+attack_method+'_'+attack_target+".npy"))
            self.test = test_data
            self.val = self.test
            print("SMAP adv test:", self.test.shape)
        if os.path.exists(os.path.join(root_path, "SMAP_test_label_adv_"+attack_method+'_'+attack_target+".npy")):
            self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label_adv_"+attack_method+'_'+attack_target+".npy"))
            print("SMAP adv test labels:", self.test_labels.shape)
            
    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="test", attack_method='FGSM',attack_target='Precision', cross_attack=False):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        # print("cross_attack:", cross_attack)
        if cross_attack:
            root_path = root_path + 'cross_AT_attack'
            print("loading from:", root_path)
        else:
            root_path = root_path + 'TCN_attack'
            print("loading from:", root_path)
        if os.path.exists(os.path.join(root_path, "MSL_train_adv_"+attack_method+'_'+attack_target+".npy")):
            data = np.load(os.path.join(root_path, "MSL_train_adv_"+attack_method+'_'+attack_target+".npy"))
            self.train = data
            print("MSL adv train:", self.train.shape)
    
        if os.path.exists(os.path.join(root_path, "MSL_test_adv_"+attack_method+'_'+attack_target+".npy")):
            test_data = np.load(os.path.join(root_path, "MSL_test_adv_"+attack_method+'_'+attack_target+".npy"))
            self.test = test_data
            self.val = self.test
            print("MSL adv test:", self.test.shape)
            
        if os.path.exists(os.path.join(root_path, "MSL_test_label_adv_"+attack_method+'_'+attack_target+".npy")):
            self.test_labels = np.load(os.path.join(root_path, "MSL_test_label_adv_"+attack_method+'_'+attack_target+".npy"))
    
    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class SWATSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="test", attack_method='FGSM', attack_target='Precision', cross_attack=False):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        if cross_attack:
            root_path = root_path + '/cross_AT_attack'
        else:
            root_path = root_path + '/TCN_attack'
            
        if os.path.exists(os.path.join(root_path, "SWAT_train_adv_"+attack_method+'_'+attack_target+".npy")):
            data = np.load(os.path.join(root_path, "SWAT_train_adv_"+attack_method+'_'+attack_target+".npy"))
            self.train = data
            print("SWAT adv train:", self.train.shape)
    
        if os.path.exists(os.path.join(root_path, "SWAT_test_adv_"+attack_method+'_'+attack_target+".npy")):
            test_data = np.load(os.path.join(root_path, "SWAT_test_adv_"+attack_method+'_'+attack_target+".npy"))
            self.test = test_data
            self.val = self.test
            print("SWAT adv test:", self.test.shape)
        if os.path.exists(os.path.join(root_path, "SWAT_test_label_adv_"+attack_method+'_'+attack_target+".npy")):
            self.test_labels = np.load(os.path.join(root_path, "SWAT_test_label_adv_"+attack_method+'_'+attack_target+".npy"))
    
    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])

class WADISegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="test", attack_method='FGSM',attack_target='Precision', cross_attack=False):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        if cross_attack:
            root_path = root_path + '/cross_AT_attack'
        else:
            root_path = root_path + '/TCN_attack'

        if os.path.exists(os.path.join(root_path, "WADI_train_adv_"+attack_method+'_'+attack_target+".npy")):
            data = np.load(os.path.join(root_path, "WADI_train_adv_"+attack_method+'_'+attack_target+".npy"))
            self.train = data
            print("WADI adv train:", self.train.shape)
    
        if os.path.exists(os.path.join(root_path, "WADI_test_adv_"+attack_method+'_'+attack_target+".npy")):
            test_data = np.load(os.path.join(root_path, "WADI_test_adv_"+attack_method+'_'+attack_target+".npy"))
            self.test = test_data
            self.val = self.test
            print("WADI adv test:", self.test.shape)
        if os.path.exists(os.path.join(root_path, "WADI_test_label_adv_"+attack_method+'_'+attack_target+".npy")):
            self.test_labels = np.load(os.path.join(root_path, "WADI_test_label_adv_"+attack_method+'_'+attack_target+".npy"))
            print("WADI adv test labels:", self.test_labels.shape)
        
    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])
