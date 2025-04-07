import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings

warnings.filterwarnings('ignore')



class PSMSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = pd.read_csv(os.path.join(root_path, 'train.csv'))
        data = data.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(os.path.join(root_path, 'test.csv'))
        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = pd.read_csv(os.path.join(root_path, 'test_label.csv')).values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

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
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        if flag == 'train':
            self.step = 1
        else:
            self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "MSL_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "MSL_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "MSL_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)

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


class SMAPSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        if flag == 'train':
            self.step = 1
        else:
            self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMAP_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMAP_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        self.val = self.test
        self.test_labels = np.load(os.path.join(root_path, "SMAP_test_label.npy"))
        print("test:", self.test.shape)
        print("train:", self.train.shape)
        print("label:", self.test_labels.shape)

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


class SMDSegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        data = np.load(os.path.join(root_path, "SMD_train.npy"))
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(os.path.join(root_path, "SMD_test.npy"))
        self.test = self.scaler.transform(test_data)
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(os.path.join(root_path, "SMD_test_label.npy"))

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
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.mode = flag
        if flag == 'train':
            self.step = 1
        else:
            self.step = step
        self.win_size = win_size
        self.MinMaxScaler = MinMaxScaler()

        if os.path.exists(root_path + '/train.npy'):
            print("load train, test, labels npy!")
            self.train = np.load(root_path + '/train.npy')
            self.test = np.load(root_path + '/test.npy')
            self.test_labels = np.load(root_path + '/labels.npy')
        else:
            print("read files!")
            # 训练数据, normal
            # os.path.join(root_path, 'swat_train2.csv'
            normal = pd.read_csv(
                root_path + '/SWaT_Dataset_Normal_v1.csv', low_memory=False)
            normal = normal.drop(["Timestamp", "Normal/Attack"], axis=1)
            for i in list(normal):
                normal[i] = normal[i].apply(lambda x: str(x).replace(",", "."))
            normal = normal.astype(float)

            x = normal.values
            x_scaled = self.MinMaxScaler.fit_transform(x)
            # 先对数据进行拟合，再进行变换
            normal = pd.DataFrame(x_scaled)
            normal = normal.values

            # 测试数据, attack
            attack = pd.read_csv(
                root_path + "/SWaT_Dataset_Attack_v0.csv", sep=";", low_memory=False)
            labels = [float(label != 'Normal')
                    for label in attack["Normal/Attack"].values]
            # Normal 为0, attack为1(label != Normal 为True)
            attack = attack.drop(["Timestamp", "Normal/Attack"], axis=1)
            # Transform all columns into float64
            for i in list(attack):
                attack[i] = attack[i].apply(lambda x: str(x).replace(",", "."))
            attack = attack.astype(float)

            x = attack.values
            x_scaled = self.MinMaxScaler.transform(x)
            attack = pd.DataFrame(x_scaled)
            attack = attack.values
            self.train = normal
            self.test = attack
            self.val = self.test
            self.test_labels = labels
            np.save(root_path + '/train.npy', self.train)
            np.save(root_path + '/test.npy', self.test)
            np.save(root_path + '/labels.npy', self.test_labels)
            print("save files!")
            
        self.train = torch.tensor(self.train, dtype=torch.float32)
        self.test = torch.tensor(self.test, dtype=torch.float32)
        self.val = torch.tensor(self.test, dtype = torch.float32)
        self.test_labels = torch.tensor(self.test_labels, dtype=torch.bool)
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return self.train[index:index + self.win_size], self.test_labels[0:self.win_size]
        elif (self.mode == 'val'):
            return self.val[index:index + self.win_size], self.test_labels[0:self.win_size]
        elif (self.mode == 'test'):
            return self.test[index:index + self.win_size], \
                self.test_labels[index:index + self.win_size]
        else:
            return self.test[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size], \
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]


class WADISegLoader(Dataset):
    def __init__(self, root_path, win_size, step=1, flag="train"):
        self.step = step
        self.mode = flag
        self.win_size = win_size
        self.MinMaxScaler = MinMaxScaler()
        self.wadi_drop = ['2_LS_001_AL', '2_LS_002_AL',
                          '2_P_001_STATUS', '2_P_002_STATUS']
        
        if os.path.exists(root_path + '/WADI.A1_9 Oct 2017/train.npy'):
            print("load train, test, labels npy!")
            self.train = np.load(root_path + '/WADI.A1_9 Oct 2017/train.npy')
            self.test = np.load(root_path + '/WADI.A1_9 Oct 2017/test.npy')
            self.labels = np.load(root_path + '/WADI.A1_9 Oct 2017/labels.npy')
        else:
            print("read files!")
            # 数据读取
            self.train = pd.read_csv(os.path.join(
                root_path, 'WADI.A1_9 Oct 2017/WADI_14days.csv'), skiprows=1000, nrows=2e5)
            self.test = pd.read_csv(os.path.join(
                root_path, 'WADI.A1_9 Oct 2017/WADI_attackdata.csv'))
            ls = pd.read_csv(root_path + '/WADI_attacklabels.csv',
                            low_memory=False)
            # ls是attacklabels文档--标记了测试集中哪些时间段是异常数据，但没有标签

            # 预处理
            self.train.dropna(how='all', inplace=True)
            self.test.dropna(how='all', inplace=True)
            self.train.fillna(0, inplace=True)
            self.test.fillna(0, inplace=True)
            self.test['Time'] = self.test['Time'].astype(str)
            self.test['Time'] = pd.to_datetime(
                self.test['Date'] + ' ' + self.test['Time'])
            self.labels = self.test.copy(deep=True)
            # 获取test中的labels

            # 应该是要删除列，而不是数据归为0
            for i in self.test.columns.tolist()[3:]:
                del self.labels[i]

            # print(self.labels.head(10))
            # 数据格式为：Row       Date      Time

            for i in ['Start Time', 'End Time']:
                ls[i] = ls[i].astype(str)
                ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
            # print(ls.head(10)['Start Time'])
            # print(ls.head(10)['End Time'])
            # 数据格式为: 日期-时间
            self.labels.insert(loc=3, column='normal', value=0)
            for index, row in ls.iterrows():
                st, et = str(row['Start Time']), str(row['End Time'])
                self.labels.loc[(self.labels['Time'] >= st) & (
                    self.labels['Time'] <= et), 'normal'] = 1

            # WADI其实可以做多分类，但这里采用单分类处理
            # print(self.labels.head(10)['normal'])
            self.labels = np.array(self.labels['normal'])
            # print(np.sum(self.labels))
            # print(self.labels[0:20])
            self.train, self.test = self.convertNumpy(
                self.train), self.convertNumpy(self.test)
            # print(self.train.shape, self.test.shape, self.labels.shape)
            # print(sum(self.labels))
            # print(sum(self.labels)/len(self.labels))
            
            np.save(root_path + '/WADI.A1_9 Oct 2017/train.npy', self.train)
            np.save(root_path + '/WADI.A1_9 Oct 2017/test.npy', self.test)
            np.save(root_path + '/WADI.A1_9 Oct 2017/labels.npy', self.labels)
            print("save npy success!")
        self.val = self.test
        self.train = torch.tensor(self.train, dtype=torch.float32)
        self.val = torch.tensor(self.val, dtype = torch.float32)
        self.test = torch.tensor(self.test, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.bool)
        
        channels_sum = torch.abs(self.test).sum((0))
        # print(len(channels_sum))
        # print(channels_sum)
        non_zero_channels = (channels_sum != 0)
        # print(non_zero_channels)
        # print(self.test.shape)
        self.test = self.test[:, non_zero_channels]
        self.train = self.train[:, non_zero_channels]
        print(self.test.shape)
        
        
        

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            print("test mode length:")
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            # thre
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        if self.mode == "train":
            return self.train[index:index + self.win_size], self.labels[0:self.win_size]
        elif (self.mode == 'val'):
            return self.val[index:index + self.win_size], self.labels[0:self.win_size]
        elif (self.mode == 'test'):
            return self.test[index:index + self.win_size], \
                self.labels[index:index + self.win_size]
        else:
            # thre
            return self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size], \
                self.labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]

    def convertNumpy(self, df):
        # x = df[df.columns[3:]].values[::10, :]
        x = df[df.columns[3:]].values
        # print(x.ptp(0).shape)
        return (x - x.min(0)) / (x.ptp(0) + 1e-4)

