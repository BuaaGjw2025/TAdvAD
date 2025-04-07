import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
import pickletools
from torch.utils.data import DataLoader
import torch.nn.functional as F
from collections import defaultdict
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 设置字体为times:
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']

# 设置图片大小为10x5:
# 针对局部样本：
# 1.绘制正常样本与对抗样本时序图
def visualize_local_sample(original_series, adversarial_series, dataset, attack_method, attack_target):
    # 输入应当是一维度numpy序列(seq_len)
    # 绘制时序图
    # 设置字号为14：
    plt.rcParams['font.size'] = 14
    plt.figure(figsize=(10, 5))
    plt.plot(original_series, linewidth=2, alpha=0.8,color='dodgerblue',label='Normal Sample')
    plt.plot(adversarial_series, linewidth=2, alpha=0.8,color='orangered',label='Adversarial Sample')
    max1 = np.max(original_series)
    max2 = np.max(adversarial_series)
    min1 = np.min(original_series)
    min2 = np.min(adversarial_series)
    # 计算MSE, 并放在图像上
    # # print((original_series - adversarial_series) ** 2)
    # mse = np.mean((original_series - adversarial_series) ** 2)
    # print("mse:", mse)
    # plt.title(f'{dataset} {attack_method} {attack_target} Local Sample Visualization\nMSE: {mse:.4f}')
    # plt.title(f'{dataset} {attack_method} {attack_target} ')
    plt.xlabel('Time')
    plt.ylabel('Value')
    if dataset == 'SWAT':
        plt.ylim(min(min1, min2)/1.02, max(max1, max2)*1.02)
    elif dataset == 'WADI':
        plt.ylim(min(min1, min2)/1.1, max(max1, max2)*1.1)
    # plt.ylim(0.18,0.28)
    
    plt.legend(loc='upper right')
    plt.savefig(f'./visualize_results/{dataset}_{attack_method}_{attack_target}_local_sample.svg', format='svg', dpi=500)
    plt.savefig(f'./visualize_results/{dataset}_{attack_method}_{attack_target}_local_sample.png',dpi=500)
    plt.show()


# 1.1.绘制正常样本与对抗样本t-SNE分布图


# 2.绘制模型输出anomaly score分布对比, 模型预测结果对比
def visualize_anomaly_score(original_score, adversarial_score, threshold, dataset, attack_method, attack_target):
    # 绘制anomaly score分布对比
    # 输入score应当是一维度numpy序列(seq_len), threshod是一个数值
    # 另外将预测结果绘制在子图2中
    ori_res = (original_score > threshold).astype(int)
    adv_res = (adversarial_score > threshold).astype(int)
    # plt.figure(figsize=(10, 5))
    plt.subplot(2, 2, 1)
    plt.plot(original_score, label='Original Score')
    # plt.plot(adversarial_score, label='Adversarial Score')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.ylim(0, threshold*3)
    plt.legend(loc='upper right')   
    plt.title('Anomaly Score Distribution')
    plt.subplot(2, 2, 2)
    plt.plot(ori_res,label='Original Prediction')
    # 只显示0和1两个y轴刻度
    plt.yticks([0, 1])
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='upper right')
    plt.title('Prediction Result')
    
    plt.subplot(2, 2, 3)
    # plt.plot(original_score, label='Original Score')
    plt.plot(adversarial_score, label='Adversarial Score')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    # plt.ylim(0, threshold*3)
    plt.legend(loc='upper right')
    plt.subplot(2, 2, 4)
    # plt.plot(ori_res, marker='o',label='Original Prediction')
    plt.plot(adv_res,label='Adversarial Prediction')
    plt.yticks([0, 1])
    plt.ylim(-0.1, 1.1)
    plt.legend(loc='upper right')
    plt.savefig(f'./visualize_results/AS_prediction_{dataset}_{attack_method}_{attack_target}.png', dpi=500)
    print("save fig 2!")
    plt.show()
    
def visualize_anomaly_score2(original_score, adversarial_score, threshold):
    # 绘制anomaly score分布对比
    # 输入score应当是一维度numpy序列(seq_len), threshod是一个数值
    # 另外将预测结果绘制在子图2中
    ori_res = (original_score > threshold).astype(int)
    adv_res = (adversarial_score > threshold).astype(int)
    # plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(original_score, label='Original Score')
    plt.plot(adversarial_score, label='Adversarial Score')
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.ylim(0, threshold*3)
    plt.legend()
    plt.title('Anomaly Score Distribution')
    plt.subplot(1, 2, 2)
    plt.plot(ori_res, marker='o',label='Original Prediction')
    plt.plot(adv_res, marker='o',label='Adversarial Prediction')
    plt.legend()
    plt.title('Prediction Result')
    plt.savefig('./visualize_results/anomaly_score_prediction2.png', dpi=500)
    print("save fig 2!")
    plt.show()
    

def visualize_tSNE(data1, data2, dataset,  attack_method, attack_target):
    '''
        data1:原始数据(batch_size, window_size, channel_size)
        data2:对抗样本(batch_size, window_size, channel_size)
    '''
    plt.rcParams['font.size'] = 16
    print("data1.shape:", data1.shape)
    print("data2.shape:", data2.shape)
    # 对数值进行归一化
    # scaler = StandardScaler()
    # X_mean1 = np.mean(data1, axis=0)
    # X_mean2 = np.mean(data2, axis=0)
    # 将data1(101, 105, 127)展开为(101*105, 127)
    # 将data2(101, 105, 9)展开为(101*105, 9)
    # 
    X_mean1 = np.reshape(data1, (data1.shape[0]*data1.shape[1], data1.shape[2]))
    X_mean2 = np.reshape(data2, (data2.shape[0]*data2.shape[1], data2.shape[2]))

    print("X_mean1.shape:", X_mean1.shape)
    print("X_mean2.shape:", X_mean2.shape)
    # 随机采样1000个点
    # np.random.seed(1)
    # X_mean1 = np.random.permutation(X_mean1)
    # X_mean2 = np.random.permutation(X_mean2)
    X_mean1 = X_mean1[10:100]
    X_mean2 = X_mean2[10:100]
    
    tsne = TSNE(n_components=2, random_state=1)
    X1_tsne = tsne.fit_transform(X_mean1)
    X2_tsne = tsne.fit_transform(X_mean2)
    # 计算X1_tsne和X2_tsne的平均距离
    # dist = np.mean(np.sqrt(np.sum((X1_tsne - X2_tsne)**2, axis=1)))
    # print("dist:", dist)

    plt.figure(figsize=(8, 8))
    plt.scatter(X1_tsne[:, 0], X1_tsne[:, 1])
    plt.scatter(X2_tsne[:, 0], X2_tsne[:, 1])
    plt.legend(['Normal', 'Adversarial'], loc='upper right')
    plt.savefig(r'./visualize_results/t-SNE_2d_{}_{}_{}.png'.format(dataset, attack_method, attack_target), dpi=600, bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(8, 8))
    tsne3d = TSNE(n_components=3, random_state=0)
    X1_tsne = tsne3d.fit_transform(X_mean1)
    X2_tsne = tsne3d.fit_transform(X_mean2)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X1_tsne[:, 0], X1_tsne[:, 1], X1_tsne[:, 2], s=75, color='dodgerblue')
    ax.scatter(X2_tsne[:, 0], X2_tsne[:, 1], X2_tsne[:, 2], s=75, color='orangered')
    plt.legend(['Normal', 'Adversarial'], loc='upper right')
    plt.savefig(r'./visualize_results/t-SNE_3d_{}_{}_{}.pdf'.format(dataset, attack_method, attack_target), dpi=600, bbox_inches='tight')
    # plt.savefig(r'./visualize_results/t-SNE_3d_{}_{}_{}.png'.format(dataset, attack_method, attack_target), dpi=600, bbox_inches='tight')
    print("save fig 3!")


