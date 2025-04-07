from data_provider.data_factory import data_provider
from data_provider.adv_data_factory import adv_data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
import torch.multiprocessing
from tqdm import tqdm

from .Visualize import *
torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

from models.Transformer_Adv_generator import Transformer_Adv_generator
from models.TS_attack import *

from affiliation.generics import *
from affiliation.metrics import *


def affiliation_res(labels, pred):
    events_gt = convert_vector_to_events(labels)
    events_pred = convert_vector_to_events(pred)
    Trange = infer_Trange(events_pred, events_gt)
    dict_out = pr_from_events(events_pred, events_gt, Trange)
    p = dict_out['Affiliation_Precision']
    r = dict_out['Affiliation_Recall']
    f1 = f1_func(p, r)
    print("Affiliation Precision: ", p, "Affiliation Recall: ", r, "F1: ", f1)
    return p, r, f1

warnings.filterwarnings('ignore')


class Exp_AdGenerator(Exp_Basic):
    def __init__(self, args):
        super(Exp_AdGenerator, self).__init__(args)
        # 在初始化的时候就调用了build_model，参考Exp_Basic的初始化

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        # 异常检测模型的定义

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def build_adv_model(self):
        adv_model_dict = {
            'Transformer_Adv':Transformer_Adv_generator,
            'Diff_Adv':None,
        }

        # 噪声生成器
        adv_model = adv_model_dict[self.args.attack_method](
            win_size=self.args.seq_len,
            enc_in=self.args.enc_in,
            epsilon=self.args.epsilon,
            d_model=128,
            d_ff=256,
            n_heads=8,
            e_layers=3,
            dropout=0.0
        )
        return adv_model
        
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _get_adv_data(self, flag):
        print("cross_attack:", self.args.cross_attack)
        data_set, data_loader = adv_data_provider(self.args, flag)
        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def test(self):
        # 测试原始模型, 记录score
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(f'./checkpoints/{self.args.data}_checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/'
        root_path = self.args.root_path + '/TCN_attack/'
        save_score_path = os.path.join(root_path, str(self.args.data) + '_score.npy')
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        # threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        new_threshold = np.percentile(train_energy, 100 - self.args.anomaly_ratio)
        # print("Threshold :", threshold)
        print("New Threshold :", new_threshold)
        np.save(folder_path + f'{self.args.data}_threshold.npy', new_threshold)
        np.save(save_score_path, test_energy)

        # (3) evaluation on the test set
        pred = (test_energy > new_threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)


        # (4) detection adjustment
        # p, r, f1 = affiliation_res(gt, pred)
        # return p,r,f1
        # PA adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

        
        return 

    def test_ori(self):
        # 仅在测试集上测试原始模型
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(f'./checkpoints/{self.args.data}_checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        # np.save(folder_path + 'threshold.npy', threshold)
        threshold = np.load(folder_path + f'{self.args.data}_threshold.npy')

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        # evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
            accuracy, precision,
            recall, f_score))

    def train_adv_gen(self):
        # 使用训练数据训练噪声生成器
        train_data, train_loader = self._get_data(flag='train')
        
        # 加载训练好的异常检测模型        
        print('loading trained anomaly detection model')
        self.model.load_state_dict(torch.load(os.path.join(f'./checkpoints/{self.args.data}_checkpoint.pth')))
        self.model.eval()

        
        self.adv_model = self.build_adv_model()
        self.adv_model.to(self.device)
        self.adv_model.train()

        # loss函数和优化器
        self.adv_criterion = nn.MSELoss()
        self.adv_optimizer = optim.Adam(self.adv_model.parameters(), lr=self.args.learning_rate)
        # self.adv_scheduler = lr_scheduler.StepLR(self.adv_optimizer, step_size=self.args.lr_step, gamma=0.1)
        

        # 训练噪声生成器

        # 一方面, 产生的噪声要足够小
        # 另一方面, 加噪数据要使得原始模型的重构loss尽可能大
        noise_loss_record = []
        rec_loss_record = []
        if self.args.attack_target == 'Precision':
            if self.args.data == 'MSL':
                lambda_noise = 0.7
                lambda_rec = 0.3
            elif self.args.data == 'SMAP':
                lambda_noise = 0.7
                lambda_rec = 0.3
            elif self.args.data == 'SWAT':
                lambda_noise = 0.7
                lambda_rec = 0.3
            elif self.args.data == 'WADI':
                lambda_noise = 0.7
                lambda_rec = 0.3             
            else:
                lambda_noise = 0.8
                lambda_rec = 0.2
        elif self.args.attack_target == 'Recall':
            lambda_noise = 0.8
            lambda_rec = 0.2
        
        for epoch in range(self.args.adv_epoch):
            for i, (batch_x, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader)):
                self.adv_optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device) 
                # # 训练模型
                # 白盒攻击 # 获得梯度信息
                # TS = batch_x.clone().detach().to(self.device)
                # TS.requires_grad = True
                # output = self.model(TS, None, None, None)
                # cost = torch.mean(self.adv_criterion(TS, output), dim=-1)
                # grad = torch.autograd.grad(
                #     cost, TS, retain_graph=False, create_graph=False
                # )[0] 
                # adversarial_noise = self.adv_model(batch_x+grad)
                # 黑盒攻击 # 不用梯度信息
                if i % 5 == 0:
                    batch_x = torch.rand_like(batch_x)
                adversarial_noise = self.adv_model(batch_x)
                new_batch_x = batch_x + adversarial_noise
                # 进行重构
                rec_batch_x = self.model(new_batch_x, None, None, None)
                # rec_loss:
                
                rec_loss = torch.mean(self.adv_criterion(rec_batch_x, new_batch_x), dim=-1)
                
                noise_loss = torch.mean(self.adv_criterion(new_batch_x, batch_x), dim=-1)
                noise_loss_record.append(noise_loss.item())
                rec_loss_record.append(rec_loss.item())

                if self.args.attack_target == 'Precision':
                    # 攻击Precision--让模型尽可能把正常数据识别为异常--增大重构误差
                    # 保持noise很小
                    loss = lambda_noise*noise_loss - lambda_rec*rec_loss  
                    # 优化器更新
                elif self.args.attack_target == 'Recall':
                    # 攻击Recall, 让模型尽可能把异常数据识别为正常 
                    # -- 减小针对异常的重构误差, 其余部分不关心
                    # 保持noise很小
                    # Not that easy!
                    loss = lambda_noise*noise_loss  + lambda_rec*rec_loss  
                loss.backward()
                self.adv_optimizer.step()
                # self.adv_scheduler.step()
        torch.save(self.adv_model.state_dict(), os.path.join(f'./checkpoints/{self.args.data}_{self.args.attack_target}_adv_checkpoint.pth'))

        # 绘制loss曲线
        plt.plot(noise_loss_record, label='noise_loss')
        plt.plot(rec_loss_record, label='rec_loss')
        plt.title(f'Adversarial Model Training Loss on {self.args.data}')
        plt.legend()
        plt.savefig(os.path.join(f'./checkpoints/{self.args.data}_{self.args.attack_target}_adv_loss.png'))
        plt.close()

    def gen_adv_test(self):
        # 在测试集合上产生对抗样本
        attack_method = self.args.attack_method
        attack_target = self.args.attack_target
        test_data, test_loader = self._get_data(flag='test')
        # train_data, train_loader = self._get_data(flag='train')
        
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(f'./checkpoints/{self.args.data}_checkpoint.pth')))

        attens_energy = []
        folder_path = './test_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # 获取参数, 并测试原始数据集上的性能
        threshold = np.load(folder_path + f'{self.args.data}_threshold.npy')
        print("threshold: ", threshold)
        # 加载数据、模型、参数
        #################

        # 定义攻击方法
        # if self.args.data == 'MSL':
        #         epsilon = 0.15
        # elif self.args.data == 'SWAT':
        #     epsilon = 0.02
        # elif self.args.data == 'WADI':
        #     epsilon = 0.02
        # elif self.args.data == 'SMAP':
        #     epsilon = 0.05
        # else:
        #     epsilon = 1/100
        # print("epsilon: ", epsilon)

        if attack_method == 'FGSM':
            attack = TS_FGSM(self.model, eps=self.args.epsilon)
        elif attack_method == 'Random':
            attack = TS_RANDOM(eps=self.args.epsilon)
        elif attack_method == 'Transformer_Adv':
            # 加载训练好的噪声生成器
            print("apply Transformer_Adv attack")
            self.adv_model = self.build_adv_model()
            self.adv_model.load_state_dict(torch.load(os.path.join(f'./checkpoints/{self.args.data}_{self.args.attack_target}_adv_checkpoint.pth')))
            self.adv_model.to(self.device)
            self.adv_model.eval()

        # # 执行攻击方法
        root_path = self.args.root_path + '/TCN_attack/'
        save_advdata_path = os.path.join(root_path, str(self.args.data) + '_test_adv_'+attack_method+'_'+attack_target+'.npy')
        save_label_path = os.path.join(root_path, str(self.args.data) + '_test_label_adv_'+attack_method+'_'+attack_target+'.npy')
        save_score_path = os.path.join(root_path, str(self.args.data) + '_score_adv_'+attack_method+'_'+attack_target+'.npy')
        # save_adv_path: ./dataset/SMAP_adv\SMAP_test_adv.npy
        self.adv_criterion = nn.MSELoss()
        test_labels = []
        attens_energy = []
        adv_inputs = []
        adv_labels = []
        
        for i, (batch_x, batch_y) in tqdm(enumerate(test_loader), total=len(test_loader)):
            batch_x = batch_x.float().to(self.device)
            # print("batch_x shape:", batch_x.shape) # [128, 96, 25]
            # print("batch_y shape:", batch_y.shape) # [128, 96]
            # # 调用攻击方法
            if attack_method == 'Transformer_Adv':
                if attack_target == 'Recall':
                    contains_one = torch.any(batch_y == 1)
                    if contains_one:
                        attack_input = self.adv_model(batch_x)+batch_x # 对抗样本
                    else:
                        attack_input = batch_x
                elif attack_target == 'Precision':
                    # # White Box
                    # TS = batch_x.clone().detach().to(self.device)
                    # TS.requires_grad = True
                    # output = self.model(TS, None, None, None)
                    # cost = torch.mean(self.adv_criterion(TS, output), dim=-1)
                    # grad = torch.autograd.grad(
                    #     cost, TS, retain_graph=False, create_graph=False
                    # )[0] # 获得梯度信息
                    # attack_input = self.adv_model(batch_x+grad)+batch_x # 对抗样本
                    # # Black Box
                    attack_input = self.adv_model(batch_x)+batch_x # 对抗样本                 
            else:
                # 传统方法
                if attack_target == 'Recall':
                    contains_one = torch.any(batch_y == 1)
                    if contains_one:
                        attack_input = attack(batch_x, batch_y, attack_target) # 对抗样本
                    else:
                        attack_input = batch_x
    
                elif attack_target == 'Precision':
                    attack_input = attack(batch_x, batch_y, attack_target) # 对抗样本
                
            # 模型处理
            # if i == 0:
            #     print("attack input:", attack_input)
            outputs = self.model(attack_input, None, None, None)
            # batch_x是原始数据, outputs是对加噪数据的重构
            # 如果是带噪声的数据进去, 就应该和带噪声的数据比对
            score = torch.mean(self.anomaly_criterion(attack_input, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            
            adv_inputs.append(attack_input.detach().cpu().numpy())
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)    

        # # 保存对抗样本
        ## 将adv_inputs变为一个length * channels的数据
        print("number of adv_inputs: ",len(adv_inputs)) # 3341
        print("shape of per adv_inputs: ",adv_inputs[0].shape) # [128, 96, 25]
        
        adv_inputs = np.concatenate(adv_inputs, axis=0) 
        print("adv_inputs.shape before reshape: ",adv_inputs.shape) #  [427522, 96, 25]
        save_labels = test_labels
        save_labels = np.concatenate(save_labels, axis=0)
        print("save_labels.shape: ",save_labels.shape) # [427522, 96]
        save_labels = save_labels.reshape(-1)
        print("save_labels.shape after reshape: ",save_labels.shape) # (41042112,)
        # 变为：numbers*win_size, channels
        adv_inputs = adv_inputs.reshape(-1, self.args.enc_in) 
        print("adv_inputs.shape: after reshape: ",adv_inputs.shape) #  (41042112, 25)
        
        ## # 保存adv_inputs
        np.save(save_advdata_path, adv_inputs)
        print("save_advdata_path: ", save_advdata_path)
        # ./all_datasets/MSL//TCN_attack/MSL_test_adv_Transformer_Adv_Precision.npy

        np.save(save_label_path, save_labels)
        np.save(save_score_path, test_energy)

        # new_threshold = np.percentile(test_energy, 100 - self.args.anomaly_ratio)
        # print("new_threshold :", new_threshold)
        # pred = (test_energy > new_threshold).astype(int)

        # # evaluation 
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)
        
        # print("pred:   ", pred.shape)
        # print("gt:     ", gt.shape)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print("pred: ", pred.shape)
        print("gt:   ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
        plt.plot(attens_energy, label='attens_energy')
        plt.savefig(os.path.join(f'./test_results/{self.args.data}_{attack_method}_{attack_target}_adv_score_gen.png'))
        
        return accuracy, precision, recall, f_score

    def test_adversarial(self):
        # 测试对抗样本的效果
        attack_method = self.args.attack_method
        attack_target = self.args.attack_target

        adv_test_data, adv_test_loader = self._get_adv_data(flag='test')
        # adv_test_data, adv_test_loader = self._get_data(flag='test')
        
        self.model.load_state_dict(torch.load(os.path.join(f'./checkpoints/{self.args.data}_checkpoint.pth')))

        folder_path = './test_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # 获取参数, 并测试原始数据集上的性能
        threshold = np.load(folder_path + f'{self.args.data}_threshold.npy')
        print("threshold: ", threshold)
        # 加载数据、模型、参数
        #################

        # 测试先前产生的对抗样本的性能
        attens_energy = []
        test_labels = []  
        
        for i, (batch_x, batch_y) in tqdm(enumerate(adv_test_loader), total=len(adv_test_loader)):
            if i == 0:
                print("attack input:", batch_x)

            batch_x = batch_x.float().to(self.device) # 原始输入数据
            outputs = self.model(batch_x, None, None, None)
            # batch_x是加噪数据, outputs是对加噪数据的重构
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            
            attens_energy.append(score)
            test_labels.append(batch_y.cpu().numpy())
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)    
        
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        
        pred = (test_energy > threshold).astype(int)
        gt = test_labels.astype(int)
        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)
        # detection adjustment
        gt, pred = adjustment(gt, pred)
        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))
        plt.plot(attens_energy, label='attens_energy')
        plt.savefig(os.path.join(f'./test_results/{self.args.data}_{attack_method}_{attack_target}_adv_score_gen_test.png'))
        return accuracy, precision, recall, f_score


   
    def visualize_adv(self):
        # 可视化测试集合对抗样本
        folder_path = './test_results/' 
        threshold = np.load(folder_path + f'{self.args.data}_threshold.npy')
        # 阈值

        attack_method = self.args.attack_method
        attack_target = self.args.attack_target
        root_path = self.args.root_path + '/TCN_attack/'
        ori_score_path = os.path.join(root_path, str(self.args.data) + '_score.npy')
        adv_score_path = os.path.join(root_path, str(self.args.data) + '_score_adv_'+attack_method+'_'+attack_target+'.npy')
        ori_score = np.load(ori_score_path)
        adv_score = np.load(adv_score_path)
        # AS

        test_data, test_loader = self._get_data(flag='test')
        adv_test_data, adv_test_loader = self._get_adv_data(flag='test')
        
        # 数据
        print("ori_score.shape: ", ori_score.shape)
        print("adv_score.shape: ", adv_score.shape)
        print("test_loader len: ", len(test_loader))
        print("adv_test_loader len: ", len(adv_test_loader))
        # 取出test_loader中随机一个batch的数据:
        
        
        # rand_id = np.random.randint(0, len(adv_test_data)-1)
        id_range = [120, 121]
        test_data_record = []
        adv_test_data_record = []
        for i in range(id_range[0], id_range[1]):
            test_data_0, test_label_0 = test_data[i]
            if type(test_data_0) == torch.Tensor:
                test_data_0 = test_data_0.numpy()
            test_data_0 = np.mean(test_data_0, axis=1)
            # test_data_0 = test_data_0[:, 2]
            test_data_record.append(test_data_0)
            adv_test_data_0, adv_test_label_0 = adv_test_data[i]
            adv_test_data_0 = np.mean(adv_test_data_0, axis=1)
            # adv_test_data_0 = adv_test_data_0[:, 2]
            adv_test_data_record.append(adv_test_data_0)

        test_data_record = np.array(test_data_record)
        test_data_record = np.concatenate(test_data_record, axis=0)
        adv_test_data_record = np.array(adv_test_data_record)
        adv_test_data_record = np.concatenate(adv_test_data_record, axis=0)
        print("test_data_record.shape: ", test_data_record.shape)
        print("adv_test_data_record.shape: ", adv_test_data_record.shape)
        
        
        visualize_local_sample(test_data_record, adv_test_data_record, self.args.data, attack_method, attack_target)

        # width = len(test_data_record)
        # original_score = ori_score[width*rand_id:width*(rand_id+1)]
        # adversarial_score = adv_score[width*rand_id:width*(rand_id+1)]
        # visualize_anomaly_score(original_score, adversarial_score, threshold, self.args.data, attack_method, attack_target)


        # 可视化T-SNE
        rand_id =  100
        num = 100
        test_data_record = []
        adv_test_data_record = []
        for i in range(num):
            test_data_0, test_label_0 = test_data[rand_id]
            adv_test_data_0, adv_test_label_0 = adv_test_data[rand_id]
            if type(test_data_0) == torch.Tensor:
                test_data_0 = test_data_0.numpy()
            test_data_record.append(test_data_0)
            adv_test_data_record.append(adv_test_data_0)
        test_data_record = np.array(test_data_record)
        adv_test_data_record = np.array(adv_test_data_record)
        # test_data_record = np.concatenate(test_data_record, axis=0)
        # adv_test_data_record = np.concatenate(adv_test_data_record, axis=0)
        print("test_data_record.shape: ", test_data_record.shape)
        print("adv_test_data_record.shape: ", adv_test_data_record.shape)
        visualize_tSNE(test_data_record, adv_test_data_record, self.args.data, attack_method, attack_target)



    @torch.no_grad()
    def evaluation_adv(self):
        # 评估测试集合对抗样本
        test_data, test_loader = self._get_data(flag='test')
        adv_test_data, adv_test_loader = self._get_adv_data(flag='test')
        
        # 计算平均差异
        L1_diff = []
        L2_diff = []
        Linf_diff = []
        Lmse_diff = []

        for i, ((batch_x, batch_y), (adv_batch_x, adv_batch_y)) in tqdm(enumerate(zip(test_loader, adv_test_loader))):
            batch_x = batch_x.float()
            adv_batch_x = adv_batch_x.float()
            delta_x = adv_batch_x - batch_x
            
            # L1 distance
            # diff1 = torch.sum(torch.abs(delta_x))
            # L1_diff.append(diff1.item())
            # L2 distance
            # diff2 = torch.sqrt(torch.sum((delta_x) ** 2))
            # L2_diff.append(diff2.item())
            # Linf distance
            # diff3 = torch.max(torch.abs(delta_x))
            # Linf_diff.append(diff3.item())


            # 计算每个样本的均方误差
            squared_noisematrix = delta_x**2
            # 计算每个 (100, 25) 样本所有位置上噪声的均方值
            mean_squares_per_batch = torch.mean(squared_noisematrix, axis=(1, 2))
            # 计算每个 (100, 25) 矩阵的 RMS 幅值
            rms_amplitudes_per_batch = torch.sqrt(mean_squares_per_batch)
            # 计算整个批次的均方误差
            overall_rms_amplitude = torch.mean(rms_amplitudes_per_batch)
            Lmse_diff.append(overall_rms_amplitude)
        
        # mean_L1_diff = np.mean(L1_diff)
        # mean_L2_diff = np.mean(L2_diff)
        # mean_Linf_diff = np.mean(Linf_diff)
        mean_Lmse_diff = np.mean(Lmse_diff)

        # print("mean L1_diff: ", mean_L1_diff)
        # print("mean L2_diff: ", mean_L2_diff)
        # print("mean Linf_diff: ", mean_Linf_diff)
        print("mean Lmse_diff: ", mean_Lmse_diff)

        plt.figure(figsize=(12, 4))
        plt.plot(Lmse_diff, label='Lmse_diff')
        # plt.ylim(0, 0.0001)
        plt.legend(loc='upper right')
        plt.title(f'MSE difference on {self.args.data}, mean MSE: {mean_Lmse_diff}, 99% percentile MSE: {np.percentile(Lmse_diff, 99)}')
        plt.savefig(f'./visualize_results/{self.args.data}_diff.png', dpi=300)
        plt.show()



    def gen_adv_train(self):
        # 在训练集上产生对抗样本
        attack_method = self.args.attack_method
        attack_target = self.args.attack_target
        train_data, train_loader = self._get_data(flag='train')
        
        # 导入生成器模型
        if attack_method == 'Transformer_Adv':
            # 加载训练好的噪声生成器
            print("apply Transformer_Adv attack")
            self.adv_model = self.build_adv_model()
            self.adv_model.load_state_dict(torch.load(os.path.join(f'./checkpoints/{self.args.data}_{self.args.attack_target}_adv_checkpoint.pth')))
            self.adv_model.to(self.device)
            self.adv_model.eval()
        

        # # 执行攻击方法
        root_path = self.args.root_path + '/TCN_attack/'
        save_advdata_path = os.path.join(root_path, str(self.args.data) + '_train_adv_'+attack_method+'_'+attack_target+'.npy')
        print("save_adv_train_data_path: ", save_advdata_path)
        self.adv_criterion = nn.MSELoss()
        test_labels = []
        attens_energy = []
        adv_inputs = []
        adv_labels = []
        for i, (batch_x, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch_x = batch_x.float().to(self.device)
            # print("batch_x shape:", batch_x.shape) # [128, 96, 25]
            # print("batch_y shape:", batch_y.shape) # [128, 96]
            # # 调用攻击方法
            if attack_method == 'Transformer_Adv':
                if attack_target == 'Recall':
                    contains_one = torch.any(batch_y == 1)
                    if contains_one:
                        attack_input = self.adv_model(batch_x)+batch_x # 对抗样本
                    else:
                        attack_input = batch_x
                elif attack_target == 'Precision':
                    # # White Box
                    # TS = batch_x.clone().detach().to(self.device)
                    # TS.requires_grad = True
                    # output = self.model(TS, None, None, None)
                    # cost = torch.mean(self.adv_criterion(TS, output), dim=-1)
                    # grad = torch.autograd.grad(
                    #     cost, TS, retain_graph=False, create_graph=False
                    # )[0] # 获得梯度信息
                    # attack_input = self.adv_model(batch_x+grad)+batch_x # 对抗样本
                    # # Black Box
                    attack_input = self.adv_model(batch_x)+batch_x # 对抗样本
            
            adv_inputs.append(attack_input.detach().cpu().numpy())
            
        
        # # 保存对抗样本
        print("number of adv_inputs: ",len(adv_inputs)) # 3341
        print("shape of per adv_inputs: ",adv_inputs[0].shape) # [128, 96, 25]
        
        adv_inputs = np.concatenate(adv_inputs, axis=0) 
        print("adv_inputs.shape before reshape: ",adv_inputs.shape) #  [427522, 96, 25]
        adv_inputs = adv_inputs.reshape(-1, self.args.enc_in) 
        print("adv_inputs.shape: after reshape: ",adv_inputs.shape) #  (41042112, 25)
        
        ## # 保存adv_inputs
        np.save(save_advdata_path, adv_inputs)
    


    def defence_train(self):
        # 使用噪声生成器, 在训练集上生成对抗样本, 对已训练好的模型进行对抗性训练
        # 测试对抗样本的效果
        attack_method = self.args.attack_method
        attack_target = self.args.attack_target
        train_data, train_loader = self._get_data(flag='train')
        adv_train_data, adv_train_loader = self._get_adv_data(flag='train')
        train_steps = len(adv_train_loader)
        
        
        print("len of adv_train_loader: ", len(adv_train_loader))
        print('loading model')
        self.model.load_state_dict(torch.load(os.path.join(f'./checkpoints/{self.args.data}_checkpoint.pth')))
        self.model.train()

        criterion = self._select_criterion()
        model_optim = self._select_optimizer()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)

        
        for epoch in range(self.args.train_epochs): 
            for i, ((batch_x, batch_y), (batch_ori_x, batch_ori_y)) in tqdm(enumerate(zip(adv_train_loader, train_loader)), total=min(len(adv_train_loader), len(train_loader))):

            # for i, (batch_x, batch_y) in tqdm(enumerate(adv_train_loader), total=len(adv_train_loader)):
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                if self.args.apply_adv_aug:
                    # 使用改进后的对抗样本训练
                    # batch_ori_x, batch_ori_y = next(iter(train_loader)) # 一一对应
                    if i== 0:
                        print("use improved defense train")
                
                    rand_train = torch.rand(1)[0] # 使用多种混合方式训练
                    if rand_train < 0.7:
                        # 70%的训练集使用对抗样本训练
                        outputs = self.model(batch_x, None, None, None)
                        loss = criterion(outputs, batch_x) # reconstruction loss
                        loss.backward()
                        model_optim.step()
                    elif rand_train < 0.9:
                        # 20%的训练集使用原数据训练
                        batch_ori_x = batch_ori_x.float().to(self.device)
                        outputs = self.model(batch_ori_x, None, None, None)
                        loss = criterion(outputs, batch_ori_x) 
                        loss.backward()
                        model_optim.step()   
                    else:
                        # 10%的训练集使用局部加噪数据训练
                        batch_ori_x = batch_ori_x.float().to(self.device)
                        rand_slice_start_idx = np.random.randint(1, 50)
                        rand_slice_length = np.random.randint(1, 50)
                        rand_slice_end_idx = rand_slice_start_idx + rand_slice_length
                        batch_x[:, rand_slice_start_idx:rand_slice_end_idx, :] = batch_ori_x[:, rand_slice_start_idx:rand_slice_end_idx, :] # 构造局部噪声片段
                        outputs = self.model(batch_x, None, None, None)
                        loss = criterion(outputs, batch_x) # reconstruction loss!!
                        loss.backward()
                        model_optim.step()
                else:
                    # 使用标准的对抗训练
                    if i==0:
                        print("use normal defense train")
                    outputs = self.model(batch_x, None, None, None)
                    loss = criterion(outputs, batch_x) # reconstruction loss
                    loss.backward()
                    model_optim.step()



            adjust_learning_rate(model_optim,scheduler, epoch + 1, self.args)

        torch.save(self.model.state_dict(), os.path.join(f'./checkpoints/{self.args.data}_{attack_method}_{attack_target}_aug_{self.args.apply_adv_aug}_defence_checkpoint.pth'))

        # # 绘制loss曲线
        # plt.plot(noise_loss_record, label='noise_loss')
        # plt.plot(rec_loss_record, label='rec_loss')
        # plt.legend()
        # plt.savefig(os.path.join(f'./checkpoints/{self.args.data}_{self.args.attack_target}_adv_loss.png'))
        # plt.close()

        return self.model



    def defence_test(self):
        attack_method = self.args.attack_method
        attack_target = self.args.attack_target
        
        self.model.load_state_dict(
            torch.load(
            os.path.join(f'./checkpoints/{self.args.data}_{attack_method}_{attack_target}_aug_{self.args.apply_adv_aug}_defence_checkpoint.pth'), weights_only=True)
        )
            
        self.model.eval()

        # 测试模型, 记录score
        ori_test_data, ori_test_loader = self._get_data(flag='test')
        ori_train_data, ori_train_loader = self._get_data(flag='train')
        train_data, train_loader = self._get_adv_data(flag='train') # 如果是在对抗训练集上
        
        attens_energy = []
        folder_path = './test_results/'
        root_path = self.args.root_path + '/TCN_attack/'
        save_score_path = os.path.join(root_path, str(self.args.data)+'_'+attack_method +'_'+attack_target+'_'+ '_defence_score.npy')
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y) in tqdm(enumerate(ori_train_loader), total=len(ori_train_loader)):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x, None, None, None)
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        ori_train_energy = np.array(attens_energy)

        # attens_energy = []
        # with torch.no_grad():
        #     for i, (batch_x, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        #         batch_x = batch_x.float().to(self.device)
        #         # reconstruction
        #         outputs = self.model(batch_x, None, None, None)
        #         # criterion
        #         score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
        #         score = score.detach().cpu().numpy()
        #         attens_energy.append(score)

        # attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        # train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in tqdm(enumerate(ori_test_loader), total=len(ori_test_loader)):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x, None, None, None)
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        # combined_energy = np.concatenate([ori_train_energy, test_energy], axis=0)
        combined_energy = ori_train_energy
        # combined_energy = np.concatenate([ori_train_energy, train_energy, test_energy], axis=0)
        
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print("Threshold :", threshold)
        np.save(folder_path + f'{attack_method}_{attack_target}_defence_threshold.npy', threshold)
        np.save(save_score_path, test_energy)

        # 计算在原始测试集上的性能
        # pred = (test_energy > threshold).astype(int)
        # test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        # test_labels = np.array(test_labels)
        # gt = test_labels.astype(int)
        # gt, pred = adjustment(gt, pred)
        # pred = np.array(pred)
        # gt = np.array(gt)
        # print("pred: ", pred.shape)
        # print("gt:   ", gt.shape)

        # accuracy = accuracy_score(gt, pred)
        # precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        # print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #     accuracy, precision,
        #     recall, f_score))


        # 测试模型, 记录score
        adv_test_data, adv_test_loader = self._get_adv_data(flag='test')
        # train_data, train_loader = self._get_data(flag='train')
        folder_path = './test_results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        # 计算在对抗数据集上的性能
        attens_energy = []
        test_labels = []  
        for i, (batch_x, batch_y) in tqdm(enumerate(adv_test_loader), total=len(adv_test_loader)):
            batch_x = batch_x.float().to(self.device) # 原始输入数据
            outputs = self.model(batch_x, None, None, None)
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y.cpu().numpy())
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)    
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        pred = (test_energy > threshold).astype(int)
        gt = test_labels.astype(int)
        gt, pred = adjustment(gt, pred)
        print("pred:   ", pred.shape)
        print("gt:     ", gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred,
                                                                              average='binary')
        print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
                accuracy, precision,
                recall, f_score))

        return accuracy, precision, recall, f_score

        
        

        


    
        

