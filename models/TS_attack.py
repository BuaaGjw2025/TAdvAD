import torch
import torch.nn as nn


class TS_RANDOM(nn.Module):
    def __init__(self, eps=5 / 100):
        super(TS_RANDOM, self).__init__()
        self.eps = eps
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def forward(self, TS, labels, attack_target):
        TS = TS.clone().detach().to(self.device)
        random_data = torch.rand_like(TS)-0.5
        # 大于0的部分置为1, 小于0的部分置为-1:
        random_data[random_data > 0] = 1
        random_data[random_data < 0] = -1
        random_data = random_data * self.eps
        adv_TS = TS + random_data
        return adv_TS
    

class TS_FGSM(nn.Module):
    def __init__(self, model, eps=5 / 100, clamp_min=-10.00, clamp_max=10.00):
        super(TS_FGSM, self).__init__()
        self.model = model
        self.eps = eps
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        

    def forward(self, TS, labels, attack_target):
        TS = TS.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # loss = nn.CrossEntropyLoss() # loss需要重新定义个
        loss = nn.MSELoss()  # reduce=False
        # 当 reduce=True（默认值）时，损失函数会计算所有单个损失的平均值，并返回一个单一的标量值。这意味着损失是所有样本点的均方误差的平均值。
        # 当 reduce=False 时，损失函数不会对单个损失值进行平均或求和，而是返回一个与输入相同大小的张量，其中每个元素都是相应输入和目标之间差的平方。
        # # 针对重构结果 

        TS.requires_grad = True
        # outputs = self.model(TS) # 模型输出的不是outputs，而是一组
        output = self.model(TS, None, None, None)
        

        # Calculate loss
        # cost = loss(outputs, labels)
        cost = torch.mean(loss(TS, output), dim=-1) # 基于重构结果

        # Update adversarial TS
        grad = torch.autograd.grad(
            cost, TS, retain_graph=False, create_graph=False
        )[0] 
        # 求导, 输入对输出求导
        # torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)
        if attack_target == 'Recall':
            delta = self.eps * grad.sign()
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            adv_TS = TS - delta
        elif attack_target == 'Precision':
            delta = self.eps * grad.sign()
            # 原来没有下面这一行
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)
            adv_TS = TS + delta
        else:
            adv_TS = TS
        
        return adv_TS
    