# 使用Transformer编写一个噪声生成器
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import Tensor
import numpy as np
from torch.utils import data
from collections import OrderedDict
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import pickle
import random
from math import sqrt
import os
from einops import rearrange, reduce, repeat
import warnings
warnings.filterwarnings('ignore')


     

class Transformer_Adv_generator(nn.Module):
    # 输入: 数据，输出：待添加的噪声
    def __init__(self, 
                 win_size,
                 enc_in,
                 d_model=128,
                 n_heads=8,
                 e_layers=3,
                 d_ff=512,
                 dropout=0.0,
                 epsilon=0.02):
        # 0.5 for train, other for test
        super(Transformer_Adv_generator, self).__init__()
        self.win_size = win_size
        self.enc_in = enc_in
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.d_model = d_model
        self.activation = 'gelu'
        self.dropout = 0.0
        self.epsilon = epsilon
        
        # Embedding
        # self.embedding = LinearEmbedding(in_size=21490, out_size=self.d_model, d_model=self.d_model)
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AdvAttention(False, attention_dropout=self.dropout),
                                  self.d_model, self.n_heads),
                    self.d_model,
                    self.d_ff,
                    dropout=self.dropout,
                    activation=self.activation
                ) for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        # Decoder
        self.projection = nn.Linear(self.d_model, self.enc_in, bias=True)

    def forward(self, x):
        enc_out = self.embedding(x)
        enc_out = self.encoder(enc_out)
        dec_out = self.projection(enc_out)
        dec_out = torch.clamp(dec_out, min=-self.epsilon, max=self.epsilon)
        return dec_out 


class Encoder(nn.Module):
    # 编码器模块
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        for attn_layer in self.attn_layers:
            x  = attn_layer(x, attn_mask=attn_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x


class EncoderLayer(nn.Module):
    # 注意力计算层
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        # 前向过程
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class AttentionLayer(nn.Module):
    # 注意力计算过程
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries_x, keys_x, values_x, attn_mask):
        B, L, _ = queries_x.shape
        _, S, _ = keys_x.shape
        H = self.n_heads
        # print("input shape:", queries_x.shape)
        # print("self.query_projection:", self.query_projection)
        # print("heads:", self.n_heads)
        
        
        queries = self.query_projection(queries_x).view(B, L, H, -1)
        keys = self.key_projection(keys_x).view(B, S, H, -1)
        values = self.value_projection(values_x).view(B, S, H, -1)
        # print("Q, K, V shape:", queries.shape, keys.shape, values.shape)
        
        out  = self.inner_attention(
            queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out)


class AdvAttention(nn.Module):
    # 注意力计算底层
    def __init__(self, mask_flag=False, scale=None, attention_dropout=0.0):
        super(AdvAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        # 输入: qkv矩阵; 输出:V
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        # scale =1./sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                # default None
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
        attn = scale * scores
        attn = self.dropout(torch.softmax(attn, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", attn, values)
        # 输出V
        return V.contiguous()


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(
                mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x



class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


# class PatchEmbedding(nn.Module):
#     def __init__(self, patchsize, d_model, dropout=0.01):
#         # 对输入数据进行patch化，再进行embedding
#         super(PatchEmbedding, self).__init__()
#         self.patchsize = patchsize
#         self.embedding_patch_size = DataEmbedding(self.patchsize, d_model, dropout)
    
#     def forward(self, x):
#         x_patch_size = x
#         # Batch channel win_size
#         print("input shape:", x_patch_size.shape)
#         # [32, 512, 21490]
        
#         x_patch_size = rearrange(
#             x_patch_size, 'b m (n p) -> (b m) n p', p=self.patchsize)
#         print("rearranged shape:", x_patch_size.shape)
#         # [32*512, 2149, 10]
#         x_patch_size = self.embedding_patch_size(x_patch_size)
#         print("shape after embedding:", x_patch_size.shape)
#         # []

#         # series_patch_size = reduce(
#         #     series_patch_size, '(b reduce_b) l m n-> b l m n', 'mean', reduce_b=self.channel)
#         # print("shape after reduce:", shape:)

#         return x_patch_size

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.

        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
            
class LinearEmbedding(nn.Module):
    def __init__(self, in_size, out_size, d_model, dropout=0.01):
        # 对输入数据进行patch化，再进行embedding
        super(LinearEmbedding, self).__init__()
        self.Linear_layer = nn.Linear(in_size, out_size)
        self.BN = nn.Sequential(nn.BatchNorm1d(out_size),
                                 nn.ReLU())
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.input_length = in_size
    
    def forward(self, x):
        # 进行长度补齐处理
        if x.shape[-1] < self.input_length:
            padding_size = self.input_length - x.shape[-1]
            x = F.pad(x, (0, padding_size), "constant", 0)
        elif x.shape[-1] > self.input_length:
            x = x[:,:, :self.input_length]


        Linear_embedding = self.BN(self.Linear_layer(x))
        # print("shape of Linear_embedding:", Linear_embedding.shape)
        Pos_embedding = self.position_embedding(Linear_embedding)
        # print("Pos_embedding:", Pos_embedding.shape)
        return Linear_embedding + Pos_embedding



if __name__ == '__main__':
    win_size=100
    enc_in = 38
    d_model=512
    n_heads=8
    e_layers=3
    d_ff=512
    model = Transformer_Adv_generator(
        win_size=win_size,
        enc_in=enc_in,
        d_model=d_model,
        n_heads=n_heads,
        e_layers=e_layers,
        d_ff=d_ff,
        dropout=0.0,
        activation='gelu'
    )

    test_input = torch.randn(20, 100, 38)
    print(test_input.shape)
    test_output = model(test_input)
    print(test_output.shape)
    