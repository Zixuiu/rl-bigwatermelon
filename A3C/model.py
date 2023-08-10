import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def normalized_columns_initializer(weights, std=1.0):
    # 初始化权重，将其归一化
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out
#主要做了初始化和模型定义
def weights_init(m):
    # 初始化模型参数的权重
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
#He正态分布初始化
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])  # 输入通道数到输出通道数的总连接数
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]  # 输出通道数到输入通道数的总连接数
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)  # 均匀分布初始化权重
        m.bias.data.fill_(0)  # 初始化偏差为0
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]  # 输入单元数
        fan_out = weight_shape[0]  # 输出单元数
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)

class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)  # 输入通道数为num_inputs，输出通道数为32
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        # 定义LSTM层
        self.lstm = nn.LSTMCell(800, 256)  # 输入大小为800，隐藏大小为256

        num_outputs = 160
        # 定义策略和值函数的全连接层
        self.critic_linear = nn.Linear(256, 1)  # 输入大小为256，输出大小为1
        self.actor_linear = nn.Linear(256, num_outputs)  # 输入大小为256，输出大小为num_outputs

        self.apply(weights_init)  # 初始化模型参数的权重
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)  # 使用 normalized_columns_initializer 对策略层权重进行初始化
        self.actor_linear.bias.data.fill_(0)  # 初始化策略层权重的偏差为0
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)  # 使用 normalized_columns_initializer 对值函数层权重进行初始化
        self.critic_linear.bias.data.fill_(0)  # 初始化值函数层权重的偏差为0

        self.lstm.bias_ih.data.fill_(0)  # 初始化LSTM输入门的偏差为0
        self.lstm.bias_hh.data.fill_(0)  # 初始化LSTM隐藏层的偏差为0

        self.train()  # 设置为训练模式，激活批量归一化和 dropout

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        # 前向传播过程
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))

        x = x.view(-1, 800)  # 将 x 重新视为第一个维度形状为 800
        hx, cx = self.lstm(x, (hx, cx))  # 将 x 和 (hx, cx) 作为输入传递给 LSTM 单元
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)
