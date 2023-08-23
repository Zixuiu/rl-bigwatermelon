import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 这段代码定义了一个Actor-Critic模型，用于强化学习中的策略优化。模型包括卷积层、LSTM层和全连接层。
# 前向传播过程中，输入经过卷积层和LSTM层得到特征表示，然后分别通过值函数层和策略层得到值函数和动作概率。模型的参数通过初始化函数进行初始化。

def normalized_columns_initializer(weights, std=1.0):
    # 初始化权重，将其归一化
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out
#主要做了初始化和模型定义
def weights_init(m):  # 定义一个函数，名为weights_init，输入参数为m，是一个神经网络模型中的模块
    # 获取当前模块的类名
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:  # 如果模块的类名包含'Conv'（即该模块是卷积层）
        # 对卷积层的权重进行初始化
        weight_shape = list(m.weight.data.size())  # 获取权重的形状（大小）
        # 计算输入通道数到输出通道数的总连接数（即所谓的fan_in）
        fan_in = np.prod(weight_shape[1:4])
        # 计算输出通道数到输入通道数的总连接数（即所谓的fan_out）
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        # He正态分布的权重初始化公式中的系数计算
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        # 使用均匀分布对权重进行初始化，权重值在-w_bound和w_bound之间
        m.weight.data.uniform_(-w_bound, w_bound)
        # 初始化偏差为0
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:  # 如果模块的类名包含'Linear'（即该模块是全连接层）
        # 对全连接层的权重进行初始化
        weight_shape = list(m.weight.data.size())
        # 计算输入单元数（即权重矩阵的列数）
        fan_in = weight_shape[1]
        # 计算输出单元数（即权重矩阵的行数）
        fan_out = weight_shape[0]
        # He正态分布的权重初始化公式中的系数计算
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        # 使用均匀分布对权重进行初始化，权重值在-w_bound和w_bound之间
        m.weight.data.uniform_(-w_bound, w_bound)
        # 初始化偏差为0
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
def forward(self, inputs):  # 定义一个名为forward的函数，该函数是网络模型的前向传播过程，接收输入数据inputs
    inputs, (hx, cx) = inputs  # 从输入数据inputs中分离出两部分，一部分是网络输入，另一部分是RNN的初始状态，即隐藏状态(hx)和细胞状态(cx)
    # 下面是网络的前向传播过程，使用四个卷积层进行特征提取
    x = F.elu(self.conv1(inputs))  # 输入数据经过第一层卷积层conv1的变换，然后通过指数线性单元（ELU）激活函数进行非线性变换
    x = F.elu(self.conv2(x))  # 经过第二层卷积层conv2的变换，再通过ELU激活函数
    x = F.elu(self.conv3(x))  # 经过第三层卷积层conv3的变换，再通过ELU激活函数
    x = F.elu(self.conv4(x))  # 经过第四层卷积层conv4的变换，再通过ELU激活函数
    # 将经过多轮卷积后的特征图x重新调整为第一个维度为800的形状，便于后续的LSTM处理
    x = x.view(-1, 800)
    # 将处理后的特征图x和RNN的初始状态(hx, cx)作为输入传递给LSTM单元进行序列处理
    hx, cx = self.lstm(x, (hx, cx))
    # 取出LSTM的最后一个时刻的隐藏状态，作为DRNN的最终输出
    x = hx
    # 分别通过两个线性层对DRNN的输出进行处理，得到评判网络（critic）和行动网络（actor）的输出
    return self.critic_linear(x), self.actor_linear(x), (hx, cx)  # 返回评判网络的输出、行动网络的输出以及更新后的RNN状态
