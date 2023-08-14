# 引入copy模块，用于复制对象  
import copy  
  
# 引入numpy模块，用于进行基本的数学运算和矩阵操作  
import numpy as np  
  
# 引入torch模块，用于实现深度学习算法  
import torch  
  
# 引入torch.nn模块，该模块提供了构建神经网络的类和函数  
import torch.nn as nn  
  
# 引入torch.nn.functional模块，该模块提供了许多用于实现各种神经网络层的函数  
import torch.nn.functional as F  
  
# 获取设备，如果在可用，使用GPU（cuda），否则使用CPU（cpu）  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  


# 上面的代码实现了Twin Delayed Deep Deterministic Policy Gradients (TD3)算法，
# 包括Actor-Critic模型的构建、策略网络和值网络的前向传播过程、TD3算法的训练过程等。


# 定义Actor-Critic模型  
class ActorCritic(torch.nn.Module):  
    def __init__(self):  
        super(ActorCritic, self).__init__()  
        # 定义卷积层，用于处理输入的图像数据  
        # 输入通道数为1，输出通道数为32，卷积核大小为3，步长为2，填充为1  
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)  
        # 定义卷积层，用于进一步处理卷积1层的输出  
        # 输入通道数为32，输出通道数为32，卷积核大小为3，步长为2，填充为1  
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)  
  
        # 定义策略网络的全连接层，用于从图像特征映射到策略分布参数（例如，tanh均方根）  
        # 输入特征数为800（经过卷积层和全连接层的特征数），输出特征数为256  
        self.l1_a = nn.Linear(800, 256)  
        # 再通过一个全连接层，将特征数降至256  
        self.l2_a = nn.Linear(256, 256)  
        # 最后的全连接层将特征数降至1，输出的结果被tanh激活函数处理后作为策略分布的参数，最大行动值为160  
        self.l3_a = nn.Linear(256, 1)  
        self.max_action = 160  
  
        # 定义值网络的全连接层，用于从状态和动作计算值函数（即，状态-动作值函数）  
        # 输入特征数为800（状态特征）+ 1（动作特征），输出特征数为256  
        self.l1_c = nn.Linear(800 + 1, 256)  
        self.l2_c = nn.Linear(256, 256)  
        # 最后的全连接层将特征数降至1，输出的结果被tanh激活函数处理后作为值函数的估计值  
        self.l3_c = nn.Linear(256, 1)  
  
        # 在值网络中额外添加一层全连接层，用于计算策略梯度的目标值（即，目标策略值）  
        self.l4_c = nn.Linear(800 + 1, 256)  
        self.l5_c = nn.Linear(256, 256)  
        self.l6_c = nn.Linear(256, 1)  
  
    # 定义策略网络的前向传播过程（即，计算策略分布）  
    def pi(self, inputs):  
        # 通过卷积层处理输入的图像数据，然后通过全连接层处理得到的特征图  
        x = F.elu(self.conv1(inputs))  
        x = F.elu(self.conv2(x))  
        x = F.elu(self.conv3(x))  
        x = F.elu(self.conv4(x))  
  
        # 将得到的特征图展平为一维向量（即，将特征图中的

    # 定义一个值函数网络的前向传播过程  
    def v(self, state, action):  
        # 使用ELU激活函数进行卷积操作  
        x = F.elu(self.conv1(state))  
        x = F.elu(self.conv2(x))  
        x = F.elu(self.conv3(x))  
        x = F.elu(self.conv4(x))  
      
        # 将特征图展平为一维向量  
        x = x.view(-1, 800)  
      
        # 将状态和动作拼接在一起作为网络的输入  
        sa = torch.cat([x, action], 1)  
      
        # 通过一系列全连接层得到Q1  
        q1 = F.relu(self.l1_c(sa))  
        q1 = F.relu(self.l2_c(q1))  
        q1 = self.l3_c(q1)  
      
        # 通过另一系列全连接层得到Q2  
        q2 = F.relu(self.l4_c(sa))  
        q2 = F.relu(self.l5_c(q2))  
        q2 = self.l6_c(q2)  
        return q1, q2  # 返回Q1和Q2的值  
      
    # 定义一个只计算Q值的网络前向传播过程  
    def Q1(self, state, action):  
        x = F.elu(self.conv1(state))  
        x = F.elu(self.conv2(x))  
        x = F.elu(self.conv3(x))  
        x = F.elu(self.conv4(x))  
      
        x = x.view(-1, 800)  
      
        sa = torch.cat([x, action], 1)  
      
        q1 = F.relu(self.l1_c(sa))  
        q1 = F.relu(self.l2_c(q1))  
        q1 = self.l3_c(q1)  
        return q1  # 返回Q1的值  
  
# 定义TD3算法的类，包括算法参数和神经网络模型等  
class TD3(object):  
    def __init__(  # 初始化函数，定义算法参数和神经网络模型等  
        self,  
        state_dim,  # 状态维度  
        action_dim,  # 动作维度  
        max_action,  # 最大动作值，用于缩放动作空间  
        discount=0.99,  # 折扣因子，用于计算未来奖励的折扣权重  
        tau=0.005,  # 软更新系数，用于更新目标网络的参数  
        policy_noise=0.2,  # 策略噪声的标准差，用于探索策略空间  
        noise_clip=0.5,  # 策略噪声的上下界，防止噪声过大影响策略选择  
        policy_freq=2  # 策略更新频率，用于控制策略更新的频率  
    ):  
        self.actor_critic = ActorCritic().to(device)  # 创建ActorCritic模型，并移动到指定设备上  
        self.actor_critic_target = copy.deepcopy(self.actor_critic)  # 创建目标网络，并设置为当前网络的深拷贝  
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=3e-4)  # 创建优化器，用于优化当前网络的参数  
  
        self.max_action = max_action  # 保存最大动作值，用于缩放动作空间  
        self.discount = discount  # 保存折扣因子，用于计算未来奖励的折扣权重  
        self.tau = tau  # 保存软更新系数，用于更新目标网络的参数  
        self.policy_noise = policy_noise  # 保存策略噪声的标准差，用于探索策略空间  
        self.noise_clip = noise_clip  # 保存策略噪声的上下界，防止噪声过大影响策略选择  
        self.policy_freq = policy_freq  # 保存策略更新频率，用于控制策略更新的频率  
  
        self.total_it = 0  # 总迭代次数，用于记录算法的训练次数

    def select_action(self, state):
        # 根据策略网络选择动作
        state = torch.FloatTensor(state).to(device)
        return self.actor_critic.pi(state.unsqueeze(0)).cpu().data.numpy().flatten()

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        # 从回放缓存中采样数据
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # 根据策略选择动作并添加噪声
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_critic_target.pi(next_state) + noise
            ).clamp(-self.max_action / 2, self.max_action / 2) + (self.max_action / 2)

            # 计算目标Q值
            target_Q1, target_Q2 = self.actor_critic_target.v(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # 计算当前Q值
        current_Q1, current_Q2 = self.actor_critic.v(state, action)

        # 计算Critic的损失函数
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # 计算Actor的损失函数
        actor_loss = -self.actor_critic.Q1(state, self.actor_critic.pi(state)).mean()

        # 优化Actor网络
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # 更新目标模型
        for param, target_param in zip(self.actor_critic.parameters(), self.actor_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        # 保存模型
        torch.save(self.actor_critic.state_dict(), filename)
        torch.save(self.optimizer.state_dict(), filename + "_optimizer")

    def load(self, filename):
        # 加载模型
        self.actor_critic.load_state_dict(torch.load(filename))
        self.optimizer.load_state_dict(torch.load(filename + "_optimizer"))
        self.actor_critic_target = copy.deepcopy(self.actor_critic)
