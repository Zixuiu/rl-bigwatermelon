import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):
        self.max_size = max_size  # 缓冲区的最大大小
        self.ptr = 0  # 当前指针位置
        self.size = 0  # 缓冲区的当前大小

        self.state = np.zeros((max_size, 1, state_dim, state_dim))  # 状态缓冲区
        self.action = np.zeros((max_size, action_dim))  # 动作缓冲区
        self.next_state = np.zeros((max_size, 1, state_dim, state_dim))  # 下一个状态缓冲区
        self.reward = np.zeros((max_size, 1))  # 奖励缓冲区
        self.not_done = np.zeros((max_size, 1))  # 结束标志缓冲区

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU


    def add(self, state, action, next_state, reward, done):
        # 将数据添加到缓冲区中
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size  # 更新指针位置
        self.size = min(self.size + 1, self.max_size)  # 更新缓冲区大小


    def sample(self, batch_size):
        # 从缓冲区中随机采样一批数据
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
