import numpy as np  
import torch  


# 这段代码实现了一个重放缓冲区（Replay Buffer）类，用于存储和采样训练数据。


# 导入numpy库，用于进行数值计算和数组操作。导入torch库，用于构建和运行神经网络模型。  
  
class ReplayBuffer(object):  
    def __init__(self, state_dim, action_dim, max_size=int(1e5)):  
        # 初始化函数，定义了ReplayBuffer类的对象。  
          
        self.max_size = max_size  # 缓冲区的最大大小  
        # 设置缓冲区的最大容量。  
  
        self.ptr = 0  # 当前指针位置  
        # 初始化当前指针位置为0，用于后续数据添加时的索引。  
  
        self.size = 0  # 缓冲区的当前大小  
        # 初始化缓冲区的当前大小为0，用于记录当前缓冲区中的数据量。  
  
        self.state = np.zeros((max_size, 1, state_dim, state_dim))  # 状态缓冲区  
        # 使用numpy创建一个大小为(max_size, 1, state_dim, state_dim)的零矩阵，作为状态缓冲区。  
  
        self.action = np.zeros((max_size, action_dim))  # 动作缓冲区  
        # 使用numpy创建一个大小为(max_size, action_dim)的零矩阵，作为动作缓冲区。  
  
        self.next_state = np.zeros((max_size, 1, state_dim, state_dim))  # 下一个状态缓冲区  
        # 使用numpy创建一个大小为(max_size, 1, state_dim, state_dim)的零矩阵，作为下一个状态缓冲区。  
  
        self.reward = np.zeros((max_size, 1))  # 奖励缓冲区  
        # 使用numpy创建一个大小为(max_size, 1)的零矩阵，作为奖励缓冲区。  
  
        self.not_done = np.zeros((max_size, 1))  # 结束标志缓冲区  
        # 使用numpy创建一个大小为(max_size, 1)的零矩阵，作为结束标志缓冲区。  
  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备为GPU或CPU  
        # 检查是否支持GPU，如果支持则将设备设置为CUDA，否则设置为CPU。  
  
    # __init__函数的结束  
  
    def add(self, state, action, next_state, reward, done):  
        # 添加数据到缓冲区的方法。  
  
        self.state[self.ptr] = state  
        # 将当前状态保存到状态缓冲区中。  
  
        self.action[self.ptr] = action  
        # 将当前动作保存到动作缓冲区中。  
  
        self.next_state[self.ptr] = next_state  
        # 将下一个状态保存到下一个状态缓冲区中。  
  
        self.reward[self.ptr] = reward  
        # 将奖励值保存到奖励缓冲区中。  
  
        self.not_done[self.ptr] = 1. - done  
        # 将1减去done的值保存到结束标志缓冲区中，如果done为True，则对应位置的值为0，否则为1。  
  
        self.ptr = (self.ptr + 1) % self.max_size  # 更新指针位置  
        # 将指针位置加1（模max_size），实现循环利用缓冲区。如果指针已经到达缓冲区末尾，则从头开始。  
  
        self.size = min(self.size + 1, self.max_size)  # 更新缓冲区大小  
        # 将缓冲区大小增加1，但不超过最大容量。  
  
    # add函数的结束  
  
    def sample(self, batch_size):  
        # 从缓冲区中随机采样一批数据的方法。  
  
        ind = np.random.randint(0, self.size, size=batch_size)  
        # 从0到self.size之间随机生成batch_size个整数作为索引。  
  
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )
