import math
import torch
import torch.optim as optim

class SharedAdam(optim.Adam):
    """实现带有共享状态的Adam算法
    """


# SharedAdam是一个继承自torch.optim.Adam的类，实现了带有共享状态的Adam算法。它重写了父类的step方法，实现了参数更新的逻辑。
# 在step方法中，遍历每个参数组和参数，根据Adam优化算法的公式更新参数。同时，通过share_memory方法，将优化器的状态信息共享到多个进程中。
    
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.zeros(1)  # 初始化优化器的步数为0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()  # 初始化一阶矩的移动平均为0
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()  # 初始化二阶矩的移动平均为0

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'].share_memory_()  # 共享优化器的步数内存
                state['exp_avg'].share_memory_()  # 共享一阶矩的移动平均内存
                state['exp_avg_sq'].share_memory_()  # 共享二阶矩的移动平均内存

    def step(self, closure=None):  
        """执行单个优化步骤  
          
        参数:  
            closure (callable, optional): 重新评估模型并返回损失的闭包函数  
          
        返回:  
            loss (float, optional): 如果提供了闭包函数，则返回优化步骤的损失  
        """  
      
        loss = None  
        # 判断是否提供了闭包函数，如果有，则执行闭包函数并获取损失  
        if closure is not None:  
            loss = closure()  
      
        # 遍历每个参数组  
        for group in self.param_groups:  
            # 遍历每个参数组中的参数  
            for p in group['params']:  
                # 判断该参数是否有梯度信息，若无则跳过  
                if p.grad is None:  
                    continue  
                # 获取该参数的梯度信息  
                grad = p.grad.data  
                # 获取该参数在优化器中的状态信息  
                state = self.state[p]  
      
                # 获取一阶矩和二阶矩的移动平均系数  
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']  
                # 获取beta1和beta2的值  
                beta1, beta2 = group['betas']  
      
                # 更新状态中的步数  
                state['step'] += 1  
      
                # 如果参数组中设置了权重衰减，则对梯度进行权重衰减  
                if group['weight_decay'] != 0:  
                    grad = grad.add(group['weight_decay'], p.data)  
      
                # 更新一阶矩和二阶矩的移动平均系数  
                # beta1是控制一阶矩移动平均的系数，beta2是控制二阶矩移动平均的系数  
                # addcmul是向量加法并乘法，这里是将梯度与自身相乘然后加到exp_avg_sq中  
                exp_avg.mul_(beta1).add_(1 - beta1, grad)  
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)  
      
                # 计算二阶矩的平方根，并添加一个小的正值eps以防止除数为0的情况  
                denom = exp_avg_sq.sqrt().add_(group['eps'])  
      
                # 计算一阶矩和二阶矩的偏差修正系数，即(1 - beta1^step) 和 (1 - beta2^step)  
                bias_correction1 = 1 - beta1 ** state['step'].item()  
                bias_correction2 = 1 - beta2 ** state['step'].item()  
                # 根据公式计算步长，其中bias_correction2是控制步长缩放的系数  
                step_size = group['lr'] * math.sqrt(  
                    bias_correction2) / bias_correction1  
      
                # 根据Adam优化算法的公式更新参数，即p.data = p.data - step_size * (exp_avg/denom)  
                p.data.addcdiv_(-step_size, exp_avg, denom)  # 更新参数

        return loss
