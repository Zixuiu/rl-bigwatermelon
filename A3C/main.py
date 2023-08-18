# 导入未来版本的print函数，这样在Python2.x和3.x中都可以运行  
from __future__ import print_function  
  
# 导入必要的库  
import argparse  
import os  

# 这段代码是一个使用A3C算法进行强化学习训练的示例代码。
# 它导入了必要的库和模块，定义了命令行参数，并创建了一个共享模型和优化器。然后，它启动了一个训练进程，并将该进程添加到进程列表中。


# 导入PyTorch库，用于构建神经网络和进行GPU加速  
import torch  
import torch.multiprocessing as mp  
  
# 导入自定义的优化器，这里可能是对标准Adam优化器的修改或扩展  
import my_optim  
# 从'env'模块导入bigwaterlemon环境，这可能是一个特定的游戏或任务环境  
from env import bigwaterlemon  
# 从'model'模块导入ActorCritic模型，这是用于强化学习的常见模型，结合了演员-评论家算法  
from model import ActorCritic  
# 从'train'模块导入train函数，这可能用于训练模型的函数  
from train import train  
  
# 基于指定链接创建ArgumentParser对象，用于解析命令行参数  
# https://githubfast.com/pytorch/examples/tree/master/mnist_hogwild  
# 这个链接似乎是过时的，可能是误写，或者应该是一个特定的代码库  
parser = argparse.ArgumentParser(description='A3C')  
# 添加学习率参数，并设置默认值为0.0001  
parser.add_argument('--lr', type=float, default=0.0001,  
                    help='学习率 (默认: 0.0001)')  
# 添加折扣因子参数，并设置默认值为0.99  
parser.add_argument('--gamma', type=float, default=0.99,  
                    help='折扣因子 (默认: 0.99)')  
# 添加GAE的lambda参数，并设置默认值为1.00  
parser.add_argument('--gae-lambda', type=float, default=1.00,  
                    help='GAE的lambda参数 (默认: 1.00)')  
# 添加熵项系数参数，并设置默认值为0.01  
parser.add_argument('--entropy-coef', type=float, default=0.01,  
                    help='熵项系数 (默认: 0.01)')  
# 添加值函数损失系数参数，并设置默认值为0.5  
parser.add_argument('--value-loss-coef', type=float, default=0.5,  
                    help='值函数损失系数 (默认: 0.5)')  
# 添加梯度裁剪系数参数，并设置默认值为50  
parser.add_argument('--max-grad-norm', type=float, default=50,  
                    help='梯度裁剪系数 (默认: 50)')  
# 添加随机种子参数，并设置默认值为1  
parser.add_argument('--seed', type=int, default=1,  
                    help='随机种子 (默认: 1)')  
# 添加训练进程数参数，并设置默认值为4  
parser.add_argument('--num-processes', type=int, default=1,  
                    help='使用的训练进程数 (默认: 4)')  
# 添加A3C中的前向步数参数，并设置默认值为20  
parser.add_argument('--num-steps', type=int, default=20,  
                    help='A3C中的前向步数 (默认: 20)')  
# 添加一个回合的最大长度参数，并设置默认值为100000  
parser.add_argument('--max-episode-length', type=int, default=100000,  
                    help='一个回合的最大长度 (默认: 100000)')  
# 添加日志目录参数，并设置默认值为'./logs/'  
parser.add_argument('--log-dir', default='./logs/',  
                    help='日志目录 (默认: ./logs/)')  
# 添加实验名称参数，并设置默认值为'test'  
parser.add_argument('--exp-name', default='test',  
                    help='实验名称 (默认: test)')  
  
# 如果当前文件被作为主程序直接运行，而不是被导入到其他文件中使用，则执行以下代码块。这是Python的一个常用模式，用于确保某些代码只在特定条件下执行。  
if __name__ == '__main__':  
  
    # 设置环境变量'OMP_NUM_THREADS'为1，这可能是为了禁用多线程的OpenMP或者指定线程数为1（具体取决于操作系统和Python环境）。  
    # OpenMP是一个用于编程语言（如C，C++，Fortran）的并行计算API，它允许程序员在程序中指定可以并行执行的任务。  
    # 在某些情况下，为了防止并行计算带来的问题（如数据竞争、线程同步等），可能需要禁用或限制OpenMP的线程数。  
    os.environ['OMP_NUM_THREADS'] = '1'  
  
    # 设置环境变量'CUDA_VISIBLE_DEVICES'为空字符串，这可能是为了在程序运行时不使用CUDA（如果程序依赖于GPU计算的话）。  
    # CUDA是NVIDIA推出的并行计算平台和API，它允许开发者使用GPU进行通用计算。但是，如果环境没有安装CUDA或者没有可用的GPU，这个设置可以防止程序在运行时出现错误。  
    os.environ['CUDA_VISIBLE_DEVICES'] = ""  
  
    # 使用argparse模块解析命令行参数  
    args = parser.parse_args()  
  
    # 设置PyTorch的随机种子，以确保结果的可重复性。这对于需要随机性的算法（如强化学习）非常重要。  
    torch.manual_seed(args.seed)  
  
    # 创建一个Actor-Critic模型，这个模型通常用于处理同时需要行为和状态估计的决策问题。这里的具体参数（1和160）不清楚，可能是某种特定的模型配置。  
    shared_model = ActorCritic(1, 160)  
  
    # 将模型设置为可以在多个进程之间共享的状态。这对于分布式强化学习算法非常重要。  
    shared_model.share_memory()  
  
    # 创建一个Adam优化器，用于在训练过程中调整模型的参数。这里的具体参数（优化器的类型、学习率等）可能根据模型和任务的不同而变化。  
    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)  
  
    # 和模型一样，将优化器设置为可以在多个进程之间共享的状态。  
    optimizer.share_memory()  
  
    # 创建一个空的进程列表，用于存储后续创建的进程。  
    processes = []  
  
    # 创建一个新的进程，指定它的目标是运行名为'train'的函数，并传递必要的参数。这里的具体参数可能根据模型、任务和训练策略的不同而变化。  
    p = mp.Process(target=train, args=(args.num_processes, args, shared_model))  
    p.start()  # 启动新创建的进程  
  
    # 将新创建的进程添加到进程列表中，以便后续可以管理和监控这些进程。  
    processes.append(p)
