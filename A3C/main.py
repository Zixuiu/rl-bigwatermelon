# 导入未来版本的print函数，这样在Python2.x和3.x中都可以运行  
from __future__ import print_function  
  
# 导入必要的库  
import argparse  
import os  
  
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
  
# 如果当前文件被作为主程序直接运行，而不是被导入到其他文件中使用，则执行以下代码块  
if __name__ == '__main__':  
    # 设置环境变量'OMP_NUM_THREADS'为1，这可能是为了禁用多线程的OpenMP或者指定线程数为1（具体取决于操作系统和Python环境）
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = ""

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    shared_model = ActorCritic(1, 160)
    shared_model.share_memory()

    optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
    optimizer.share_memory()

    processes = []

    p = mp.Process(target=train, args=(args.num_processes, args, shared_model))
    p.start()
    processes.append(p)
