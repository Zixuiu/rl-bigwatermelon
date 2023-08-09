from __future__ import print_function

import argparse
import os

import torch
import torch.multiprocessing as mp

import my_optim
from env import bigwaterlemon
from model import ActorCritic
from train import train

# 基于
# https://githubfast.com/pytorch/examples/tree/master/mnist_hogwild
# 训练设置
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='学习率 (默认: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='折扣因子 (默认: 0.99)')
parser.add_argument('--gae-lambda', type=float, default=1.00,
                    help='GAE的lambda参数 (默认: 1.00)')
parser.add_argument('--entropy-coef', type=float, default=0.01,
                    help='熵项系数 (默认: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5,
                    help='值函数损失系数 (默认: 0.5)')
parser.add_argument('--max-grad-norm', type=float, default=50,
                    help='梯度裁剪系数 (默认: 50)')
parser.add_argument('--seed', type=int, default=1,
                    help='随机种子 (默认: 1)')
parser.add_argument('--num-processes', type=int, default=1,
                    help='使用的训练进程数 (默认: 4)')
parser.add_argument('--num-steps', type=int, default=20,
                    help='A3C中的前向步数 (默认: 20)')
parser.add_argument('--max-episode-length', type=int, default=100000,
                    help='一个回合的最大长度 (默认: 1000000)')
parser.add_argument('--log-dir', default='./logs/',
                    help='日志目录 (默认: ./logs/)')
parser.add_argument('--exp-name', default='test',
                    help='实验名称 (默认: test)')

if __name__ == '__main__':
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
