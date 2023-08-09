import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from env import bigwaterlemon  # 导入自定义环境 bigwaterlemon
from model import ActorCritic  # 导入自定义模型 ActorCritic

def ensure_shared_grads(model, shared_model):
    # 确保共享模型的梯度与当前模型的梯度相同
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad

def train(rank, args, shared_model, optimizer=None):
    torch.manual_seed(args.seed + rank)  # 设置随机种子

    env = bigwaterlemon()  # 创建环境对象

    model = ActorCritic(1, 160)  # 创建 ActorCritic 模型对象

    if optimizer is None:
        optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)  # 创建优化器对象

    model.train()  # 设置模型为训练模式

    state = env.reset()  # 初始化环境状态
    state = torch.from_numpy(state)  # 转换状态为张量
    done = True

    episode_length = 0  # 记录当前回合的步数
    episode_reward = 0  # 记录当前回合的累计奖励
    while True:
        # 与共享模型同步
        model.load_state_dict(shared_model.state_dict())
        if done:
            cx = torch.zeros(1, 256)  # 初始化隐藏状态
            hx = torch.zeros(1, 256)  # 初始化细胞状态
        else:
            cx = cx.detach()  # 分离隐藏状态
            hx = hx.detach()  # 分离细胞状态

        values = []  # 存储值函数
        log_probs = []  # 存储对数概率
        rewards = []  # 存储奖励
        entropies = []  # 存储熵

        for step in range(args.num_steps):
            episode_length += 1  # 增加步数计数
            value, logit, (hx, cx) = model((state.unsqueeze(0), (hx, cx)))  # 前向传播得到值函数、策略概率和隐藏状态
            prob = F.softmax(logit, dim=-1)  # 计算策略概率
            log_prob = F.log_softmax(logit, dim=-1)  # 计算对数概率
            entropy = -(log_prob * prob).sum(1, keepdim=True)  # 计算熵
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).detach()  # 根据策略概率采样动作
            log_prob = log_prob.gather(1, action)  # 获取采样动作的对数概率

            state, reward, done = env.step(action.numpy()[0][0])  # 执行动作并获取下一状态、奖励和是否结束标志
            done = done or episode_length >= args.max_episode_length  # 判断是否达到最大步数限制

            episode_reward += reward  # 累计奖励

            if done:
                episode_length = 0  # 重置步数计数
                state = env.reset()  # 重置环境状态

            state = torch.from_numpy(state)  # 转换下一状态为张量
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                episode_reward = 0  # 重置累计奖励
                break

        R = torch.zeros(1, 1)
        if not done:
            value, _, _ = model((state.unsqueeze(0), (hx, cx)))  # 计算值函数
            R = value.detach()

        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]  # 更新累计回报
            advantage = R - values[i]  # 计算优势值
            value_loss = value_loss + 0.5 * advantage.pow(2)  # 计算值函数损失

            # 广义优势估计
            delta_t = rewards[i] + args.gamma * values[i + 1] - values[i]
            gae = gae * args.gamma * args.gae_lambda + delta_t

            policy_loss = policy_loss - log_probs[i] * gae.detach() - args.entropy_coef * entropies[i]

        optimizer.zero_grad()  # 梯度清零

        (policy_loss + args.value_loss_coef * value_loss).backward()  # 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)  # 梯度裁剪

        ensure_shared_grads(model, shared_model)  # 确保共享梯度
        optimizer.step()  # 更新模型参数

        torch.save(model, "model.pkl")  # 保存模型
