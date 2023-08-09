import numpy as np
import torch
import argparse
import os

import utils
import TD3
from env import bigwaterlemon

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)              # 设置Gym、PyTorch和Numpy的种子
    parser.add_argument("--start_timesteps", default=25e3, type=int)# 初始随机策略使用的时间步数
    parser.add_argument("--max_timesteps", default=1e6, type=int)   # 运行环境的最大时间步数
    parser.add_argument("--expl_noise", default=0.1)                # 高斯探索噪声的标准差
    parser.add_argument("--batch_size", default=100, type=int)      # actor和critic的批大小
    parser.add_argument("--discount", default=0.99)                 # 折扣因子
    parser.add_argument("--tau", default=0.005)                     # 目标网络更新速率
    parser.add_argument("--policy_noise", default=0.2)              # 在critic更新期间添加到目标策略的噪声
    parser.add_argument("--noise_clip", default=0.5)                # 目标策略噪声的范围
    parser.add_argument("--policy_freq", default=2, type=int)       # 延迟的策略更新频率
    parser.add_argument("--save_model", action="store_true")        # 保存模型和优化器参数
    parser.add_argument("--load_model", default="")                 # 模型加载文件名，""表示不加载，"default"使用默认文件名
    args = parser.parse_args()

    file_name = f"TD3_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: TD3, Env: BigWatermelon, Seed: {args.seed}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = bigwaterlemon()

    # 设置种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 80
    action_dim = 1
    max_action = 160

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    kwargs["policy_noise"] = args.policy_noise * max_action
    kwargs["noise_clip"] = args.noise_clip * max_action
    kwargs["policy_freq"] = args.policy_freq
    policy = TD3.TD3(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

    state, done = env.reset(), False
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):

        episode_timesteps += 1

        # 随机选择动作或根据策略选择动作
        if t < args.start_timesteps:
            action = env.sample_action()
        else:
            action = (
                policy.select_action(np.array(state))
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clamp(-max_action / 2, max_action / 2) + (max_action / 2)

        # 执行动作
        next_state, reward, done = env.step(action)
        done_bool = float(done) if episode_timesteps < 10000 else 0

        # 将数据存储在回放缓冲区中
        replay_buffer.add(state, action, next_state, reward, done_bool)

        state = next_state
        episode_reward += reward

        # 收集足够的数据后训练智能体
        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            # +1是为了考虑从0开始计数。+0是因为即使done=True，ep_timesteps也会增加+1
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            # 重置环境
            state, done = env.reset(), False
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        policy.save(file_name)
