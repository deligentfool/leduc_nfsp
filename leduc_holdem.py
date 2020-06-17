import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import rlcard
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

from NFSPAgent import NFSPAgent
from RandomAgent import RandomAgent

import os

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    episode_num = 50000
    evaluate_every = 1000
    evaluate_num = 1000
    env = rlcard.make('leduc-holdem', config={'seed': 0})
    eval_env = rlcard.make('leduc-holdem', config={'seed': 0})
    agents = []
    for _ in range(2):
        agent = NFSPAgent(
            observation_dim=env.state_shape[0],
            action_dim=env.action_num,
            epsilon_init=0.06,
            decay=1000000,
            epsilon_min=0,
            update_freq=1000,
            sl_lr=0.005,
            rl_lr=0.01,
            sl_capa=100000,
            rl_capa=30000,
            n_step=1,
            gamma=0.99,
            eta=0.1,
            rl_start=1000,
            sl_start=1000,
            train_freq=1,
            rl_batch_size=256,
            sl_batch_size=256,
            device=device,
            eval_mode='average'
        )
        agents.append(agent)
    random_agent = RandomAgent(eval_env.action_num)

    env.set_agents(agents)
    eval_env.set_agents([agents[0], random_agent])

    os.makedirs('./log', exist_ok=True)
    log_dir = './log'
    logger = Logger(log_dir)

    for episode in range(episode_num):
        for agent in agents:
            agent.choose_policy_mode()

        trajectories, _ = env.run(is_training=True)

        for i in range(env.player_num):
            for ts in trajectories[i]:
                if len(ts) > 0:
                    agents[i].add_traj([ts[0]['obs'], ts[1], ts[2], ts[3]['obs'], ts[4]])

        if episode % (evaluate_every // 10) == 0:
            print('episode : {}'.format(episode))
        if episode % evaluate_every == 0:
            logger.log_performance(env.timestep, tournament(eval_env, evaluate_num)[0])

    logger.close_files()
    logger.plot('NFSP')
