import gym
import lasertag
import numpy as np
from model import dueling_ddqn, policy
from buffer import reservoir_buffer, n_step_replay_buffer, replay_buffer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from utils import action_mask


class NFSPAgent(object):
    def __init__(self, observation_dim, action_dim, epsilon_init, decay, epsilon_min, update_freq, sl_lr, rl_lr, sl_capa, rl_capa, n_step, gamma, eta, rl_start, sl_start, train_freq, rl_batch_size, sl_batch_size, device, eval_mode):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.epsilon_init = epsilon_init
        self.decay = decay
        self.epsilon_min = epsilon_min
        self.update_freq = update_freq
        self.sl_lr = sl_lr
        self.rl_lr = rl_lr
        self.sl_capa = sl_capa
        self.rl_capa = rl_capa
        self.n_step = n_step
        self.gamma = gamma
        self.eta = eta
        self.sl_start = sl_start
        self.rl_start = rl_start
        self.train_freq = train_freq
        self.rl_batch_size = rl_batch_size
        self.sl_batch_size = sl_batch_size
        self.device = device
        self.use_raw = False
        self.eval_mode = eval_mode

        self.sl_buffer = reservoir_buffer(self.sl_capa)
        if self.n_step > 1:
            self.rl_buffer = n_step_replay_buffer(self.rl_capa, self.n_step, self.gamma)
        else:
            self.rl_buffer = replay_buffer(self.rl_capa)
        self.dqn_eval = dueling_ddqn(self.observation_dim, self.action_dim).to(self.device)
        self.dqn_target = dueling_ddqn(self.observation_dim, self.action_dim).to(self.device)
        self.dqn_target.load_state_dict(self.dqn_eval.state_dict())

        self.policy = policy(self.observation_dim, self.action_dim).to(self.device)

        self.epsilon = lambda x: self.epsilon_min + (self.epsilon_init - self.epsilon_min) * math.exp(-1. * x / self.decay)

        self.dqn_optimizer = torch.optim.Adam(self.dqn_eval.parameters(), lr=self.rl_lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.sl_lr)

        self.choose_policy_mode()

        self.count = 0

    def rl_train(self):
        observation, action, reward, next_observation, done = self.rl_buffer.sample(self.rl_batch_size)

        observation = torch.FloatTensor(observation).to(self.device)
        action = torch.LongTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device)
        next_observation = torch.FloatTensor(next_observation).to(self.device)
        done = torch.FloatTensor(done).to(self.device)

        q_values = self.dqn_eval.forward(observation)
        next_q_values = self.dqn_target.forward(next_observation)
        argmax_actions = self.dqn_eval.forward(next_observation).max(1)[1].detach()
        next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + self.gamma * (1 - done) * next_q_value

        loss = (q_value - expected_q_value.detach()).pow(2).mean()

        self.dqn_optimizer.zero_grad()
        loss.backward()
        self.dqn_optimizer.step()
        if self.count % self.update_freq == 0:
            self.dqn_target.load_state_dict(self.dqn_eval.state_dict())

    def sl_train(self):
        observation, action = self.sl_buffer.sample(self.sl_batch_size)

        observation = torch.FloatTensor(observation).to(self.device)
        action = torch.LongTensor(action).to(self.device)

        probs = self.policy.forward(observation)
        prob = probs.gather(1, action.unsqueeze(1)).squeeze(1)
        log_prob = prob.log()
        loss = -log_prob.mean()

        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()

    def choose_policy_mode(self):
        self.policy_mode = 'average' if random.random() > self.eta else 'best'

    def step(self, state):
        self.count += 1
        obs = state['obs']
        legal_actions = state['legal_actions']
        if self.policy_mode == 'best':
            probs = self.dqn_eval.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device), self.epsilon(self.count))
        else:
            probs = self.policy.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device))
        probs = action_mask(probs, legal_actions)
        action = np.random.choice(len(probs), p=probs)
        return action

    def eval_step(self, state):
        obs = state['obs']
        legal_actions = state['legal_actions']
        if self.eval_mode == 'best':
            probs = self.dqn_eval.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device), 0)
        else:
            probs = self.policy.act(torch.FloatTensor(np.expand_dims(obs, 0)).to(self.device))
        probs = action_mask(probs, legal_actions)
        action = np.random.choice(len(probs), p=probs)
        return action, probs

    def add_traj(self, traj):
        self.rl_buffer.store(* traj)
        if self.policy_mode == 'best':
            self.sl_buffer.store(* traj[: 2])
        if len(self.rl_buffer) > self.rl_start and len(self.sl_buffer) > self.sl_start and self.count % self.train_freq == 0:
            self.rl_train()
            self.sl_train()