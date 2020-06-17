import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random


class CReLU(nn.Module):
    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], 1)


class dueling_ddqn(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(dueling_ddqn, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.feature_net = nn.Sequential(
            nn.Linear(self.observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.advantage_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.value_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim)
        )

    def forward(self, observation):
        feature = self.feature_net(observation)
        adv = self.advantage_net(feature)
        val = self.value_net(feature)
        return adv + val - adv.mean()

    def act(self, observation, epsilon):
        q_value = self.forward(observation)
        action = q_value.max(1)[1].data[0].item()
        probs = np.ones(self.action_dim, dtype=float) * epsilon / self.action_dim
        probs[action] += (1.0 - epsilon)
        return probs


class policy(nn.Module):
    def __init__(self, observation_dim, action_dim):
        super(policy, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim

        self.feature_net = nn.Sequential(
            nn.Linear(self.observation_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.fc_net = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, observation):
        return self.fc_net(self.feature_net(observation))

    def act(self, observation):
        probs = self.forward(observation)
        return probs.squeeze(0).cpu().detach().numpy()