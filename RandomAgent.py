import numpy as np


class RandomAgent(object):
    def __init__(self, action_num):
        self.use_raw = False
        self.action_num = action_num

    @staticmethod
    def step(state):
        return np.random.choice(state['legal_actions'])

    def eval_step(self, state):
        probs = [0 for _ in range(self.action_num)]
        for i in state['legal_actions']:
            probs[i] = 1/len(state['legal_actions'])
        return self.step(state), probs