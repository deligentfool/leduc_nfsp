import numpy as np

def action_mask(probs, legal_actions):
    action_probs = np.zeros_like(probs)
    action_probs[legal_actions] = probs[legal_actions]
    if np.sum(action_probs) == 0:
        action_probs[legal_actions] = 1. / len(legal_actions)
    else:
        action_probs /= sum(action_probs)
    return action_probs