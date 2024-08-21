import sys
from enum import Enum

import torch


class Action(Enum):
    """ Performable actions. """
    MOVE = 1
    TURN_LEFT = 2
    TURN_RIGHT = 3
    PICK_MARKER = 4
    PUT_MARKER = 5
    FINISH = 6

    def idx_to_action(idx):
        """ Maps an action index to the Action. """
        if idx == 0: return Action.MOVE
        elif idx == 1: return Action.TURN_LEFT
        elif idx == 2: return Action.TURN_RIGHT
        elif idx == 3: return Action.PICK_MARKER
        elif idx == 4: return Action.PUT_MARKER
        elif idx == 5: return Action.FINISH
        else: sys.exit("Error: Invalid action index.")

    def idxs_to_action(idxs):
        """ Maps a list of action indices to a list of actions. """
        actions = []
        for i in idxs:
            actions.append(Action.idx_to_action(i))
        return actions

def select_action_from_probabilities(action_probs):
    """ Selects a random action depending on the given distributen. """
    distribution = torch.distributions.Categorical(action_probs)
    idxs = distribution.sample()
    return Action.idxs_to_action(idxs), distribution.log_prob(idxs)

def select_optimal_actions(action_probs, sequences):
    """ Selects the optimal action and returns the log_prob """
    distribution = torch.distributions.Categorical(action_probs)
    actions = list(map(lambda x: x.get_next(), sequences))
    idxs = torch.tensor(list(a.value - 1 for a in actions))
    return Action.idxs_to_action(idxs), distribution.log_prob(idxs)

def select_max_actions(action_probs):
    """ Selects the max action and returns the log_prob. """
    distribution = torch.distributions.Categorical(action_probs)
    idxs = torch.tensor(list(map(lambda x: torch.argmax(x) , action_probs)))
    return Action.idxs_to_action(idxs), distribution.log_prob(idxs)

def select_max_action(action_probs):
    """ Selects the action with the highest probability. """
    idx_max = torch.argmax(action_probs)
    return Action.idx_to_action(idx_max)

 