from actions import Action


def reward_get_immediate(action, state, optim_a = None) -> float:
    """ Given a state and an action, return the immediate reward. """
    if action == Action.FINISH: return 10
    else: return 0

def reward_crash() -> float:
    """ Returns the crash reward. """
    return 0
