import numpy as np

from actions import *
from direction import *
from rewards import *
from sequence import Sequence


class State:
    """ This class represents an entire state including pre- and post-grid. """

    def __init__(self, data=None) -> None:
        self.__reset_state()
        if data != None:
            self.load_from_data(data)
    
    def load_from_data(self, data) -> None:
        """ Loads a state (pre- and post-grid) from a json data."""

        x_curr = data['pregrid_agent_col']
        y_curr = data['pregrid_agent_row']
        dir_curr = direction_get_value(data['pregrid_agent_dir'])
        
        x_goal = data['postgrid_agent_col']
        y_goal = data['postgrid_agent_row']
        dir_goal = direction_get_value(data['postgrid_agent_dir'])

        walls = data['walls']

        markers_curr = data['pregrid_markers']
        markers_goal = data['postgrid_markers']


        self.__set_curr_agent(x_curr, y_curr, dir_curr)
        self.__set_goal_agent(x_goal, y_goal, dir_goal)

        self.__set_walls(walls)

        self.__set_curr_markers(markers_curr)
        self.__set_goal_markers(markers_goal)       

    def pretty_print(self) -> None:
        """ Prints the state matrices to the console. """
        print('current:')
        print('\t agents \t markers \t walls')
        for i in range(4):
            print('\t', end='')
            print(self.__agent_curr[i], end='')
            print('\t', end='')
            print(self.__markers_curr[i], end='')
            print('\t', end='')
            print(self.__walls[i])
        print('goal:')
        print('\t agents \t markers \t walls')
        for i in range(4):
            print('\t', end='')
            print(self.__agent_goal[i], end='')
            print('\t', end='')
            print(self.__markers_goal[i], end='')
            print('\t', end='')
            print(self.__walls[i])

    def perform_action(self, action, optim_a = None) -> tuple:
        """ Performs the action on the current state. Returns the immediate reward. """
        if action == Action.MOVE: return self.move(optim_a)
        elif action ==  Action.TURN_LEFT: return (self.turn_left(optim_a), False)
        elif action ==  Action.TURN_RIGHT: return (self.turn_right(optim_a), False)
        elif action ==  Action.PICK_MARKER: return self.pick_marker(optim_a)
        elif action ==  Action.PUT_MARKER: return self.put_marker(optim_a)
        elif action ==  Action.FINISH: return self.finish(optim_a)
        else: sys.exit("Error: Invalid action in perform_action().")

    def move(self, optim_a = None) -> tuple:
        """ Performs the move action on the current state. Returns the immediate reward. """
        pos = np.argwhere(self.__agent_curr > 0)[0]
        dir = int(self.__agent_curr[pos[0], pos[1]])
        offset = direction_move(dir)
        new_pos = pos + offset
        crash = not self.__check_tile(new_pos)
        if not crash:
            self.__agent_curr = np.zeros((4,4))
            self.__agent_curr[new_pos[0], new_pos[1]] = dir
            return (reward_get_immediate(Action.MOVE, self, optim_a=optim_a), False)
        else:
            return (reward_crash(), True)
        

    def turn_left(self, optim_a = None) -> int:
        """ Performs the turn_left action on the current state. Returns the immediate reward. """
        self.__agent_curr[self.__agent_curr > 0] = direction_turn_left(self.__agent_curr[self.__agent_curr > 0][0])
        return reward_get_immediate(Action.TURN_LEFT, self, optim_a=optim_a)

    def turn_right(self, optim_a = None) -> int:
        """ Performs the turn_right action on the current state. Returns the immediate reward. """
        self.__agent_curr[self.__agent_curr > 0] = direction_turn_right(self.__agent_curr[self.__agent_curr > 0][0])
        return reward_get_immediate(Action.TURN_RIGHT, self, optim_a=optim_a)
        

    def pick_marker(self, optim_a = None) -> tuple:
        """ Performs the pick_marker action on the current state. Returns the immediate reward. """
        pos = np.argwhere(self.__agent_curr > 0)[0]
        crash = not self.__check_marker(pos)
        if not crash:
            self.__markers_curr[pos[0],pos[1]] = 0.0
            return (reward_get_immediate(Action.PICK_MARKER, self, optim_a=optim_a), False)
        else:
            return (reward_crash(), True)
        

    def put_marker(self, optim_a = None) -> tuple:
        """ Performs the put_marker action on the current state. Returns the immediate reward. """
        pos = np.argwhere(self.__agent_curr > 0)[0]
        crash = self.__check_marker(pos)
        if not crash:
            self.__markers_curr[pos[0],pos[1]] = 1.0
            return (reward_get_immediate(Action.PUT_MARKER, self, optim_a=optim_a), False)
        else:
            return (reward_crash(), True)

    def finish(self, optim_a = None) -> tuple:
        """ Performs the finish action on the current state. Returns the immediate reward. """
        crash = not self.check_finish()
        if not crash:
            return (reward_get_immediate(Action.FINISH, self, optim_a=optim_a), True)
        else: 
            return (reward_crash(), True)
    
    def get_feature_representation(self):
        """ Returns the feature representation of the state as a deep copy. """
        features = np.stack((self.__agent_curr, self.__markers_curr, self.__walls, self.__agent_goal, self.__markers_goal))
        return features

    def __check_tile(self, pos) -> bool:
        """ Checks if the given position is a valid position for the agent. """
        return pos[0] >= 0 and pos[0] <= 3 and \
           pos[1] >= 0 and pos[1] <= 3 and \
           self.__walls[pos[0], pos[1]] == 0

    def __check_marker(self, pos) -> bool:
        """ Checks if the tile  at the given position contains a marker. """
        return self.__markers_curr[pos[0], pos[1]] != 0

    def check_finish(self) -> bool:
        """ Checks if the current configuration is the goal configuration. """
        return np.array_equal(self.__agent_goal, self.__agent_curr) and \
            np.array_equal(self.__markers_goal, self.__markers_curr)

    def __reset_state(self) -> None:
        """ Resets the state matrices to zero. """
        self.__markers_curr = np.zeros((4,4))
        self.__agent_curr = np.zeros((4,4))

        self.__markers_goal = np.zeros((4,4))
        self.__agent_goal = np.zeros((4,4))

        self.__walls = np.zeros((4,4))

    def __set_curr_agent(self, x, y, dir) -> None:
        """ Sets the current agent value at position [y,x] to dir. """
        self.__agent_curr[y,x] = dir

    def __set_curr_markers(self, markers) -> None:
        """ Sets the current markers. """
        for m in markers:
            self.__markers_curr[m[0], m[1]] = 1

    def __set_goal_agent(self, x, y, dir) -> None:
        """ Sets the goal agent value at position [y,x] to dir. """
        self.__agent_goal[y,x] = dir

    def __set_goal_markers(self, markers) -> None:
        """ Sets the goal markers. """
        for m in markers:
            self.__markers_goal[m[0], m[1]] = 1

    def __set_walls(self, walls) -> None:
        """ Sets the walls. """
        for w in walls:
            self.__walls[w[0], w[1]] = 1

def get_states_and_sequences(data):
    """ Returns the states list and sequences list in data. """
    return tuple(zip(*list(map(lambda d: (State(d[0]), Sequence(d[1])), data))))

def stack_features(states):
    """ Stack the features representation matrices of the states. """
    return np.stack(list(map(lambda s: s.get_feature_representation(), states)))

def make_feature_tensor(states):
    """ Make feature tensor from states list. """
    in_tensor = stack_features(states)
    in_tensor = torch.from_numpy(in_tensor).float()
    return in_tensor

def perform_actions_on_states(states, actions, sequences=None):
    """ Performs the actions on the states using optimal sequences. """
    if sequences != None:
        return tuple(zip(*list(map(lambda x: x[0].perform_action(x[1],x[2].get_next()), zip(states, actions, sequences)))))
    else:
        return tuple(zip(*list(map(lambda x: x[0].perform_action(x[1]), zip(states, actions)))))

def perform_optimal_actions(states, sequences):
    """ Performs the optimal actions on each state. """
    return tuple(zip(*list(map(lambda x: x[0].perform_action(x[1].pop_action()), zip(states, sequences)))))

def remove_done(states, sequences, done, removed=None):
    """ Removes the finished/crashed states and corresponding sequences. """
    states = list(filter(lambda x: not x[1], zip(states, done)))
    states = list(map(lambda x: x[0], states))
    
    rem =  None
    if removed != None:
        rem = list()
        idx = 0
        for r in removed:
            if r == True: rem.append(True)
            else:
                if done[idx] == True: rem.append(True)
                else: rem.append(False)
                idx += 1
    
    if sequences != None:
        sequences = list(filter(lambda x: not x[1], zip(sequences, done)))
        sequences = list(map(lambda x: x[0], sequences))
    return states, sequences, rem
