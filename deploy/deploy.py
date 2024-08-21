
import json
from argparse import ArgumentParser

import torch

from cnn import CNN
from environment import *


def main():
    desc = 'Karel Task Neural Network Solver!\nInput a .json file with a Karel task and let the magic happen.'
    t_desc = 'Relative path to the .json file with the Karel task.'
    s_desc = 'Enable this if you do not want to pretty print the entire states.'

    parser = ArgumentParser(description=desc)
    parser.add_argument('-t', '--task', dest='task_path', help=t_desc, required=True)
    parser.add_argument('-s', '--short', action='store_true', help=s_desc)
    args = vars(parser.parse_args())

    short = args['short']
    task_path = args['task_path']

    print('Solving the task: ', task_path)

    solve(task_path, short)

def load_task(task_path):
    """ Loads the Karel task from the .json. """
    path = task_path
    with open(path, 'r') as f:
        task_data = json.load(f)
    
    state = State()
    state.load_from_data(task_data)

    return state


def solve(task_path, short):
    """ Solves a Karel task given with task_path. """
    state = load_task(task_path)

    model = CNN()
    model.to('cpu')
    saved = torch.load('model.pth', map_location='cpu')
    model.load_state_dict(saved['model'])
    model.eval()

    if not short:
        print('Initial state:')
        state.pretty_print()
        print()
        print()
        print('Performing moves:')

    done = False
    while not done:
        input_features = torch.from_numpy(state.get_feature_representation()).float()
        input_features = input_features.unsqueeze(0)

        action_probs, _ = model(input_features)

        action = select_max_action(action_probs)

        _, done = state.perform_action(action)

        print()
        print()
        print('ACTION: ', str(action))
        print(f'\t(move: {int(action_probs[0][0]*100)}% left: {int(action_probs[0][1]*100)}% right: {int(action_probs[0][2]*100)}% pick: {int(action_probs[0][3]*100)}% put: {int(action_probs[0][4]*100)}% finish: {int(action_probs[0][5]*100)}%)')
        
        if not short: state.pretty_print(goal=False)


if __name__ == "__main__":
    main()
