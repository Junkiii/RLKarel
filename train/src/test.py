import os

import torch

from actions import *
from dataset import DataSet
from environment import State

if __name__ == "__main__":
    id = 234
    model_path = os.path.join(os.getcwd(), 'trained/model.pt')
    model = torch.load(model_path)
    model.eval()
    model.to("cpu")

    dataset = DataSet('data_easy/train')
    task, seq = dataset.__getitem__(id)

    state = State()
    state.load_from_data(task)

    state.pretty_print()

    done = False
    while not done:
        input_features = torch.from_numpy(state.get_feature_representation()).float()
        input_features = input_features.unsqueeze(0)

        action_probs, _ = model(input_features)

        action = select_max_action(action_probs)

        _ , done = state.perform_action(action)

        print(action_probs)
        print('Action: ', action)
        state.pretty_print()

        