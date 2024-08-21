import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from actions import Action
from dataset import DataSet, collate_fn
from environment import *
from utils import *


def validate_checkpoints(run_path, data_name, batch_size, save=False):
    """ Validates all the checkpoints of one training run. """
    path = os.path.join(os.getcwd(), run_path)
    print(path)

    checkpoints = list()
    for f in os.listdir(path):
        name = os.fsdecode(f)
        if 'checkpoint' in name and '.pt' in name:
            checkpoints.append(os.path.join(path, name))
    
    # https://nedbatchelder.com/blog/200712/human_sorting.html
    sort_nicely(checkpoints)
    validation = list()

    for idx, cp in enumerate(checkpoints):
        finished, optimal_finished, total = validate(cp, data_name, batch_size=batch_size)
        validation.append(((idx+1)*20-1, finished, optimal_finished, total))

    if save:
        with open(os.path.join(path, 'checkpoints_validation.txt'), 'w+') as f:
            for item in validation:
                f.write(str(item)+'\n')


def validate(model_name, data_name, batch_size=1, max_steps=1000) -> tuple:
    """ Validates the model on the specified data set based on optimal solved tasks. """
    print(f'Running optimal validation for {model_name} with {data_name}...')
    dataset = DataSet(data_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model_path = os.path.join(os.getcwd(), model_name)
    model = torch.load(model_path)
    model.eval()
    model.to(device)

    optimal_finished = 0
    finished = 0

    for batch in dataloader:
        states, sequences = get_states_and_sequences(batch)
        
        for t in range(max_steps):
            in_tensor = make_feature_tensor(states)
            in_tensor = move(in_tensor, device)
            action_probs, _ = model(in_tensor)
            action_probs = move(action_probs, 'cpu')
            actions, _ = select_max_actions(action_probs)
            _, done = perform_actions_on_states(states, actions)

            # Checking how many finished with an optimal sequence
            finished += sum(list(map(lambda s,a,d: (a == Action.FINISH) and d and s.check_finish(), states, actions, done)))
            optimal_finished += sum(list(map(lambda s, a, d, q: (a == Action.FINISH) and d and s.check_finish() and (t+1) == len(q), states, actions, done, sequences)))

            states, sequences, _ = remove_done(states, sequences, done)
            if len(states) == 0: break
    
    print('Validation complete!')
    print(f'Optimal Validation: \t{optimal_finished}/{len(dataset)} ({optimal_finished / len(dataset) * 100:.1f}%)')
    print(f'Validation: \t\t{finished}/{len(dataset)} ({finished / len(dataset) * 100:.1f}%)')
    return finished, optimal_finished, len(dataset)


if __name__ == "__main__":
    # Single validation run code:
    model_name = 'runs/run_2022-01-30|02:06:55/model.pt'
    run_path = 'runs/run_2022-01-30|02:06:55'
    data_name = 'data_medium/val'
    batch_size = 256
    
    # validate(model_name, data_name, batch_size)

    validate_checkpoints(run_path, data_name, batch_size=batch_size, save=True)

