import itertools
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from actions import select_action_from_probabilities
from dataset import DataSet, collate_fn
from environment import *
from logger import *
from models.cnn import CNN
from models.mlp import MLP
from utils import *


def enroll_episodes(states, max_steps, device, model):
    """ Enroll all states. """
    action_list = list()
    rewards_list = list()
    log_probs_list = list()
    state_value_list = list()
    
    removed = [False for _ in states]
    for i in range(max_steps):
        in_tensor = make_feature_tensor(states)
        in_tensor = move(in_tensor, device)
        action_probs, state_values = model(in_tensor)
        action_probs = move(action_probs, 'cpu')
        state_values = move(state_values, 'cpu')
        actions, log_probs = select_action_from_probabilities(action_probs)  
        rewards, done = perform_actions_on_states(states, actions)
        
        rewards = fill_with_none(rewards, removed)
        log_probs = fill_with_none(log_probs, removed)
        state_values = fill_with_none(state_values, removed)

        states, _, removed = remove_done(states, None, done, removed=removed)
       
        state_value_list.append(state_values)
        log_probs_list.append(log_probs)
        action_list.append(actions)
        rewards_list.append(rewards)

        if len(states) == 0: break
    return rewards_list, action_list, log_probs_list, state_value_list

def enroll_optimal_episodes(states, sequences, max_steps, device, model):
    """ Enrolls all states with respect to the optimal sequences. """
    action_list = list()
    rewards_list = list()
    log_probs_list = list()
    state_value_list = list()

    correct = 0
    total = 0

    removed = [False for _ in states]
    for i in range(max_steps):
        in_tensor = make_feature_tensor(states)
        in_tensor = move(in_tensor, device)
        action_probs, state_values = model(in_tensor)
        action_probs = move(action_probs, 'cpu')
        state_values = move(state_values, 'cpu')
        actions, log_probs = select_optimal_actions(action_probs, sequences)  
        pred_actions, _ = select_action_from_probabilities(action_probs.detach())
        correct += len(list(filter(lambda x: x[0] == x[1], zip(actions, pred_actions))))
        total += len(actions)
        rewards, done = perform_optimal_actions(states, sequences)
        
        rewards = fill_with_none(rewards, removed)
        log_probs = fill_with_none(log_probs, removed)
        state_values = fill_with_none(state_values, removed)

        states, sequences, removed = remove_done(states, sequences, done, removed=removed)
        
        state_value_list.append(state_values)
        log_probs_list.append(log_probs)
        action_list.append(actions)
        rewards_list.append(rewards)
        
        if len(states) == 0: break
    return rewards_list, action_list, log_probs_list, state_value_list, (correct / total)*100

def backward_pass(model, optimizer, discount, rewards, log_probs, state_values) -> float:
    """ Finishes an episodes with the backward pass (parameter updates). """
    rewards_t = list(map(list, itertools.zip_longest(*rewards, fillvalue=None)))
    returns = list(map(lambda x: cumulative_sum(x, discount), rewards_t))
    log_probs_t = list(map(list, itertools.zip_longest(*log_probs, fillvalue=None)))
    state_values_t = list(map(list, itertools.zip_longest(*state_values, fillvalue=None)))
    
    policy_losses = list()
    value_losses = list()
    for r, l, s in zip(returns, log_probs_t, state_values_t):      
        s = list(filter(lambda x: x != None , s))
        l = list(filter(lambda x: x != None, l))
        
        advantage = list(map(lambda x: x[0] - x[1].item(), zip(r, s)))
        pol_loss = list(map(lambda x: -x[0] * x[1], zip(l, advantage)))
        val_loss = list(map(lambda x: F.smooth_l1_loss(x[1],torch.tensor([x[0]])), zip(r, s)))

        policy_losses.append(sum(pol_loss))
        value_losses.append(sum(val_loss))

    loss = mean(policy_losses) + mean(value_losses)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss, returns


def train() -> None:
    """ This is the main training function containing the training loop. """

    model_path  = None
    data_path   = 'data/train'
    batch_size  = 256
    discount    = 0.97
    max_steps   = int(5 / (1 - discount))
    lr          = 0.0001
    num_epochs  = 10000000
    imitation   = True
    start_epoch = 0

    logger = Logger('runs')
    # logger = NoneLogger()

    dataset = DataSet(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, collate_fn=collate_fn)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = CNN()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    if model_path != None: 
        state = torch.load(os.path.join(os.getcwd(), model_path), map_location=device)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        
    model.train()
    
    logger.log_meta(batch_size, lr, num_epochs, optimizer, data_path, model)

    try:
        logger.log_start()
        run_reward = list()
        run_correct = list()
        for epoch in range(start_epoch, num_epochs):
            time_enroll = 0
            time_backward = 0
            start = time.perf_counter()
            loss_sum = 0
            for batch in dataloader:
                states, sequences = get_states_and_sequences(batch)
                
                time_1 = time.perf_counter()
                correct = None
                if not imitation:
                    rewards, actions, log_probs, state_values = enroll_episodes(states, max_steps, device, model)
                else: 
                    rewards, actions, log_probs, state_values, correct = enroll_optimal_episodes(states, sequences, max_steps, device, model)
                time_2 = time.perf_counter()
                loss, returns = backward_pass(model, optimizer, discount, rewards, log_probs, state_values)
                time_3 = time.perf_counter()
                

                run_reward.append(mean(list(map(lambda r: r[-1], returns))))
                if len(run_reward) > 5: run_reward.pop(0)
                if correct is not None: run_correct.append(correct)
                if len(run_correct) > 5: run_correct.pop(0)
                loss_sum += loss.item()
                
                time_enroll += time_2 - time_1
                time_backward += time_3 - time_2
            end = time.perf_counter()
            logger.log_all(epoch, loss_sum / (len(dataset) / batch_size), mean(run_reward), mean(run_correct), end - start, time_enroll / (end - start) * 100, time_backward / (end - start) * 100)
            logger.flush()
            if logger.is_checkpoint(epoch): save(model, logger.get_run_dir(), 'checkpoint_'+str(epoch))
    except KeyboardInterrupt:
        print(' Keyboard interrupt.')
        logger.log_end(epoch)
        logger.close()
        save(model, logger.get_run_dir(), 'model', optimizer=optimizer, epoch=epoch)
        return
    logger.log_end(epoch)
    logger.close()
    save(model, logger.get_run_dir(), 'model', optimizer=optimizer, epoch=epoch)
        
if __name__ == "__main__":
    train()
