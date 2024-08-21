import os
import re

import torch


def cumulative_sum(rewards, discount):
    """ Calculates the cumulative sum with a discount factor. """
    R = 0
    ret = list()
    for r in reversed(rewards):
        if r == None: continue
        R = r + discount * R
        ret.insert(0, R)
    return ret

def mean(t):
    """ Calculates the mean of the list. """
    if len(t) == 0: return 0
    return sum(t) / len(t)

def save(model, path, name, optimizer=None, epoch=None) -> None:
    """ Saves the model and prints finished message to console. """
    torch.save(model, os.path.join(path, name+'.pt'))
    print('Model saved!')
    if optimizer is not None and epoch is not None:
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, os.path.join(path, 'state_'+name+'.pth'))
        print('Optimizer and current epoch saved!')

def move(tensor, dev):
    """ Moves tensor to device. """
    return tensor.to(dev)

def fill_with_none(l : list, removed):
    """ Fill list with None where removed is True. """
    ret = list()
    idx = 0
    for r in removed:
        if r == True: ret.append(None)
        else: 
            ret.append(l[idx])
            idx += 1
    return ret

def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks. "z23a" -> ["z", 23, "a"] """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect. """
    l.sort(key=alphanum_key)
