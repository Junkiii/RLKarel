import math
import os

import matplotlib.pyplot as plt
import numpy as np


def load_log(path):
    """ Loads a log file from the given path and returns a list. """
    ret = list()
    with open(path) as f:
        lines = f.readlines()
    for l in lines:
        l = l.replace('(','')
        l = l.replace(')','')
        l = l.strip()
        tmp = l.split(',')
        ret.append((int(tmp[0]),float(tmp[1])))
    return ret

def visualize_log(path):
    """ Visualizes a log file using matplotlib. """
    log = load_log(path)
    iteration = np.array(list(map(lambda l: l[0], log)))
    values = np.array(list(map(lambda l: l[1], log)))

    fig = plt.figure()
    ax = fig.add_subplot(111, label='log')
    ax.plot(iteration, values, color='k')

    plt.show()

def visualize_log_multiple(paths):
    """ Visualizes multiple log files in one plot. Cuts off the longer logs! """
    iterations_list = list()
    values_list = list()

    shortest = math.inf

    for p in paths:
        log = load_log(p)
        iteration = np.array(list(map(lambda l: l[0], log)))
        values = np.array(list(map(lambda l: l[1], log)))
        iterations_list.append(iteration)
        values_list.append(values)
        shortest = min(shortest, len(iteration))
    
    idx = 0
    for iteration, values in zip(iterations_list, values_list):
        # Cut off:
        remove = len(iteration) - shortest
        if remove != 0:
            iteration = iteration[:-remove]
            values = values[:-remove]
            iterations_list[idx] = iteration
            values_list[idx] = values
        idx += 1

    for iteration, values in zip(iterations_list, values_list):
        plt.plot(iteration, values) 

    # Remember to set the correct legend!
    plt.xlabel('Percentage / Loss')
    plt.ylabel('Epochs')
    plt.legend(['% correct actions during training', 'loss during training'])

    plt.show()

def visualize_checkpoint_validation(path):
    """ Plots the checkpoints validation with matplotlib. """
    with open(path) as f:
        lines = f.readlines()
    validation = list()
    for l in lines:
        l = l.replace('(','')
        l = l.replace(')','')
        l = l.strip()
        tmp = l.split(',')
        validation.append((int(tmp[0]), int(tmp[1]), int(tmp[2]), int(tmp[3])))
    
    iteration = np.array(list(map(lambda v: v[0], validation)))
    finished = np.array(list(map(lambda v: v[1] / v[3] * 100, validation)))
    finished_optimal = np.array(list(map(lambda v: v[2] / v[3] * 100, validation)))
    
    plt.plot(iteration, finished)
    plt.plot(iteration, finished_optimal)
    
    plt.xlabel('Percentage')
    plt.ylabel('Epochs')
    plt.legend(['% tasks finished', '% tasks finished optimally'])

    plt.show()

if __name__ == "__main__":
    log_file_path1 = os.path.join(os.getcwd(), 'runs/run_2022-01-29|02:22:11/bwd_time_log.txt')
    log_file_path2 = os.path.join(os.getcwd(), 'runs/run_2022-01-29|02:22:11/enr_time_log.txt')

    checkpoints_validation_path = os.path.join(os.getcwd(), 'runs/run_2022-01-30|02:06:55/checkpoints_validation.txt')

    # visualize_log(log_file_path1)

    # visualize_log_multiple([log_file_path1, log_file_path2])

    visualize_checkpoint_validation(checkpoints_validation_path)