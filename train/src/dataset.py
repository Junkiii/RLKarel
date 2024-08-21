import json
import os

from torch.utils.data import Dataset


def collate_fn(data):
    """ Collate function for the data loader. """
    return data

class DataSet(Dataset):
    """ This is the dataset class for loading specific data samples while training. """
    def __init__(self, path) -> None:
        root = os.getcwd()
        self.task_path = os.path.join(root, 'datasets/'+path+'/task') 
        self.seq_path = os.path.join(root, 'datasets/'+path+'/seq')
        self.n_samples = int(len(os.listdir(self.task_path)))
        if 'val' in path: self.val = True
        else: self.val = False

    def __getitem__(self, index):
        """ Returns the task state data and the sequence state data. """
        if self.val: index += 100_000
        task_path = os.path.join(self.task_path, str(index)+'_task.json')
        seq_path = os.path.join(self.seq_path, str(index)+'_seq.json')

        with open(task_path, 'r') as f:
            task_data = json.load(f)
        with open(seq_path, 'r') as f:
            seq_data = json.load(f)

        return task_data, seq_data
    
    def __len__(self):
        """ Returns the number of samples in the dataset. """
        return self.n_samples
        

