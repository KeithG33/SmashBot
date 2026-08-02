import os
import random
from multiprocessing import Pool

import hickle
import numpy as np
import torch
from torch.utils.data import Dataset


class SmashBrosDataset(Dataset):
    def __init__(self, file_pairs, num_processes=1):
        self.num_processes = num_processes
        self.inputs = []
        self.outputs = []

        self.load_data(file_pairs)

    def load_data(self, file_pairs):
        for inp, out in file_pairs:
            inp = hickle.load(inp)
            out = hickle.load(out)
            self.inputs.append(inp)
            self.outputs.append(out)
            
        # Concatenate all data
        self.inputs = np.concatenate(self.inputs, axis=0)
        self.outputs = np.concatenate(self.outputs, axis=0)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return torch.as_tensor(self.inputs[index], dtype=torch.float32), torch.as_tensor(self.outputs[index], dtype=torch.float32)


def sample_data(directory, num_samples):
    input_files = [f.path for f in os.scandir(directory) if "inputs" in f.name]
    sampled_input_files = random.sample(input_files, num_samples)
    sampled_output_files = [f.replace("inputs", "outputs") for f in sampled_input_files]
    return list(zip(sampled_input_files, sampled_output_files))

