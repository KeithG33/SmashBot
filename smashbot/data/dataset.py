import os
import random
from multiprocessing import Pool




import hickle
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

MISC_TYPE = 1
PROJECTILE_TYPE = 2
PLAYER_TYPE = 3
NANA_TYPE = 4
ACTION_TYPE = 5


def load_pair(file_pair):
    input_file, output_file = file_pair
    input_data = hickle.load(input_file)
    output_data = hickle.load(output_file)
    seq_len = input_data.shape[1]  # Assuming sequence length is the second dimension

    return seq_len, input_data, output_data


class SequenceBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        # self.dataset = dataset
        self.batch_size = batch_size
        self.batches = []

        # Group indices by sequence length
        seq_len_indices = {}
        for index, (seq_len, _) in enumerate(dataset.index_map):
            if seq_len not in seq_len_indices:
                seq_len_indices[seq_len] = []
            seq_len_indices[seq_len].append(index)

        # Create batches for each sequence length group
        for indices in seq_len_indices.values():
            np.random.shuffle(indices)  # Shuffle to randomize batch composition
            # Form batches
            for i in range(0, len(indices), self.batch_size):
                self.batches.append(indices[i:i+self.batch_size])

        np.random.shuffle(self.batches)  # Optional: shuffle all batches to randomize batch order between epochs

    def __iter__(self):
        np.random.shuffle(self.batches)

        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
    

class SmashBrosDataset(Dataset):
    def __init__(self, file_pairs, num_processes=1):
        self.num_processes = num_processes
        self.inputs = {}
        self.outputs = {}
        self.index_map = []  # To store mapping from flat index to (seq_len, batch_index)

        self.load_data(file_pairs)

    def load_data(self, file_pairs):        
        # Use multiprocessing to load data if more than 1 process is requested
        if self.num_processes > 1:
            with Pool(self.num_processes) as p:
                data = p.map(load_pair, file_pairs)
        else:
            data = [load_pair(pair) for pair in file_pairs]
        
        print("Data loaded, organizing...")
        # Organize data by sequence length
        for seq_len, inp, out in data:
            if seq_len not in self.inputs:
                self.inputs[seq_len] = inp
                self.outputs[seq_len] = out
            else:
                self.inputs[seq_len] = np.concatenate([self.inputs[seq_len], inp], axis=0)
                self.outputs[seq_len] = np.concatenate([self.outputs[seq_len], out], axis=0)

        # Create index map
        for seq_len in sorted(self.inputs):
            for batch_index in range(self.inputs[seq_len].shape[0]):
                self.index_map.append((seq_len, batch_index))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, index):
        seq_len, batch_index = self.index_map[index]
        input_tensor = self.inputs[seq_len][batch_index]
        output_tensor = self.outputs[seq_len][batch_index]
        return torch.tensor(input_tensor, dtype=torch.float32), torch.tensor(output_tensor, dtype=torch.float32)

def sample_data(directory, num_samples):
    input_files = [f.path for f in os.scandir(directory) if "inputs" in f.name]
    sampled_input_files = random.sample(input_files, num_samples)
    sampled_output_files = [f.replace("inputs", "outputs") for f in sampled_input_files]
    return list(zip(sampled_input_files, sampled_output_files))


# file_pairs = sample_data('/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/hickle_shuffle/test', 20)
# t1 = time.perf_counter()
# dataset = SmashBrosDataset(file_pairs, 1)
# print(f"Dataset size: {len(dataset)} in {time.perf_counter() - t1:.2f} seconds")

# t2 = time.perf_counter()
# sampler = SequenceBatchSampler(dataset, 5)
# print(f"Number of batches: {len(sampler)} in {time.perf_counter() - t2:.2f} seconds")
# t3 = time.perf_counter()
# loader = DataLoader(dataset, batch_sampler=sampler)
# print(f"DataLoader created in {t3 - time.perf_counter()} seconds")
# for i, batch in enumerate(loader):
#     print(batch[0].shape, batch[1].shape)
#     if i == 200: # Print only first 10 batches
#         break