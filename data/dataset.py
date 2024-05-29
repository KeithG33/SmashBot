from multiprocessing import Pool
import os
import pickle
import time
import torch
from torch.utils.data import Dataset
import copy
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from torch.utils.data import Sampler
from random import shuffle
from collections import OrderedDict, defaultdict


MISC_TYPE = 1
PROJECTILE_TYPE = 2
PLAYER_TYPE = 3
NANA_TYPE = 4
ACTION_TYPE = 5

def generate_input_combined(observation, prev_action, player_index):
    """
    Generate the input tensor from the observation. Create a sequence of
    features to handle the variable sized number of players and projectiles

    First index indicates whether tensor corresponds to player, projectile, nana,
    or misc info. Negative value indicates currently active player.

    Also includes prev action in observation 
    """

    MISC_TYPE = 1
    PROJECTILE_TYPE = 2
    PLAYER_TYPE = 3
    NANA_TYPE = 4
    ACTION_TYPE = 5
    
    all_tensors = []

    # misc_info includes distance, frame, and stage
    # players is a list of lists of length 20
    # nanas is a list of lists of length 20
    # projectiles is a list of lists of length 8
    misc = torch.tensor(observation[:3]).view(1, 3)
    players = observation[4:]  # 2 to 4 players
    projectiles = observation[3]
    nana_states = [obs.pop(9) for obs in players]

    players = torch.tensor(players)
    player_types = torch.full((players.shape[0], 1), PLAYER_TYPE)
    player_types[player_index] = -PLAYER_TYPE  # Neg value for active player
    players = torch.cat([player_types, players], dim=1)
    all_tensors.append(players)

    nana_list = []
    for i, nana in enumerate(nana_states):
        if nana is not None:
            nana_tensor = torch.tensor(nana).view(1, -1)
            nana_type = -NANA_TYPE if i == player_index else NANA_TYPE
            nana_types = torch.full((1, 1), nana_type)
            nana_tensor = torch.cat([nana_types, nana_tensor], dim=1)
            nana_list.append(nana_tensor)
    if nana_list: 
        nana_tensors = torch.cat(nana_list, dim=0)
        all_tensors.append(nana_tensors)

    misc_types = torch.full((1, 1), MISC_TYPE)
    misc_padded = F.pad(misc, (0, 20 - misc.shape[1]))
    misc = torch.cat([misc_types, misc_padded], dim=1)
    all_tensors.append(misc)

    # TODO: experiment conditioning on previous action
    #
    # if prev_action is not None:
    #     prev_action = prev_action + [0] * (20 - len(prev_action))
    #     prev_action = torch.tensor(prev_action).view(1, -1)
    # else:
    #     prev_action = torch.zeros(1, 20)
    # action_types = torch.full((len(prev_action), 1), ACTION_TYPE)
    # action_tensors = torch.cat([action_types, prev_action], dim=1)
    # all_tensors.append(action_tensors)

    if projectiles:
        projectile_tensors = torch.tensor(projectiles)
        projectile_types = torch.full((projectile_tensors.shape[0], 1), PROJECTILE_TYPE)
        projectile_padded = F.pad(projectile_tensors, (0, 20 - projectile_tensors.shape[1]))
        projectiles = torch.cat([projectile_types, projectile_padded], dim=1)
        all_tensors.append(projectiles)
        
    combined_tensor = torch.cat(all_tensors, dim=0)  # (seq, 21)
    return combined_tensor



class BucketBatchSampler(Sampler):
    def __init__(self, sequences, batch_size):
        self.batch_size = batch_size
        self.index_and_lengths = [(index, sequence.shape[0]) for index, sequence in enumerate(sequences)]
        self.batches = self._create_batches()
        self.total_batches = len(self.batches)

    def _create_batches(self):
        # Shuffle index and lengths to randomize bucket grouping
        shuffle(self.index_and_lengths)
        
        # Organize sequences by their length for bucketing
        length_to_indices_map = defaultdict(list)
        for index, length in self.index_and_lengths:
            length_to_indices_map[length].append(index)

        # Create batches from the length-sorted index list
        batched_indices = []
        for indices in length_to_indices_map.values():
            # Creating batches within each bucket
            for batch_start in range(0, len(indices), self.batch_size):
                batched_indices.append(indices[batch_start:batch_start + self.batch_size])
        return batched_indices

    def __len__(self):
        return len(self.index_and_lengths)

    def __iter__(self):
        # Shuffle batches to ensure model does not learn any order
        shuffle(self.batches)
        for batch in self.batches:
            yield batch

class SmashBrosDataset(Dataset):
    def __init__(self, files, num_processes=1):
        self.num_processes = num_processes
        self.inputs, self.outputs = self.load_data(files)
        
    def load_data(self, files):
        """Load all pickle files from the provided files"""
        data = []
        if self.num_processes > 1:    
            with Pool(processes=self.num_processes) as pool:
                results = pool.map(self.load_file, files)

            for result in results:
                data.extend(result)
            return data
        
        for file in files:
            res = self.load_file(file)
            data.extend(res)
        return data
    
    def load_file(self, file_path):
        """Load a single file and process its content."""
        inputs, outputs = [], []
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        for d in data:
            obs = d["observation"]
            actions = d["actions"]
            prev_actions = d["prev_actions"]

            if prev_actions is None:
                prev_actions = [None] * len(actions)

            for i in range(len(actions)):
                player_input = generate_input_combined(copy.deepcopy(obs), prev_actions[i], i)
                player_output = torch.tensor(actions[i])
                inputs.append(player_input)
                outputs.append(player_output)

        return inputs, outputs
    
    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]