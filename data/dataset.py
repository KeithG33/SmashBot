from collections import defaultdict
from multiprocessing import Pool
import multiprocessing
import pickle, os
from random import shuffle
import copy
import time


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler


MISC_TYPE = 1
PROJECTILE_TYPE = 2
PLAYER_TYPE = 3
NANA_TYPE = 4
ACTION_TYPE = 5

# TODO: the only difference here is the player_index parts
#       Only need to call this once and then manipulate the player indices
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
    
    misc_types = torch.full((1, 1), MISC_TYPE)
    misc_padded = F.pad(misc, (0, 20 - misc.shape[1]))
    misc = torch.cat([misc_types, misc_padded], dim=1)
    all_tensors.append(misc)
    
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
        """Load all pickle files from the provided files and return inputs and outputs."""
        inputs, outputs = [], []

        if self.num_processes > 1:
            with Pool(processes=self.num_processes) as pool:
                results = pool.map(self.load_file, files)
        else:
            results = map(self.load_file, files)

        for file_inputs, file_outputs in results:
            inputs.extend(file_inputs)
            outputs.extend(file_outputs)

        return inputs, outputs
    
    def load_file(self, file_path):
        """Load a single file and process its content."""
        inputs, outputs = [], []
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        print(f"Loading {file_path}...")

        for d in data:
            obs = d["observation"]
            actions = d["actions"]

            for i in range(len(actions)):
                player_input = generate_input_combined(copy.deepcopy(obs), None, i)
                player_output = torch.tensor(actions[i])
                inputs.append(player_input)
                outputs.append(player_output)

        return inputs, outputs
    
    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


if __name__ == "__main__":
    # Test the dataset
    # TODO fix multiprocessing: currently OSError: too many open files :(
    PKL_DIR_TEST = '/home/kage/smashbot_workspace/dataset/pickle_files/test'
    PKL_DIR_TRAIN = '/home/kage/smashbot_workspace/dataset/pickle_files/train'

    # Test files successfully load
    test_data = [pkl.path for pkl in os.scandir(PKL_DIR_TEST) if pkl.name.endswith(".pkl")]
    train_data = [pkl.path for pkl in os.scandir(PKL_DIR_TRAIN) if pkl.name.endswith(".pkl")]

    testing_data = test_data + train_data

    t1 = time.perf_counter()
    dataset = SmashBrosDataset(testing_data[:4], num_processes=1)
    print(f"Loaded {len(dataset)} data points in {time.perf_counter() - t1:.2f} seconds")
    # for pkl_file in testing_data:
    #     t1 = time.perf_counter()
    #     print(f"Loading {pkl_file}...")
    #     dataset = SmashBrosDataset([pkl_file], num_processes=2)
    #     print(f"Loaded {pkl_file} in {time.perf_counter() - t1:.2f} seconds")
