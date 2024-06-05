from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
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
def generate_input_python(observation, prev_action, player_index):
    """
    Generate the input data from the observation using plain Python lists,
    converting the data structure to only use lists and preparing it for later
    conversion to tensors if needed outside of this function.

    First index indicates whether the data corresponds to player, projectile, nana,
    or misc info. Negative value indicates currently active player.
    """

    copy_observation = copy.deepcopy(observation)
    all_tensors = []

    # misc_info includes distance, frame, and stage
    misc = copy_observation[:3]  # Assuming this slices correctly
    projectiles = copy_observation[3]
    players = copy_observation[4:]  # Assuming players data start from index 4
    nana_states = [obs.pop(9) for obs in players]  # Modify this if structure differs

    # Creating misc tensor data
    misc_types = [MISC_TYPE]
    misc_padded = misc + [0] * (20 - len(misc))
    misc = misc_types + misc_padded
    all_tensors.append(misc)
    
    # Processing players
    players_list = []
    for i, player in enumerate(players):
        player_type = [-PLAYER_TYPE] if i == player_index else [PLAYER_TYPE]
        player_data = player_type + player + [0] * (20 - len(player))
        players_list.append(player_data)
    all_tensors.extend(players_list)

    # Processing Nana states
    nana_list = []
    for i, nana in enumerate(nana_states):
        if nana is not None:
            nana_type = [-NANA_TYPE] if i == player_index else [NANA_TYPE]
            nana_data = nana_type + nana + [0] * (20 - len(nana))
            nana_list.append(nana_data)
    all_tensors.extend(nana_list)

    # Handling projectiles
    if projectiles:
        projectile_list = []
        for projectile in projectiles:
            projectile_type = [PROJECTILE_TYPE]
            projectile_data = projectile_type + projectile + [0] * (20 - len(projectile))
            projectile_list.append(projectile_data)
        all_tensors.extend(projectile_list)
    
    # TODO: experiment conditioning on previous action. Switch this to python lists
    #
    # if prev_action is not None:
    #     prev_action = prev_action + [0] * (20 - len(prev_action))
    #     prev_action = torch.tensor(prev_action).view(1, -1)
    # else:
    #     prev_action = torch.zeros(1, 20)
    # action_types = torch.full((len(prev_action), 1), ACTION_TYPE)
    # action_tensors = torch.cat([action_types, prev_action], dim=1)
    # all_tensors.append(action_tensors)

    return all_tensors


class BucketBatchSampler(Sampler):
    def __init__(self, sequences, batch_size):
        self.batch_size = batch_size
        self.index_and_lengths = [(index, len(sequence)) for index, sequence in enumerate(sequences)]
        self.batches = self._create_batches()
        self.total_batches = len(self.batches)

    def _create_batches(self):
        ''' Create mapping of indices to sequence length so we can batch in our dataset '''

        shuffle(self.index_and_lengths)
        
        length_to_indices_map = defaultdict(list)
        for index, length in self.index_and_lengths:
            length_to_indices_map[length].append(index)

        batched_indices = []
        for indices in length_to_indices_map.values():
            for batch_start in range(0, len(indices), self.batch_size):
                batched_indices.append(indices[batch_start:batch_start + self.batch_size])
        return batched_indices

    def __len__(self):
        return len(self.index_and_lengths)

    def __iter__(self):
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
                player_input = generate_input_python(obs, None, i)
                player_output = actions[i]
                inputs.append(player_input)
                outputs.append(player_output)

        return inputs, outputs
    
    def __len__(self):
        return len(self.outputs)

    def __getitem__(self, index):
        return torch.tensor(self.inputs[index]), torch.tensor(self.outputs[index])

    
def get_memory_usage():
    process = psutil.Process(os.getpid())
    mem_usage = process.memory_info().rss / float(2 ** 20)  # in MB
    return mem_usage

import psutil
if __name__ == "__main__":
    # Test the dataset
    PKL_DIR_TEST = '/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/pickle/test'
    # PKL_DIR_TRAIN = '/home/kage/smashbot_workspace/dataset/pickle_files/train'

    # TODO: test batch_2067.pkl or _2344 or _636

    FILES = [pkl.path for pkl in os.scandir(PKL_DIR_TEST) if pkl.name.endswith(".pkl")]
    print(f"Found {len(FILES)} files")

    for i in range(10):
        dataset = SmashBrosDataset(FILES[:3], num_processes=3)
        print(f"Loop {i+1}, Memory usage: {get_memory_usage()} MB")
        del dataset  # Explicitly delete the dataset object to free up memory



    # with open(FILES[0], 'rb') as f:
    #     data = pickle.load(f)
    #     print(f"Loaded {len(data)} data points from {FILES[0]}")
    # # Test files successfully load
    # # test_data = [pkl.path for pkl in os.scandir(PKL_DIR_TEST) if pkl.name.endswith(".pkl")]
    # # train_data = [pkl.path for pkl in os.scandir(PKL_DIR_TRAIN) if pkl.name.endswith(".pkl")]

    # # testing_data = test_data + train_data
    # # testing_data = [pkl.path for pkl in os.scandir(FILES) if pkl.name.endswith(".pkl")]

    # t1 = time.perf_counter()
    # # dataset = SmashBrosDataset(testing_data[:4], num_processes=4, inner_num_processes=4)9
    # dataset = SmashBrosDataset(FILES, num_processes=3)
    # print(f"Loaded {len(dataset)} data points in {time.perf_counter() - t1:.2f} seconds")
