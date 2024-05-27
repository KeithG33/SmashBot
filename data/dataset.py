from multiprocessing import Pool
import os
import pickle
import torch
from torch.utils.data import Dataset
import copy
import torch.nn.functional as F

def generate_input(observation, prev_action, player_index):
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

    print(f"Shape of players: {players.shape}")

    misc_types = torch.full((1, 1), MISC_TYPE)
    misc_padded = F.pad(misc, (0, 20 - misc.shape[1]))
    misc = torch.cat([misc_types, misc_padded], dim=1)
    all_tensors.append(misc)

    print(f"Shape of misc: {misc.shape}")

    if prev_action is not None:
        prev_action = prev_action + [0] * (20 - len(prev_action))
        prev_action = torch.tensor(prev_action).view(1, -1)
    else:
        prev_action = torch.zeros(1, 20)
    action_types = torch.full((len(prev_action), 1), ACTION_TYPE)
    action_tensors = torch.cat([action_types, prev_action], dim=1)
    all_tensors.append(action_tensors)

    print(f"Shape of prev action: {action_tensors.shape}")

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

        print(f"Shape of nanas: {nana_tensors.shape}")

    if projectiles:
        print(projectiles)
        projectile_tensors = torch.tensor(projectiles)
        projectile_types = torch.full((projectile_tensors.shape[0], 1), PROJECTILE_TYPE)
        projectile_padded = F.pad(projectile_tensors, (0, 20 - projectile_tensors.shape[1]))
        projectiles = torch.cat([projectile_types, projectile_padded], dim=1)
        all_tensors.append(projectiles)

        print(f"Shape of projectiles: {projectiles.shape}")
    combined_tensor = torch.cat(all_tensors, dim=0)  # (seq, 21)
    return combined_tensor


class SmashBrosDataset(Dataset):
    def __init__(self, files, num_processes=1):
        self.data = []
        self.load_data(files)
        self.num_processes = num_processes


    def load_data(self, files):
        """Load all pickle files from the provided files using multiprocessing."""
        # Set the number of processes to the number of available CPUs
        with Pool(processes=self.num_processes) as pool:
            results = pool.map(self.load_file, files)

        # Flatten the list of lists returned by pool.map
        for result in results:
            self.data.extend(result)
    
    def load_file(self, file_path):
        """Load a single file and process its content."""
        temp_data = []
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        for d in data:
            obs = d["observation"]
            actions = d["actions"]
            prev_actions = d["prev_actions"]

            if prev_actions is None:
                prev_actions = [None] * len(actions)

            for i in range(len(actions)):
                player_input = generate_input(copy.deepcopy(obs), prev_actions[i], i)
                player_output = torch.tensor(actions[i])
                temp_data.append((player_input, player_output))
        return temp_data
    
    def __len__(self):
        """Denotes the total number of samples."""
        return len(self.data)

    def __getitem__(self, index):
        """Generates one sample of data."""
        return self.data[index]