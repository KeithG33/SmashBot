import os, pickle
import time
import torch
import torch.nn.functional as F

import copy

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

    # print(f"Shape of players: {players.shape}")

    misc_types = torch.full((1, 1), MISC_TYPE)
    misc_padded = F.pad(misc, (0, 20 - misc.shape[1]))
    misc = torch.cat([misc_types, misc_padded], dim=1)
    all_tensors.append(misc)

    # print(f"Shape of misc: {misc.shape}")

    if prev_action is not None:
        prev_action = prev_action + [0] * (20 - len(prev_action))
        prev_action = torch.tensor(prev_action).view(1, -1)
    else:
        prev_action = torch.zeros(1, 20)
    action_types = torch.full((len(prev_action), 1), ACTION_TYPE)
    action_tensors = torch.cat([action_types, prev_action], dim=1)
    all_tensors.append(action_tensors)

    # print(f"Shape of prev action: {action_tensors.shape}")

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
        projectile_tensors = torch.tensor(projectiles)
        projectile_types = torch.full((projectile_tensors.shape[0], 1), PROJECTILE_TYPE)
        projectile_padded = F.pad(projectile_tensors, (0, 20 - projectile_tensors.shape[1]))
        projectiles = torch.cat([projectile_types, projectile_padded], dim=1)
        all_tensors.append(projectiles)

        # print(f"Shape of projectiles: {projectiles.shape}")
    combined_tensor = torch.cat(all_tensors, dim=0)  # (seq, 21)
    return combined_tensor


def pickle_to_tensor(pickle_file, save_dir):
    with open(pickle_file, "rb") as f:
        data = pickle.load(f)

    print(f"Loaded {len(data)} data points from {pickle_file}")

    data_list = []
    t1 = time.perf_counter()
    for d in data:
        obs = d["observation"]
        actions = d["actions"]
        prev_actions = d["prev_actions"]

        num_players = len(actions)

        if prev_actions is None: 
            prev_actions = [None] * num_players
    
        for i in range(num_players):
            player_input = generate_input(copy.deepcopy(obs), prev_actions[i], i)
            player_output = torch.tensor(actions[i])

            data_list.append((player_input, player_output))
    print(f"Processed {len(data_list)} data points in {time.perf_counter() - t1:.2f} seconds")


    


SAMPLE_PICKLE = "/home/kage/smashbot_workspace/dataset/pickle_files/batch_10.pkl"
SAVE_DIR = "/home/kage/smashbot_workspace/dataset/pt_files"
pickle_to_tensor(SAMPLE_PICKLE, SAVE_DIR)