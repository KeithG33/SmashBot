""" Test file to parse slippi files and extract relevant gamestate information """

import copy
from functools import partial
import multiprocessing
import pickle
import hickle as hkl
import random
import sys
from typing import LiteralString

import numpy as np
import melee
import glob
import shutil
import os
import torch

from melee.enums import Button


# Enums for indicating data type
MISC_TYPE = 1
PROJECTILE_TYPE = 2
PLAYER_TYPE = 3
NANA_TYPE = 4
ACTION_TYPE = 5


def buttons_to_list(button_dict):
    """Order is [A, B, L, X, Z]"""
    button_list = []
    button_list.append(int(button_dict.get(Button.BUTTON_A)))
    button_list.append(int(button_dict.get(Button.BUTTON_B)))
    button_list.append(
        int(button_dict.get(Button.BUTTON_L) or button_dict.get(Button.BUTTON_R))
    )
    button_list.append(
        int(button_dict.get(Button.BUTTON_X) or button_dict.get(Button.BUTTON_Y))
    )
    button_list.append(int(button_dict.get(Button.BUTTON_Z)))
    return button_list


def analog_to_list(main_stick, c_stick, l_shoulder, r_shoulder):
    """Order is [Main Stick, C-Stick, L_shoulder]"""
    # L and R are equivalent so take max. Zero is not pressed
    shoulder = [max(l_shoulder, r_shoulder)]
    sticks = [main_stick[0], main_stick[1], c_stick[0], c_stick[1]]
    analog_list = sticks + shoulder
    return analog_list


def parse_projectiles(projectiles):
    projectile_state_list = []
    for projectile in projectiles:
        projectile_state = [
            projectile.frame,
            projectile.owner,
            projectile.position.x,
            projectile.position.y,
            projectile.speed.x,
            projectile.speed.y,
            projectile.subtype,
            projectile.type.value,
        ]
        projectile_state_list.append(projectile_state)
    return projectile_state_list


def parse_nana(nana):
    if nana is None:
        return nana

    player_state = [
        nana.action.value,          nana.action_frame,
        nana.character.value,       int(nana.facing),
        int(nana.hitlag_left),    
        nana.hitstun_frames_left,   nana.invulnerability_left,
        int(nana.invulnerable),   nana.jumps_left,
                                    int(nana.on_ground), # no nana
        nana.percent,               nana.position.x,
        nana.position.y,            nana.shield_strength, 
        nana.speed_air_x_self,      nana.speed_ground_x_self,
        nana.speed_x_attack,        nana.speed_y_attack,     
        nana.speed_y_self,          nana.stock,
    ]
    return player_state


def parse_game_state(gamestate, in_game=False):
    """
    Get relevant observations from gamestate object (https://libmelee.readthedocs.io/en/latest/gamestate.html)
    """
    # 1. environment state info
    env_info = [gamestate.distance, gamestate.frame, gamestate.stage.value]
    projectiles = parse_projectiles(gamestate.projectiles)
    env_info.append(projectiles)

    # 2. player state info
    playerstate_list = []
    controllerstate_list = []

    for port, pstate in gamestate.players.items():
        # Player state
        # nana = parse_nana(pstate.nana)

        player_state = [
            pstate.action.value,          pstate.action_frame,
            pstate.character.value,       int(pstate.facing),
            int(pstate.hitlag_left), 
            pstate.hitstun_frames_left,   pstate.invulnerability_left,
            int(pstate.invulnerable),   pstate.jumps_left,
            # nana,                         int(pstate.on_ground),
                                          int(pstate.on_ground),
            pstate.percent,               pstate.position.x,
            pstate.position.y,            pstate.shield_strength,
            pstate.speed_air_x_self,      pstate.speed_ground_x_self,
            pstate.speed_x_attack,        pstate.speed_y_attack,
            pstate.speed_y_self,          pstate.stock,
        ]

        if not in_game:
            # Player action
            controller_button_state = buttons_to_list(pstate.controller_state.button)
            controller_analog_state = analog_to_list(
                pstate.controller_state.main_stick,
                pstate.controller_state.c_stick,
                pstate.controller_state.l_shoulder,
                pstate.controller_state.r_shoulder,
            )

            controller_state = controller_button_state + controller_analog_state
            controllerstate_list.append(controller_state)

        playerstate_list.append(player_state)

    observation = env_info + playerstate_list
    actions = controllerstate_list

    return observation, actions


def generate_input_python(observation, prev_action, player_index):
    """
    Generate the input data from the observation. Uses plain python lists instead of arrays
    or tensors to limit memory usage during multiprocessing.

    Output of function can be converted to tensors or arrays directly with torch.tensor()
    or np.array(). Will have shape (S,21), where S depends on number of players, projectiles, nanas, etc

    First index indicates whether the data corresponds to player, projectile, nana,
    or misc info. Negative value indicates currently active player.
    """

    # copy_observation = copy.deepcopy(observation)
    all_tensors = []

    misc = observation[:3] # distance, frame, stage
    projectiles = observation[3]
    players = observation[4:]
    # nana_states = [obs.pop(9) for obs in players]

    if prev_action is None:
        prev_action = [-1] * 10
    
    # Creating misc tensor data
    misc += prev_action
    misc = [MISC_TYPE] + misc + [0] * (20 - len(misc))
    all_tensors.append(misc)

    # Processing players
    players_list = []
    for i, player in enumerate(players):
        player_type = [-PLAYER_TYPE] if i == player_index else [PLAYER_TYPE]
        player_data = player_type + player + [0] * (20 - len(player))
        players_list.append(player_data)
    all_tensors.extend(players_list)

    # Handling projectiles
    if projectiles:
        projectile_list = []
        for projectile in projectiles:
            projectile_data = (
                [PROJECTILE_TYPE] + projectile + [0] * (20 - len(projectile))
            )
            projectile_list.append(projectile_data)
        all_tensors.extend(projectile_list)

    return all_tensors


def process_files(file_batch, output_dir, batch_number):
    grouped_data = {}
    file_counters = {}  # Track files saved for each input_length
    try:

        for index, slp_file in enumerate(file_batch):
            print(f"Processing batch {batch_number} / file {index}:", slp_file)
            console = melee.Console(is_dolphin=False, allow_old_version=True, path=slp_file)
            console.connect()

            prev_actions = [None, None]
            
            while gamestate := console.step():
                if len(gamestate.players) > 2:
                    print("Skipping game with more than 2 players")
                    break

                characters = [player.character for player in gamestate.players.values()]
                if melee.enums.Character.POPO in characters or melee.enums.Character.NANA in characters:
                    print("Skipping Ice Climbers game")
                    break

                obs, actions = parse_game_state(gamestate)

                # Generate input and group by length
                for i in range(len(actions)):
                    player_input = generate_input_python(obs, prev_actions[i], i)
                    player_output = actions[i]
                    input_length = len(player_input)

                    if input_length not in grouped_data:
                        grouped_data[input_length] = []
                        file_counters[input_length] = 0

                    grouped_data[input_length].append((player_input, player_output))

                    if len(grouped_data[input_length]) >= 2_000_000:
                        save_as_hickle(
                            grouped_data[input_length],
                            input_length,
                            output_dir,
                            batch_number,
                            file_counters[input_length],
                        )
                        grouped_data[input_length] = []
                        file_counters[input_length] += 1

                    
                    prev_actions[i] = player_output

    except Exception as e:
        print(f"An error occurred while processing file {slp_file}: {e}")
    else:
        # Convert and save any remaining data after processing
        for length, data in grouped_data.items():
            if data:
                save_as_hickle(data, length, output_dir, batch_number, file_counters[length])
                file_counters[length] += 1


def save_as_hickle(data, input_length, output_dir, batch_number, file_index):
    # Array of inputs (B,S,21) and outputs (B,10)
    inputs, outputs = map(np.asarray, zip(*data))

    hkl.dump(
        inputs,
        f"{output_dir}/inputs{input_length}_batch{batch_number}_{file_index}.hkl",
        compression="gzip",
        mode="w",
    )
    hkl.dump(
        outputs,
        f"{output_dir}/outputs{input_length}_batch{batch_number}_{file_index}.hkl",
        compression="gzip",
        mode="w",
    )
    print(
        f"Saved {output_dir}/inputs{input_length}_batch{batch_number}_{file_index}.hkl"
    )


def extract_dataset(slp_dir, output_dir, num_workers=32, chunk_size=100):
    slp_files = glob.glob(slp_dir + "**/*.slp", recursive=True)
    random.shuffle(slp_files)
    chunks = [
        slp_files[i : i + chunk_size] for i in range(0, len(slp_files), chunk_size)
    ]

    if num_workers == 1:
        for i, chunk in enumerate(chunks):
            process_files(chunk, output_dir, i + 1)
    else:
        with multiprocessing.Pool(num_workers) as pool:
            jobs = [(chunk, output_dir, i + 1) for i, chunk in enumerate(chunks)]
            pool.starmap(process_files, jobs)


# def main():
#     SLIPPI_FILE_DIR = (
#         "/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/slp"
#     )
#     OUTPUT_DIR = "/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/hickle"
#     NUM_WORKERS = 32
#     CHUNK_SIZE = 100  # Original batch size doubled

#     slp_files = glob.glob(SLIPPI_FILE_DIR + "**/*.slp", recursive=True)
#     random.shuffle(slp_files)
#     chunks = [
#         slp_files[i : i + CHUNK_SIZE] for i in range(0, len(slp_files), CHUNK_SIZE)
#     ]

#     with multiprocessing.Pool(NUM_WORKERS) as pool:
#         jobs = [(chunk, OUTPUT_DIR, i + 1) for i, chunk in enumerate(chunks)]
#         pool.starmap(process_files, jobs)


if __name__ == "__main__":
    # main()

    # Path to directory containing the Slippi files
    SLIPPI_DIR = '/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/slp'
    OUTPUT_DIR = '/home/kage/smashbot_workspace/dataset/hickle_action'
    num_workers = 20
    chunk_size = 85

    extract_dataset(SLIPPI_DIR, OUTPUT_DIR, num_workers, chunk_size)