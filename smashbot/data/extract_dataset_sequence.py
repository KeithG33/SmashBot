""" Test file to parse slippi files and extract relevant gamestate information """

from collections import deque
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


FILE_SIZE = 1_000_000
SEQ_LEN = 10
OBS_DIM = 10 + 2 + 20 + 20 # 10 actions, 2 misc, 20 player1, 20 player2

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


def parse_game_state(gamestate, in_game=False):
    """
    Get relevant observations from gamestate object (https://libmelee.readthedocs.io/en/latest/gamestate.html)
    """
    # 1. environment state info
    env_info = [gamestate.distance, gamestate.stage.value]
    
    # 2. player state info
    playerstate_list = []
    controllerstate_list = [] 

    for port, pstate in gamestate.players.items():
        # player_state = [
        #     pstate.action.value,          pstate.action_frame,
        #     pstate.character.value,       int(pstate.facing),
        #     int(pstate.hitlag_left),    pstate.hitstun_frames_left,
        #     pstate.invulnerability_left,  int(pstate.invulnerable),
        #     pstate.jumps_left,            int(pstate.on_ground),
        #     pstate.percent,               pstate.position.x,
        #     pstate.position.y,            pstate.shield_strength,
        #     pstate.speed_air_x_self,      pstate.speed_ground_x_self,
        #     pstate.speed_x_attack,        pstate.speed_y_attack,
        #     pstate.speed_y_self,          pstate.stock,
        # ]

        # NOTE: testing max/min x/y for some data gave {min/max}_x = -272/272, {min/max}_y = -150/350.
        #       
        player_state = [
            pstate.action.value,
            pstate.action_frame,
            pstate.character.value,
            int(pstate.facing),
            int(pstate.hitlag_left), 
            pstate.hitstun_frames_left,
            pstate.invulnerability_left, 
            int(pstate.invulnerable),
            pstate.jumps_left / 5,
            int(pstate.on_ground),
            pstate.percent / 300,    
            pstate.position.x / (272*2),
            pstate.position.y / (550),     
            pstate.shield_strength / 60,
            pstate.speed_air_x_self,   
            pstate.speed_ground_x_self,
            pstate.speed_x_attack,    
            pstate.speed_y_attack,
            pstate.speed_y_self,       
            pstate.stock / 4,
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

    # observation is [distance, stage, player1, player2]
    # actions is [player1, player2]
    return observation, actions

def generate_inputs(observation, prev_action):
    misc_info = observation[:2]
    
    p1_info = observation[2]
    p2_info = observation[3]

    # save and maybe use as flat input array
    player1_obs = prev_action[0] + p1_info + p2_info + misc_info
    player2_obs = prev_action[1] + p2_info + p1_info + misc_info

    return player1_obs, player2_obs

def process_files(file_batch, output_dir, batch_number):
    # Add variables/data structures to store data. Lightweight best 
    data_list = []
    file_counter = 0

    init_obs = [-1] * OBS_DIM

    try:
        for index, slp_file in enumerate(file_batch):
            print(f"Processing batch {batch_number} / file {index}:", slp_file)

            try:
                console = melee.Console(is_dolphin=False, allow_old_version=False, path=slp_file)
                console.connect()
                gamestate = console.step()
            except Exception as e:
                if type(e).__name__ == "SlippiVersionTooLow":
                    console = melee.Console(is_dolphin=False, allow_old_version=True, path=slp_file)
                    console.connect()
                    gamestate = console.step()
            
            game_buffer_p1 = deque([init_obs] * SEQ_LEN, maxlen=SEQ_LEN)
            game_buffer_p2 = deque([init_obs] * SEQ_LEN, maxlen=SEQ_LEN)

            prev_action1 = [-1] * 10
            prev_action2 = [-1] * 10
            prev_actions = [prev_action1, prev_action2]

            while gamestate := console.step():
                if len(gamestate.players) > 2:
                    print("Skipping game with more than 2 players")
                    break

                characters = [player.character for player in gamestate.players.values()]
                if melee.enums.Character.POPO in characters or melee.enums.Character.NANA in characters:
                    print("Skipping Ice Climbers game")
                    break

                obs, actions = parse_game_state(gamestate)
                p1_obs, p2_obs = generate_inputs(obs, prev_actions)
               
                # Update buffers with new observations
                game_buffer_p1.append(p1_obs)
                game_buffer_p2.append(p2_obs)
                
                p1_data = (np.array(game_buffer_p1), actions[0])
                p2_data = (np.array(game_buffer_p2), actions[1])

                data_list.append(p1_data)
                data_list.append(p2_data)

                prev_actions = actions

                if len(data_list) >= FILE_SIZE:
                    save_as_hickle(data_list, output_dir, batch_number, file_counter)
                    file_counter += 1   
                    data_list = []
                

    except Exception as e:
        print(f"An error occurred while processing file {slp_file}: {e}")
    else:
        # Convert and save any remaining data after processing
        if data_list:
            save_as_hickle(data_list, output_dir, batch_number, file_counter)
            file_counter += 1
            data_list = []


def save_as_hickle(data, output_dir, batch_number, file_index):
    # Array of inputs (B,S,2+21+21) and outputs (B,10)
    inputs, outputs = map(np.asarray, zip(*data))

    hkl.dump(
        inputs,
        f"{output_dir}/inputs_{batch_number}-{file_index}.hkl",
        compression="gzip",
        mode="w",
    )
    hkl.dump(
        outputs,
        f"{output_dir}/outputs_{batch_number}-{file_index}.hkl",
        compression="gzip",
        mode="w",
    )
    print(
        f"Saved file {batch_number}/{file_index}"
    )

def extract_dataset(slp_dir, output_dir, num_workers=32, chunk_size=100):
    slp_files = glob.glob(slp_dir + "**/*.slp", recursive=True)
    random.shuffle(slp_files)
    chunks = [
        slp_files[i : i + chunk_size] for i in range(0, len(slp_files), chunk_size)
    ]

    os.makedirs(output_dir, exist_ok=True)

    if num_workers == 1:
        for i, chunk in enumerate(chunks):
            process_files(chunk, output_dir, i + 1)
    else: 
        with multiprocessing.Pool(num_workers) as pool:
            jobs = [(chunk, output_dir, i + 1) for i, chunk in enumerate(chunks)]
            pool.starmap(process_files, jobs)


def main():
    SLIPPI_FILE_DIR = (
        "/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/slp"
    )
    OUTPUT_DIR = "/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/hickle"
    NUM_WORKERS = 32
    CHUNK_SIZE = 100  # Original batch size doubled

    slp_files = glob.glob(SLIPPI_FILE_DIR + "**/*.slp", recursive=True)
    random.shuffle(slp_files)
    
    chunks = [
        slp_files[i : i + CHUNK_SIZE] for i in range(0, len(slp_files), CHUNK_SIZE)
    ]

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        jobs = [(chunk, OUTPUT_DIR, i + 1) for i, chunk in enumerate(chunks)]
        pool.starmap(process_files, jobs)


if __name__ == "__main__":
    # main()

    SLIPPI_DIR = "/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/slp"
    # SLIPPI_DIR = "/home/kage/smashbot_workspace/dataset/SlippiGames/slp"
    OUTPUT_DIR = "/home/kage/smashbot_workspace/dataset/hickle_sequence"
    num_workers = 15
    chunk_size = 100

    extract_dataset(SLIPPI_DIR, OUTPUT_DIR, num_workers, chunk_size)
