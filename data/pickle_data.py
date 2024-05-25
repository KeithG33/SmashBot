""" Test file to parse slippi files and extract relevant gamestate information """

from functools import partial
import multiprocessing
import pickle
import sys
import melee
import glob
import shutil
import os
import torch

from melee.enums import Button


def buttons_to_list(button_dict):
    """ Order is [A, B, L, X, Z] """
    button_list = []

    button_list.append(int(button_dict.get(Button.BUTTON_A)))
    button_list.append(int(button_dict.get(Button.BUTTON_B)))
    button_list.append(int(button_dict.get(Button.BUTTON_L) or button_dict.get(Button.BUTTON_R)))
    button_list.append(int(button_dict.get(Button.BUTTON_X) or button_dict.get(Button.BUTTON_Y)))
    button_list.append(int(button_dict.get(Button.BUTTON_Z)))
    
    return button_list


def analog_to_list(main_stick, c_stick, l_shoulder, r_shoulder):
    """ Order is [Main Stick, C-Stick, L_shoulder] """
    # L and R are equivalent so take max. Zero is not pressed
    shoulder = [max(l_shoulder, r_shoulder)]
    sticks = [main_stick[0], main_stick[1], c_stick[0], c_stick[1]]
    analog_list = sticks + shoulder
    return analog_list

def parse_projectiles(projectiles):
    projectile_state_list = []
    for projectile in projectiles:
        projectile_state = [
            projectile.frame, projectile.owner, 
            projectile.position.x, projectile.position.y,
            projectile.speed.x, projectile.speed.y,
            projectile.subtype, projectile.type.value, 
        ]
        projectile_state_list.extend(projectile_state)
    return projectile_state_list

def parse_nana(nana):
    if nana is None: return nana
        
    player_state = [
        nana.action.value, nana.action_frame, nana.character.value,
        int(nana.facing), int(nana.hitlag_left), nana.hitstun_frames_left, 
        nana.invulnerability_left, int(nana.invulnerable), nana.jumps_left,
        nana.nana, int(nana.on_ground), nana.percent, 
        nana.position.x, nana.position.y, nana.shield_strength,
        nana.speed_air_x_self, nana.speed_ground_x_self, nana.speed_x_attack,
        nana.speed_y_attack, nana.speed_y_self, nana.stock
    ]
    return player_state 


def parse_game_state(gamestate):
    """ 
    Get relevant observations from gamestate object (https://libmelee.readthedocs.io/en/latest/gamestate.html)
    
    In no particular order:
    
    1. Environment state info    
        - distance                        (float)
        - frame                           (int)
        - stage                           (enums.Stage)
        - projectile state info:
            - frame                       (int)
            - owner                       (int)
            - position                    (tuple(float,float))
            - speed                       (tuple(float,float))
            - type                        (enums.ProjectileType)
            - subtype                     (int)
    2. player state info (*)              (dict[port, PlayerState])
        - action (animation)              (enum.Action)
        - action_frame                    (int)
        - characters                      (enum.Character)
        - facing                          (bool)
        - position                        (tuple(float,float))
        - shield strength                 (float)
        - ground/air self/attack speed    (float/float float/float)
        - stocks                          (int)
        - hitlag_left                     (bool)
        - hitstun_frames_left             (int)
        - invulnerable                    (bool)
        - invulnerability_left            (int)
        - jumps_left                      (int)
        - nana                            (PlayerState)
        - percent                         (int)
        - on_ground                       (bool)
    3. controller state (*)               (controller.ControllerState)
        - button                          (dict[enums.Button, bool])
        - c_stick                         (tuple(float,float))
        - l_shoulder                      (float)
        - main_stick                      (tuple(float,float))
    4. previous actions **                (list[controller.ControllerState])

    NOTES:
        - asterisk (*) applies to both both players
        - double asterisk (**) means not actually part of gamestate, but is part of model obvservation
        - will use one shoulder L-trigger and one of x or y. This means action space becomes:
            - [A, B, L, X, Z] + [Main Stick, C-Stick]

    """
    # 1. environment state info
    env_info = [gamestate.distance, gamestate.frame, gamestate.stage.value]
    projectiles = parse_projectiles(gamestate.projectiles)
    env_info.append(projectiles)

    # 2. player state info 
    playerstate_list = []
    controllerstate_list = []

    assert len(gamestate.players) == 2, "Only 2 players are supported"

    for port, pstate in gamestate.players.items():
        # Player state
        nana = parse_nana(pstate.nana)

        player_state = [
            pstate.action.value, pstate.action_frame, pstate.character.value,
            int(pstate.facing), int(pstate.hitlag_left), pstate.hitstun_frames_left, 
            pstate.invulnerability_left, int(pstate.invulnerable), pstate.jumps_left,
            nana, int(pstate.on_ground), pstate.percent, 
            pstate.position.x, pstate.position.y, pstate.shield_strength,
            pstate.speed_air_x_self, pstate.speed_ground_x_self, pstate.speed_x_attack,
            pstate.speed_y_attack, pstate.speed_y_self, pstate.stock
        ]

        # Player action
        controller_button_state = buttons_to_list(pstate.controller_state.button)
        controller_analog_state = analog_to_list(
            pstate.controller_state.main_stick,
            pstate.controller_state.c_stick,
            pstate.controller_state.l_shoulder,
            pstate.controller_state.r_shoulder)
        
        controller_state = controller_button_state + controller_analog_state

        playerstate_list.extend(player_state)
        controllerstate_list.append(controller_state)
    
    observation = env_info + playerstate_list
    
    p1_action = controllerstate_list[0] 
    p2_action = controllerstate_list[1]

    return observation, p1_action, p2_action

def save_data(data, output_dir, batch_number):
    """ Save the data to a pickle file with a unique batch number """
    filename = f'{output_dir}/batch_{batch_number}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def process_files(file_batch, output_dir, batch_number):
    game_data_batch = []

    for index, slp_file in enumerate(file_batch):
        print(f"Processing file {batch_number} / {index} :", slp_file)
        console = melee.Console(is_dolphin=False, allow_old_version=True, path=slp_file)
        console.connect()

        previous_p1_action = None
        previous_p2_action = None
        while True:
            gamestate = console.step()
            if gamestate is None:
                break

            obs, p1_action, p2_action = parse_game_state(gamestate)
            game_data_batch.append({
                "observation": obs,
                "p1_action": p1_action,
                "p2_action": p2_action,
                "prev_p1_action": previous_p1_action,
                "prev_p2_action": previous_p2_action
            })
            previous_p1_action = p1_action
            previous_p2_action = p2_action

    save_data(game_data_batch, output_dir, batch_number)


def main():
    SLIPPI_FILE_DIR = '/home/kage/smashbot_workspace/dataset/Slippi_Public_Dataset_v3/slp'
    OUTPUT_DIR = '/home/kage/smashbot_workspace/dataset/pickle_files/'
    NUM_WORKERS = 20  # Number of processes

    slp_files = glob.glob(SLIPPI_FILE_DIR + '**/*.slp', recursive=True)
    batch_size = 50  # Define the batch size per worker
    chunks = [slp_files[i:i + batch_size] for i in range(0, len(slp_files), batch_size)]

    with multiprocessing.Pool(NUM_WORKERS) as pool:
        jobs = [(chunk, OUTPUT_DIR, i + 1) for i, chunk in enumerate(chunks)]
        pool.starmap(process_files, jobs)


if __name__ == "__main__":
    main()
