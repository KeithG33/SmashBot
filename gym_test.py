import numpy as np
from smashbot.model.smash_transformer import SmashTransformer
from smashbot.data.extract_dataset import generate_input_python, parse_game_state

import torch

import melee

import gymnasium as gym 

INDEX_TO_DIGITAL = {
    0: melee.Button.BUTTON_A,
    1: melee.Button.BUTTON_B,
    2: melee.Button.BUTTON_L,
    3: melee.Button.BUTTON_X,
    4: melee.Button.BUTTON_Z
}


# SLIPPI_ONLINE_PATH = '/home/kage/smashbot_workspace/Slippi_Online-x86_64.AppImage'

# CPU_PORT = 1
# HUMAN_PORT = 2

# console = melee.Console(path=SLIPPI_ONLINE_PATH,
#                         is_dolphin=True)

# controller_cpu = melee.Controller(console=console, port=CPU_PORT)
# controller_human = melee.Controller(console=console,
#                                     port=HUMAN_PORT,
#                                     type=melee.ControllerType.GCN_ADAPTER)

# console.run()
# console.connect()

# controller_cpu.connect()
# controller_human.connect()



def gamestate_to_model_input(gamestate, player_port_index):
    """
    Converts gamestate to model input. 
    """
    obs, _, _ = parse_game_state(gamestate, in_game=True)
    player_input = generate_input_python(obs, None, player_port_index)
    player_input_tensor = torch.tensor(player_input, dtype=torch.float32).unsqueeze(0)

    return player_input_tensor


def check_game_over(gamestate):
    # check player stocks 
    for port, player in gamestate.players.items():
        if player.stock == 0:
            return True, port
    return False, None

def apply_action(controller, digital, analog):
    # Press the buttons
    for button in melee.enums.Button:
        if button in digital:
            controller.press_button(button)
            continue
        controller.release_button(button)

    # Control the sticks
    ax, ay, cx, cy, l = analog
    controller.tilt_analog(melee.Button.BUTTON_MAIN, ax, ay)
    controller.tilt_analog(melee.Button.BUTTON_C, cx, cy)
    controller.press_shoulder(melee.Button.BUTTON_L, l)
    


def main():
    NUM_EPI = 10

    # Load model
    model = SmashTransformer(action_dim=10, embed_dim=224, model_dim=504, nhead=24, num_layers=6)
    model.load_state_dict(torch.load('/home/kage/smashbot_workspace/SmashBotTransformer_embed224_model504_layer6.pt'))

    env = gym.make('melee-test-v0', slippi_path='/home/kage/smashbot_workspace/Slippi_Online-x86_64.AppImage')
    num_episode = 0
    while num_episode < NUM_EPI:
        obs = env.reset()
        done = False

        print(f"Starting observation: {obs}")

        while not done:
            # The console object keeps track of how long your bot is taking to process frames
            #   And can warn you if it's taking too long
            # if env.console.processingtime * 1000 > 12:
            #     print("WARNING: Last frame took " + str(env.console.processingtime*1000) + "ms to process.")

            # Returns digital as multi-hot encoding
            digital, analog = model.get_action(obs)

            obs, reward, done, _ = env.step({"digital": digital, "analog": analog})

            if reward != (0,0):
                print(f"Reward is {reward} and done is {done}")


main()