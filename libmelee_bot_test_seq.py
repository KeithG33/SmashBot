from collections import deque
import os
import numpy as np
from smashbot.model.smash_transformer_sequence import SmashTransformer
from smashbot.data.extract_dataset_sequence import generate_inputs, parse_game_state

import torch

import melee
from ppo import PPO

INDEX_TO_DIGITAL = {
    0: melee.Button.BUTTON_A,
    1: melee.Button.BUTTON_B,
    2: melee.Button.BUTTON_L,
    3: melee.Button.BUTTON_X,
    4: melee.Button.BUTTON_Z
}


SLIPPI_ONLINE_PATH = '/home/kage/smashbot_workspace/Slippi_Online-x86_64.AppImage'

CPU_PORT = 1
HUMAN_PORT = 2

console = melee.Console(path=SLIPPI_ONLINE_PATH,
                        allow_old_version=True,
                        save_replays=False,
                        fullscreen=False,
                        replay_dir="~/smashbot_workspace/replays",
                        blocking_input=True,
                        is_dolphin=True)

controller_cpu = melee.Controller(console=console, port=CPU_PORT)
# controller_cpu2 = melee.Controller(console=console,
#                                     port=HUMAN_PORT,
#                                     type=melee.ControllerType.GCN_ADAPTER,
#                                    )
controller_cpu2 = melee.Controller(console=console,
                                    port=HUMAN_PORT
                                   )

console.run()
console.connect()

controller_cpu.connect()
controller_cpu2.connect()



def get_digital_action(model_output):
    """ 
    Recall the model_output is [A, B, L/R, X/Y, Z] + [Ax, Ay, Cx, Cy, L/R]

    Returns list of top two digital button enums (melee.Button.BUTTON_A, etc.)
    that have values greater than 0.5.
    """
    
    digital = model_output[:, :5].cpu().numpy()  # Get the first five digital outputs
    valid_indices = np.where(digital > 0.5)  # Find indices where outputs are greater than 0.5
    print(f"Valid indices: {valid_indices}")
    # Filter the digital outputs and their indices to those above the threshold
    valid_values = digital[valid_indices]
    valid_buttons = valid_indices[1]  # Get the button indices from the second element of valid_indices
    
    # Sort the indices by the corresponding values in descending order
    sorted_indices = valid_buttons[np.argsort(-valid_values)]
    
    # Return the top two button indices, if more than two buttons are above 0.5
    return [INDEX_TO_DIGITAL[idx] for idx in sorted_indices[:2]]


def gamestate_to_model_input(gamestate, prev_action, player1=True, in_game=True):
    """
    Converts gamestate to model input. 
    """
    obs, actions = parse_game_state(gamestate, in_game=in_game)
    p1_input, p2_input = generate_inputs(obs, prev_action)

    return p1_input, p2_input, actions


def check_game_over(gamestate):
    # check player stocks 
    for port, player in gamestate.players.items():
        if player.stock == 0:
            return True, port


model = SmashTransformer(action_dim=10, embed_dim=112, model_dim=384, nhead=24, num_layers=5, dropout=0.02)
model = PPO(feature_extractor=model)
model = model.cuda()

checkpoint = '/home/kage/smashbot_workspace/SmashBotTransformer-seq_fix.pt'
if os.path.exists(checkpoint):
    model.load_state_dict(torch.load(checkpoint))
    print("Loaded model weights from previous training session")
model.eval()


OBS_DIM = 10 + 2 + 20 + 20 # 10 actions, 2 misc, 20 player1, 20 player2
init_obs = [-1] * OBS_DIM
SEQ_LEN = 10
game_buffer_p1 = deque([init_obs] * SEQ_LEN, maxlen=SEQ_LEN)
game_buffer_p2 = deque([init_obs] * SEQ_LEN, maxlen=SEQ_LEN)

prev_action1 = [-1] * 10
prev_action2 = [-1] * 10
prev_actions = [prev_action1, prev_action2]

while gamestate := console.step():
    
    if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
        """ Enter controller_cpu logic here"""

        done = check_game_over(gamestate)
        if done: 
            print(f"Game over! Player {done[1]} lost")
            break
        p1_input, p2_input, _ = gamestate_to_model_input(gamestate, prev_actions)

        game_buffer_p1.append(p1_input)
        game_buffer_p2.append(p2_input)

        with torch.inference_mode():
            p1_input_tensor = torch.tensor(game_buffer_p1, dtype=torch.float32, device='cuda').unsqueeze(0)
            p2_input_tensor = torch.tensor(game_buffer_p2, dtype=torch.float32, device='cuda').unsqueeze(0)

            output = model.fc1(p1_input_tensor)
            output2 = model.fc1(p2_input_tensor)
            output[:,:5] = torch.sigmoid(output[:,:5])
            output2[:,:5] = torch.sigmoid(output2[:,:5])
        

        digital_outputs = get_digital_action(output)
        digital_outputs2 = get_digital_action(output2)

        # clip the analog values to [0, 1]
        output_clamped = torch.clamp(output[:, 5:], 0, 1)
        print(f"Output analog: {output_clamped.squeeze().tolist()}")
        print(f"Output digital: {output[:,:5].squeeze().tolist()}")
        print(f"Digital outputs: {digital_outputs}")
        print('\n')

        # Press the buttons
        # for button in melee.enums.Button:
        #     if button in digital_outputs:
        #         controller_cpu.press_button(button)
        #         continue
        #     controller_cpu.release_button(button)
        # for button in melee.enums.Button:
        #     if button in digital_outputs2:
        #         controller_cpu2.press_button(button)
        #         continue
        #     controller_cpu2.release_button(button)

        # Control the sticks
        ax, ay, cx, cy, l = output_clamped.squeeze().tolist()
        controller_cpu.tilt_analog(melee.Button.BUTTON_MAIN, ax, ay)
        # controller_cpu.tilt_analog(melee.Button.BUTTON_C, cx, cy)
        # controller_cpu.press_shoulder(melee.Button.BUTTON_L, l)
        # ax, ay, cx, cy, l = output2[:, 5:].squeeze().tolist()
        # controller_cpu2.tilt_analog(melee.Button.BUTTON_MAIN, ax, ay)
        # controller_cpu2.tilt_analog(melee.Button.BUTTON_C, cx, cy)
        # controller_cpu2.press_shoulder(melee.Button.BUTTON_L, l)
    else:
        melee.MenuHelper.menu_helper_simple(gamestate,
                                            controller_cpu,
                                            melee.Character.FOX,
                                            melee.Stage.YOSHIS_STORY,
                                            "",
                                            autostart=True,
                                            swag=True)
        melee.MenuHelper.menu_helper_simple(gamestate,
                                            controller_cpu2,
                                            melee.Character.FOX,
                                            melee.Stage.YOSHIS_STORY,
                                            "",
                                            autostart=True,
                                            swag=True)
