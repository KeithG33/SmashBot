import itertools
import random, copy
import sys
import time
from typing import List
import numpy as np
import melee

import gymnasium as gym
from gymnasium import spaces

from smashbot.data.extract_dataset import generate_input_python, parse_game_state


INDEX_TO_DIGITAL = {
    0: melee.Button.BUTTON_A,
    1: melee.Button.BUTTON_B,
    2: melee.Button.BUTTON_L,
    3: melee.Button.BUTTON_X,
    4: melee.Button.BUTTON_Z
}

def digital_to_buttons(digital_out):
    """ 
    digital_out is a multi-hot encoded array for [A, B, L/R, X/Y, Z].

    Returns a list of digital button enums (melee.Button.BUTTON_A, etc.) where the value is 1.
    """

    digital = digital_out
    digital_indices = np.where(digital == 1)[0]
    digital_buttons = [INDEX_TO_DIGITAL[idx.item()] for idx in digital_indices]
    return digital_buttons

def check_game_over(gamestate):
    """ Assumed to be two player"""
    # check player stocks 
    for i, player in enumerate(gamestate.players):
        if player.stock == 0:
            return True, i
    return False, None
        

class GameStateTracker:
    """ 
    Tracks changes in percent, stocks for each player, whether game is over, and returns reward
    based on this.
    """
    stocks = [4,4]
    percent = [0,0]

    def reset(self):
        self.stocks = [4,4]
        self.percent = [0,0]

    def step(self, gamestate):        
        curr_stocks = [player.stock for port, player in gamestate.players.items()]
        curr_percent = [player.percent for port, player in gamestate.players.items()]

        # Assuming one player will have lost all stocks. Will be done when
        # Gamestate is None so this is just for reward
        for i, stock in enumerate(curr_stocks):
            if stock == 0:
                print(f"PLAYER {i} HAS LOST ALL STOCKS")
                reward = (0, 1000) if i == 0 else (1000, 0)
                return True, reward

        stock_change = [prev-curr for curr, prev in zip(curr_stocks, self.stocks)]
        percent_change = [max(0,curr-prev) for curr, prev in zip(curr_percent, self.percent)] 

        percent_coeff = 0.01
        stock_coeff = 50

        # Use other player for reward.
        reward1 = percent_change[1]*percent_coeff + stock_change[1]*stock_coeff
        reward2 = percent_change[0]*percent_coeff + stock_change[0]*stock_coeff
        reward = (reward1, reward2)

        self.stocks = curr_stocks
        self.percent = curr_percent

        return False, reward


MISC_TYPE = 1
PROJECTILE_TYPE = 2
PLAYER_TYPE = 3
NANA_TYPE = 4
ACTION_TYPE = 5

def generate_both_inputs_python(observation):
    """
    Generate input data for both players from the observation using plain python lists.
    This function reduces the need to call a deep copy and the overall function twice
    for each player.

    Outputs two lists of observations that can be converted to tensors or arrays directly
    with torch.tensor() or np.array(), with each having shape (S, 21) where S depends on
    number of players, projectiles, nanas, etc.

    :param observation: The original game observation.
    :return: A tuple containing two lists of observations for each player.
    """
    copy_observation = copy.deepcopy(observation)
    all_tensors_p1 = []
    all_tensors_p2 = []

    misc = copy_observation[:3]  # distance, frame, stage
    projectiles = copy_observation[3]
    players = copy_observation[4:] 
    nana_states = [obs.pop(9) for obs in players] 

    # Creating misc tensor data
    misc_types = [MISC_TYPE]
    misc_padded = misc + [0] * (20 - len(misc))
    misc = misc_types + misc_padded
    all_tensors_p1.append(misc.copy())
    all_tensors_p2.append(misc.copy())
    
    # Processing players and Nanas for both players
    for i in range(len(players)):
        player = players[i]
        nana = nana_states[i]

        # Player data
        player_type = [-PLAYER_TYPE] if i == 0 else [PLAYER_TYPE]
        player_data = player_type + player + [0] * (20 - len(player))
        if i == 0:
            all_tensors_p1.append(player_data)
            all_tensors_p2.append(player_data.copy())
        else:
            all_tensors_p2.append(player_data)
            all_tensors_p1.append(player_data.copy())
        
        # Nana data
        if nana is not None:
            nana_type = [-NANA_TYPE] if i == 0 else [NANA_TYPE]
            nana_data = nana_type + nana + [0] * (20 - len(nana))
            if i == 0:
                all_tensors_p1.append(nana_data)
                all_tensors_p2.append(nana_data.copy())
            else:
                all_tensors_p2.append(nana_data)
                all_tensors_p1.append(nana_data.copy())

    # Handling projectiles for both players
    if projectiles:
        for projectile in projectiles:
            projectile_type = [PROJECTILE_TYPE]
            projectile_data = projectile_type + projectile + [0] * (20 - len(projectile))
            all_tensors_p1.append(projectile_data.copy())
            all_tensors_p2.append(projectile_data.copy())

    return all_tensors_p1, all_tensors_p2


class SmashMeleeEnv(gym.Env):
    """ 
    Gym self-play environment for Super Smash Bros Melee. 
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, slippi_path):
        super().__init__()
        self.slippi_path = slippi_path
        
        if 'ExiAI' in slippi_path:    
            self.console = melee.Console(
                path=slippi_path,
                save_replays=False,
                gfx_backend="Null",
                disable_audio=True,
                use_exi_inputs=False,
                enable_ffw=False,
                blocking_input=True
            )
        else:
            self.console = melee.Console(path=slippi_path, fullscreen=False, save_replays=False)

        self.controller1 = melee.Controller(console=self.console, port=1)
        self.controller2 = melee.Controller(console=self.console, port=2)

        self.console.run(iso_path='/home/kage/slippi/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso')

        self.console.connect()
        self.controller1.connect()
        self.controller2.connect()

        # Define action space for controllers
        digital = spaces.MultiBinary(5)  # Five digital buttons
        analog = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)  # Five analog inputs
        controller_action_space = spaces.Dict({"digital": digital, "analog": analog})
        self.action_space = spaces.Dict({"p1": controller_action_space,
                                          "p2": controller_action_space})

        # Define observation space using Sequence for each controller
        sequence_space = spaces.Sequence(spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32), seed=None)
        self.observation_space = spaces.Dict({"p1": sequence_space,
                                              "p2": sequence_space})

        banned_chars = [
            melee.Character.WIREFRAME_MALE,
            melee.Character.WIREFRAME_FEMALE,
            melee.Character.UNKNOWN_CHARACTER,
            melee.Character.SANDBAG,
            melee.Character.NANA,
            melee.Character.GIGA_BOWSER,
            melee.Character.SHEIK,
            melee.Character.ZELDA
        ]

        self.valid_chars = [char for char in list(melee.Character) if char not in banned_chars]
        
        self.stage_list = [
            melee.Stage.BATTLEFIELD,
            melee.Stage.DREAMLAND,
            melee.Stage.FINAL_DESTINATION,
            melee.Stage.FOUNTAIN_OF_DREAMS,
            melee.Stage.POKEMON_STADIUM,
            melee.Stage.YOSHIS_STORY
        ]

        self.training_chars = [
            melee.Character.FOX,
            melee.Character.FALCO,
            melee.Character.MARTH
        ]
        self.game_tracker = GameStateTracker()

        start_index = 1313

        self.triplet_gen = self.create_triplet_generator(
                                                            self.valid_chars,
                                                            self.stage_list,
                                                            start_index)
        self.triplet_cnt = start_index
    

    def create_triplet_generator(self, characters, stages, start_index=0):
        """Generator function to create all possible triplets of two characters and one stage, starting from a given index."""
        total_combinations = len(characters) * len(characters) * len(stages)
        print(f"Total number of triplets: {total_combinations}")

        # Create the product iterator
        product_iterator = itertools.product(characters, characters, stages)
        # Use islice to skip to the start index
        sliced_iterator = itertools.islice(product_iterator, start_index, None)
        for triplet in sliced_iterator:
            yield triplet


    def _get_observations(self, obs):
        # player1_obs = generate_input_python(obs, None, 0) 
        # player2_obs = generate_input_python(obs, None, 1)
        player1_obs, player2_obs = generate_both_inputs_python(obs)
        return player1_obs, player2_obs


    def step(self, actions):
        """ 
        Actions is a dict with "digital" and "analog" keys. Digital is multi-hot encoded array
        for buttons. 
        """
        analog1 = actions["p1"]["analog"]
        digital1 = actions["p1"]["digital"]
        digital1 = digital_to_buttons(digital1)

        analog2 = actions["p2"]["analog"]
        digital2 = actions["p2"]["digital"]
        digital2 = digital_to_buttons(digital2)

        self.apply_action(self.controller1, digital1, analog1)
        self.apply_action(self.controller2, digital2, analog2)

        gamestate = self.console.step()

        if self.console.processingtime * 1000 > 12:
            print("WARNING: Last frame took " + str(self.console.processingtime*1000) + "ms to process.")

        if gamestate.menu_state not in [melee.enums.Menu.IN_GAME, melee.enums.Menu.SUDDEN_DEATH]:
            return None, (0,0), True, False, {}

        done, reward = self.game_tracker.step(gamestate)
 
        # Extract observations for both players
        obs, actions = parse_game_state(gamestate, in_game=True)
        p1_obs, p2_obs = self._get_observations(obs)

        truncated = False

        return {"p1": p1_obs, "p2": p2_obs}, reward, done, truncated, {}


    def apply_action(self, controller, digital, analog):
        # Press the buttons
        for button in melee.enums.Button:
            if button in digital:
                controller.press_button(button)
                continue
            controller.release_button(button)

        # Control the sticks
        ax, ay, cx, cy, l = analog.squeeze().tolist()
        controller.tilt_analog(melee.Button.BUTTON_MAIN, ax, ay)
        controller.tilt_analog(melee.Button.BUTTON_C, cx, cy)
        controller.press_shoulder(melee.Button.BUTTON_L, l)


    def reset(self, seed=None, options = {}):
        """
        fails: 
        - MARTH / FALCO / BATTLEFIELD (invalid read writes)
        """
        gamestate = self.console.step() 
        self.game_tracker.reset()

        try:
            p1_char, p2_char, stage = next(self.triplet_gen)
            self.triplet_cnt += 1
        except StopIteration:
            print("All triplets exhausted, done experiment.")
            sys.exit(0)
        
        # while melee.bad_ffw_combinations.is_bad_ffw_combination(p1_char, p2_char, stage):
        #     p1_char, p2_char, stage = next(self.triplet_gen)
        #     self.triplet_cnt += 1

        print(f"On triplet {self.triplet_cnt} - {p1_char}, {p2_char}, {stage}")
        

        # p1_char = options.get("p1_char", melee.Character.MARIO,)
        # p2_char = options.get("p2_char", melee.Character.MARTH,)
        # stage = options.get("stage", melee.Stage.BATTLEFIELD,)

        # p1_char = options.get("p1_char", random.choice(self.valid_chars))
        # p2_char = options.get("p2_char", random.choice(self.valid_chars))
        # stage = options.get("stage", random.choice(self.stage_list))
        # while melee.bad_ffw_combinations.is_bad_ffw_combination(p1_char, p2_char, stage):
        #     p1_char = random.choice(self.valid_chars)
        #     p2_char = random.choice(self.valid_chars)
        #     stage = random.choice(self.stage_list)

        print(f"CHOSEN SETUP: {p1_char}, {p2_char}, {stage}")

        while gamestate.menu_state not in [melee.enums.Menu.IN_GAME, melee.enums.Menu.SUDDEN_DEATH]:
            melee.MenuHelper.menu_helper_simple(gamestate,
                                                self.controller1,
                                                p1_char,
                                                stage,
                                                "",
                                                autostart=False,
                                                swag=False) 
            melee.MenuHelper.menu_helper_simple(gamestate,
                                                self.controller2,
                                                p2_char,
                                                stage,
                                                "",
                                                autostart=True,
                                                swag=True)
            gamestate = self.console.step()

        obs, _ = parse_game_state(gamestate, in_game=True)
        p1_obs, p2_obs = self._get_observations(obs)
        return {"p1": p1_obs, "p2": p2_obs}, {}
        

    def render(self, mode='human', close=False):
        # Optional visualization
        pass

    def close(self):
        # Clean up resources
        self.console.stop()


class SmashMeleeTestEnv(gym.Env):
    """ 
    Gym test environment for Super Smash Bros Melee. Has one human player
    and one CPU player.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, slippi_path, iso_path='/home/kage/slippi/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso'):
        super().__init__()
        self.slippi_path = slippi_path
        
        self.console = melee.Console(path=slippi_path, is_dolphin=True)
        self.controller_cpu = melee.Controller(console=self.console, port=1)
        self.controller_human = melee.Controller(console=self.console, port=2,
                                                 type=melee.ControllerType.GCN_ADAPTER)
        self.console.run(iso_path=iso_path)
        self.console.connect()
        self.controller_cpu.connect()
        self.controller_human.connect()

        # Define action space for controllers
        digital = spaces.MultiBinary(5)  # Five digital buttons
        analog = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)  # Five analog inputs
        self.action_space = spaces.Dict({"digital": digital, "analog": analog})
        
        # Define observation space using Sequence for each controller
        self.observation_space = spaces.Sequence(spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32), seed=None)

        banned_chars = [
            melee.Character.WIREFRAME_MALE,
            melee.Character.WIREFRAME_FEMALE,
            melee.Character.UNKNOWN_CHARACTER,
            melee.Character.NANA,
            melee.Character.GIGA_BOWSER,
            melee.Character.SANDBAG,
            melee.Character.SHEIK
        ]

        self.valid_chars = [char for char in list(melee.Character) if char not in banned_chars]
        
        self.game_tracker = GameStateTracker()

    def _get_observations(self, obs):
        player1_obs = generate_input_python(obs, None, 0) 
        return player1_obs

    def step(self, actions):
        """ 
        Actions is a dict with "digital" and "analog" keys. Digital is multi-hot encoded array
        for buttons.
        """
        digital = actions["digital"]
        analog = actions["analog"]

        digital = digital_to_buttons(digital)
        self.apply_action(self.controller_cpu, digital, analog)

        gamestate = self.console.step()

        done, reward = self.game_tracker.step(gamestate)
 
        # +1000 for winning
        if done:
            return None, reward, done, {}

        # Extract observations for both players
        obs, actions = parse_game_state(gamestate, in_game=True)
        observations = self._get_observations(obs)

        return observations, reward, done, {}

    def apply_action(self, controller, digital, analog):
        # Press the buttons
        for button in melee.enums.Button:
            if button in digital:
                controller.press_button(button)
                continue
            controller.release_button(button)

        # Control the sticks
        ax, ay, cx, cy, l = analog.squeeze().tolist()
        controller.tilt_analog(melee.Button.BUTTON_MAIN, ax, ay)
        controller.tilt_analog(melee.Button.BUTTON_C, cx, cy)
        controller.press_shoulder(melee.Button.BUTTON_L, l)

    def reset(self):
        self.game_tracker.reset()
        while True:
            gamestate = self.console.step()
            
            if gamestate.menu_state in [melee.Menu.IN_GAME, melee.Menu.SUDDEN_DEATH]:
                break
            else:
                melee.MenuHelper.menu_helper_simple(gamestate,
                                                    self.controller_cpu,
                                                    melee.Character.FOX,
                                                    melee.Stage.FINAL_DESTINATION,
                                                    "",
                                                    autostart=True)

        obs, _ = parse_game_state(self.console.step(), in_game=True)

        p1_obs = self._get_observations(obs)
        p1_obs = [np.array(seq) for seq in p1_obs]

        return p1_obs
        

    def render(self, mode='human', close=False):
        # Optional visualization
        pass

    def close(self):
        # Clean up resources
        self.controller_cpu.disconnect()
        self.controller_human.disconnect()
        self.console.stop()