import random
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
    """
    stocks = [4,4]
    percent = [0,0]

    def reset(self):
        self.stocks = [4,4]
        self.percent = [0,0]

    def step(self, gamestate):
        # In case zero stocks not read
        if gamestate is None: 
            return True, (0,0)
        
        curr_stocks = [player.stock for port, player in gamestate.players.items()]
        curr_percent = [player.percent for port, player in gamestate.players.items()]

        # Assuming one player will have lost all stocks. Will be done when
        # Gamestate is None so this is just for reward
        for i, stock in enumerate(curr_stocks):
            if stock == 0:
                reward = (0, 1000) if i == 0 else (1000, 0)
                return True, reward

        # Should be boolean list of 1 or 0
        stock_change = [prev-curr for curr, prev in zip(curr_stocks, self.stocks)]
        # Min to ignore decreases in percent (new stock). Although should maybe consider rewarding
        # something like Ness's healing ability
        percent_change = [max(0,curr-prev) for curr, prev in zip(curr_percent, self.percent)] 

        # Use other player deltas as reward.
        percent_coeff = 0.01
        stock_coeff = 50

        reward1 = percent_change[1]*percent_coeff + stock_change[1]*stock_coeff
        reward2 = percent_change[0]*percent_coeff + stock_change[0]*stock_coeff
        reward = (reward1, reward2)

        self.stocks = curr_stocks
        self.percent = curr_percent

        return False, reward

        
class SmashMeleeEnv(gym.Env):

    """ 
    Gym self-play environment for Super Smash Bros Melee. 
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, slippi_path):
        super().__init__()
        self.slippi_path = slippi_path
        
        self.console = melee.Console(path=slippi_path, is_dolphin=True)
        self.controller1 = melee.Controller(console=self.console, port=1)
        # self.controller2 = melee.Controller(console=self.console, port=2)
        self.controller2 = melee.Controller(console=self.console, port=2,
                                                 type=melee.ControllerType.GCN_ADAPTER)
        self.console.run(iso_path='/home/kage/slippi/Super Smash Bros. Melee (USA) (En,Ja) (v1.02).iso')
        self.console.connect()
        self.controller1.connect()
        self.controller2.connect()

        # Define action space for controllers
        digital = spaces.MultiBinary(5)  # Five digital buttons
        analog = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)  # Five analog inputs
        controller_action_space = spaces.Dict({"digital": digital, "analog": analog})
        self.action_space = spaces.Tuple((controller_action_space, controller_action_space))

        # Define observation space using Sequence for each controller
        sequence_space = spaces.Sequence(spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32), seed=None)
        self.observation_space = spaces.Dict({"p1": sequence_space,
                                              "p2": sequence_space})
        # self.observation_space = spaces.Tuple((sequence_space, sequence_space))

        banned_chars = [
            melee.Character.WIREFRAME_MALE,
            melee.Character.WIREFRAME_FEMALE,
            melee.Character.UNKNOWN_CHARACTER,
            melee.Character.NANA,
            melee.Character.GIGA_BOWSER,
            melee.Character.SANDBAG
        ]

        self.valid_chars = [char for char in list(melee.Character) if char not in banned_chars]
        
        self.game_tracker = GameStateTracker()

    def _get_observations(self, obs):
        player1_obs = generate_input_python(obs, None, 0) 
        player2_obs = generate_input_python(obs, None, 1)
        return player1_obs, player2_obs

    def step(self, actions):
        """ Actions is a dict of digital and analog action spaces for each controller """
        action1, action2 = actions

        digital1 = action1["digital"]
        analog1 = action1["analog"]
        digital2 = action2["digital"]
        analog2 = action2["analog"]

        digital1 = digital_to_buttons(digital1)
        digital2 = digital_to_buttons(digital2)

        self.apply_action(self.controller1, digital1, analog1)
        self.apply_action(self.controller2, digital2, analog2)

        gamestate = self.console.step()

        done, reward = self.game_tracker.step(gamestate)
 
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
                                                    self.controller1,
                                                    melee.Character.FOX,
                                                    melee.Stage.FINAL_DESTINATION,
                                                    "",
                                                    autostart=True)
                melee.MenuHelper.choose_character(gamestate=gamestate,
                                                  character=melee.Character.FOX,
                                                  controller=self.controller2,
                                                  start=True)

        obs, _ = parse_game_state(self.console.step(), in_game=True)

        p1_obs, p2_obs = self._get_observations(obs)
        p1_obs = [np.array(seq) for seq in p1_obs]
        p2_obs = [np.array(seq) for seq in p2_obs]

        return {"p1": p1_obs, "p2": p2_obs}
        

    def render(self, mode='human', close=False):
        # Optional visualization
        pass

    def close(self):
        # Clean up resources
        self.controller1.release_all()
        self.controller2.release_all()
        self.controller1.disconnect()
        self.controller2.disconnect()


class SmashMeleeTestEnv(gym.Env):
    """ 
    Gym test environment for Super Smash Bros Melee. Meant for having one human player
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
        # self.observation_space = spaces.Dict({"p1": sequence_space,
        #                                       "p2": s)
        # self.observation_space = spaces.Tuple((sequence_space, sequence_space))

        banned_chars = [
            melee.Character.WIREFRAME_MALE,
            melee.Character.WIREFRAME_FEMALE,
            melee.Character.UNKNOWN_CHARACTER,
            melee.Character.NANA,
            melee.Character.GIGA_BOWSER,
            melee.Character.SANDBAG
        ]

        self.valid_chars = [char for char in list(melee.Character) if char not in banned_chars]
        
        self.game_tracker = GameStateTracker()

    def _get_observations(self, obs):
        player1_obs = generate_input_python(obs, None, 0) 
        return player1_obs

    def step(self, actions):
        """ 
        Actions is a dict with "digital" and "analog" keys. Digital is mult-hot encoded with up to two buttons
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
        self.controller_cpu.release_all()
        self.controller_human.release_all()
        self.controller_cpu.disconnect()
        self.controller_human.disconnect()
        self.console.stop()