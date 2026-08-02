

# Enums for indicating data type
import copy


MISC_TYPE = 1
PROJECTILE_TYPE = 2
PLAYER_TYPE = 3
NANA_TYPE = 4
ACTION_TYPE = 5


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
        projectile_state_list.append(projectile_state)
    return projectile_state_list

def parse_nana(nana):
    if nana is None: return nana
        
    player_state = [
        nana.action.value, nana.action_frame, nana.character.value,
        int(nana.facing), int(nana.hitlag_left), nana.hitstun_frames_left, 
        nana.invulnerability_left, int(nana.invulnerable), nana.jumps_left,
        int(nana.on_ground), nana.percent, # nana index removed
        nana.position.x, nana.position.y, nana.shield_strength,
        nana.speed_air_x_self, nana.speed_ground_x_self, nana.speed_x_attack,
        nana.speed_y_attack, nana.speed_y_self, nana.stock
    ]
    return player_state 

def parse_game_state(gamestate, in_game=False):
    
    # 1. environment state info
    env_info = [gamestate.distance, gamestate.frame, gamestate.stage.value]
    projectiles = parse_projectiles(gamestate.projectiles)
    env_info.append(projectiles)

    # 2. player state info 
    playerstate_list = []

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
    
        playerstate_list.append(player_state)
    
    observation = env_info + playerstate_list

    return observation

def generate_input_python(observation, player_index):
    """
    Generate the input data from the observation. Uses plain python lists instead of arrays
    or tensors to limit memory usage during multiprocessing. 
    
    Output of function can be converted to tensors or arrays directly with torch.tensor()
    or np.array(). Will have shape (S,21), where S depends on number of players, projectiles, nanas, etc

    First index indicates whether the data corresponds to player, projectile, nana,
    or misc info. Negative value indicates currently active player.
    """

    copy_observation = copy.deepcopy(observation)
    all_tensors = []
    
    misc = copy_observation[:3]  # distance, frame, stage
    projectiles = copy_observation[3]
    players = copy_observation[4:] 
    nana_states = [obs.pop(9) for obs in players] 

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

    return all_tensors

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