import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb 

from data.dataset import MISC_TYPE, ACTION_TYPE, PROJECTILE_TYPE, PLAYER_TYPE, NANA_TYPE

from melee.enums import Stage, Action, Character, ProjectileType


TYPE_LIST = [MISC_TYPE, ACTION_TYPE, PROJECTILE_TYPE, PLAYER_TYPE, NANA_TYPE, -PLAYER_TYPE, -NANA_TYPE]

stage_to_index      =  {stage.value: index for index, stage in enumerate(Stage)}
action_to_index     =  {action.value: index for index, action in enumerate(Action)}
character_to_index  =  {character.value: index for index, character in enumerate(Character)}
projectile_to_index =  {projectile.value: index for index, projectile in enumerate(ProjectileType)}
type_to_index       =  {type_: index for index, type_ in enumerate(TYPE_LIST)}

class ResidualBlock(nn.Module):
    def __init__(self, model_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.linear1 = nn.Linear(model_dim, model_dim)
        self.linear2 = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        x = x + self.dropout1(self.act1(self.linear1(self.norm1(x))))
        x = x + self.dropout2(self.act2(self.linear2(self.norm2(x))))
        return x

class SmashTransformer(nn.Module):
    def __init__(self, action_dim, embed_dim=224, model_dim=384, type_embed_dim=16, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        encoder_layer = nn.TransformerEncoderLayer(model_dim, nhead, model_dim*4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Mask for player/nana state
        C = 21
        feat_indices = torch.arange(C)
        self.player_mask = (feat_indices != 1) & (feat_indices != 3)

        # Buncha embeddings for the enums
        # action and character are both part of {player/nana}state so they split the model_dim
        self.stage_embedding = nn.Embedding(len(stage_to_index), embed_dim-3) # -3 for id, distance, frame
        self.action_embedding = nn.Embedding(len(action_to_index), (embed_dim-20)//2+1) # -20 for rest of {player/nana}state. //2 for character +1 for action importance vs char
        self.character_embedding = nn.Embedding(len(character_to_index), (embed_dim-20)//2) 
        self.projectile_embedding = nn.Embedding(len(projectile_to_index), embed_dim-8) # -7 for rest of projectile state
        self.type_embedding = nn.Embedding(len(type_to_index), type_embed_dim)
        
        # Lookups for embeddings
        max_stage_val = max(stage_to_index.keys())
        stage_lookup_tensor = torch.full((max_stage_val+1,), -1)
        for stage, idx in stage_to_index.items():
            stage_lookup_tensor[stage] = idx
        self.register_buffer('stage_lookup_tensor', stage_lookup_tensor)

        max_action_val = max(action_to_index.keys())
        action_lookup_tensor = torch.full((max_action_val+1,), -1)
        for action, idx in action_to_index.items():
            action_lookup_tensor[action] = idx
        self.register_buffer('action_lookup_tensor', action_lookup_tensor)

        max_character_val = max(character_to_index.keys())
        character_lookup_tensor = torch.full((max_character_val+1,), -1)
        for character, idx in character_to_index.items():
            character_lookup_tensor[character] = idx
        self.register_buffer('character_lookup_tensor', character_lookup_tensor)
        
        max_projectile_val = max(projectile_to_index.keys())
        projectile_lookup_tensor = torch.full((max_projectile_val+1,), -1)
        for projectile, idx in projectile_to_index.items():
            projectile_lookup_tensor[projectile] = idx
        self.register_buffer('projectile_lookup_tensor', projectile_lookup_tensor)

        max_type_val = max(type_to_index.keys())
        type_lookup_tensor = torch.full((max_type_val+1,), -1)
        for type_, idx in type_to_index.items():
            type_lookup_tensor[type_] = idx
        self.register_buffer('type_lookup_tensor', type_lookup_tensor)

        # Learnable pred token sequence        
        self.pred_token = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.kaiming_normal_(self.pred_token, mode='fan_in', nonlinearity='relu')
        self.pred_sequences = 4 # learnable + misc + 2players minimum

        self.policy_head = nn.Sequential(
            ResidualBlock(self.pred_sequences*model_dim, 2*self.pred_sequences*model_dim),
            nn.LayerNorm(self.pred_sequences*model_dim),
            nn.Linear(self.pred_sequences*model_dim, action_dim)
        )

        self.embed_linear = nn.Linear(embed_dim+type_embed_dim-1, model_dim)
        self.embed_norm = nn.LayerNorm(embed_dim+type_embed_dim-1)
        self.non_embedded_player_feats = [i for i in range(C) if i not in (1, 3)]

    def calculate_loss(self, pred_action, target_action):
        # cross entropy for buttons and mse for sticks
        buttons = pred_action[:, :self.action_dim//2]
        sticks = pred_action[:, self.action_dim//2:]

        buttons_loss = F.cross_entropy(buttons, target_action[:, :self.action_dim//2])
        sticks_loss = F.mse_loss(sticks, target_action[:, self.action_dim//2:])
        total_loss = buttons_loss + sticks_loss
        return total_loss, buttons_loss, sticks_loss
        
    def forward(self, src):
        # Each of the s in S contain info related to playerstate, or nanastate, or projectiles, or misc
        # The misc info is distance (btwn players), frame, and stage.
        B, S, C = src.shape 

        all_sequence_embeddings = torch.zeros(B, S, self.embed_dim, device=src.device)

        # Process MISC_TYPE
        # misc is always 0th sequence
        stage_indices = self.stage_lookup_tensor[src[:,0,3].long()] # stage is 3rd feature
        embedded_stage = self.stage_embedding(stage_indices)
        all_sequence_embeddings[:,0,:] = torch.cat([src[:,0,:3], embedded_stage], dim=-1)

        # Process PROJECTILE_TYPE
        projectile_mask = (src[:, :, 0] == PROJECTILE_TYPE)
        if projectile_mask.any():
            proj_values = src[:,:,8][projectile_mask]
            proj_indices = self.projectile_lookup_tensor[proj_values.long()]
            embedded_projectile_type = self.projectile_embedding(proj_indices)
            projectile_rest = src[:,:,:8][projectile_mask]
            all_sequence_embeddings[projectile_mask] = torch.cat([embedded_projectile_type, projectile_rest], dim=-1)

        # Process PLAYER_TYPE / NANA_TYPE
        player_types_mask = (torch.abs(src[:, :, 0]) == PLAYER_TYPE) | (torch.abs(src[:, :, 0]) == NANA_TYPE)
        if player_types_mask.any():
            action_indices = src[:,:,1][player_types_mask]
            character_indices = src[:,:,3][player_types_mask]

            if action_indices.any():
                action_indices = self.action_lookup_tensor[action_indices.long()]
                embedded_actions = self.action_embedding(action_indices)
            if character_indices.any():
                character_indices = self.character_lookup_tensor[character_indices.long()]
                embedded_characters = self.character_embedding(character_indices)
        
            rest_features = src[:, :, self.non_embedded_player_feats][player_types_mask]
            all_sequence_embeddings[player_types_mask] = torch.cat([embedded_actions, embedded_characters, rest_features], dim=-1)
        
        type_indices = self.type_lookup_tensor[src[:,:,0].long()]
        embedded_types = self.type_embedding(type_indices)

        all_sequence_embeddings = torch.cat([embedded_types, all_sequence_embeddings[:,:,1:]], dim=-1)        
        all_sequence_embeddings = self.embed_norm(all_sequence_embeddings)
        all_sequence_embeddings = self.embed_linear(all_sequence_embeddings)

        # Add learnable pred tokens
        all_sequence_embeddings = torch.cat([self.pred_token.expand(B, 1, -1), all_sequence_embeddings], dim=1)
        
        output = self.transformer_encoder(all_sequence_embeddings)
        output = self.policy_head(output[:, :self.pred_sequences, :].view(B, -1))
        return output
    