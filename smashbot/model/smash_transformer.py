import wandb 

from melee.enums import Stage, Action, Character, ProjectileType

import torch
import torch.nn as nn
import torch.nn.functional as F

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)



# From dataset.py
MISC_TYPE = 1
PROJECTILE_TYPE = 2
PLAYER_TYPE = 3
NANA_TYPE = 4
ACTION_TYPE = 5

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
    
class PolicyHead(nn.Module):
    def __init__(self, model_dim, action_dim, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(model_dim)
        self.linear = nn.Linear(model_dim, model_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.final_norm = nn.LayerNorm(model_dim)
        self.fc = nn.Linear(model_dim, action_dim)

    def forward(self, x):
        out = self.dropout(self.act(self.linear(self.norm(x))))
        out = out + x # skip
        out = self.final_norm(out)
        out = self.fc(out)
        return out
       
class SmashTransformer(nn.Module):
    def __init__(self, action_dim, embed_dim=224, model_dim=384, type_embed_dim=16, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.model_dim = model_dim
        self.action_dim = action_dim
        self.embed_dim = embed_dim
        encoder_layer = nn.TransformerEncoderLayer(model_dim, nhead, model_dim*4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        C = 21

        # Buncha embeddings for the enums
        # action and character are both part of {player/nana}state so they split the model_dim
        self.stage_embedding = nn.Embedding(len(stage_to_index), embed_dim-13) # -3 for id, distance, frame, -10 action
        self.action_embedding = nn.Embedding(len(action_to_index), (embed_dim-20)//2+1) # -20 for rest of {player/nana}state. //2 for character +1 for action importance vs char
        self.character_embedding = nn.Embedding(len(character_to_index), (embed_dim-20)//2) 
        self.projectile_embedding = nn.Embedding(len(projectile_to_index), embed_dim-8) # -7 for rest of projectile state
        self.type_embedding = nn.Embedding(len(type_to_index), type_embed_dim)
        
        # Stage embeddings
        max_stage_val = max(stage_to_index.keys())
        stage_lookup_tensor = torch.full((max_stage_val+1,), -1)
        for stage, idx in stage_to_index.items():
            stage_lookup_tensor[stage] = idx
        self.register_buffer('stage_lookup_tensor', stage_lookup_tensor)

        # Action embeddings
        max_action_val = max(action_to_index.keys())
        action_lookup_tensor = torch.full((max_action_val+1,), -1)
        for action, idx in action_to_index.items():
            action_lookup_tensor[action] = idx
        self.register_buffer('action_lookup_tensor', action_lookup_tensor)

        # Character embeddings
        max_character_val = max(character_to_index.keys())
        character_lookup_tensor = torch.full((max_character_val+1,), -1)
        for character, idx in character_to_index.items():
            character_lookup_tensor[character] = idx
        self.register_buffer('character_lookup_tensor', character_lookup_tensor)
        
        # Projectile embeddings
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
        # Register lookup tensors
        # self.register_buffer('stage_lookup_tensor', self.create_lookup_tensor(stage_to_index))
        # self.register_buffer('action_lookup_tensor', self.create_lookup_tensor(action_to_index))
        # self.register_buffer('character_lookup_tensor', self.create_lookup_tensor(character_to_index))
        # self.register_buffer('projectile_lookup_tensor', self.create_lookup_tensor(projectile_to_index))
        # self.register_buffer('type_lookup_tensor', self.create_lookup_tensor(type_to_index))

        # Learnable pred token sequence        
        self.pred_token = nn.Parameter(torch.empty(1, 1, model_dim))
        nn.init.kaiming_normal_(self.pred_token, mode='fan_in', nonlinearity='relu')
        self.pred_sequences = 4 # learnable + misc + 2players minimum

        self.policy_head = PolicyHead(self.pred_sequences*model_dim, action_dim, dropout=dropout)

        self.embed_linear = nn.Linear(embed_dim, model_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)
        self.non_embedded_player_feats = [i for i in range(C) if i not in (1, 3)]
    
    def create_lookup_tensor(self, index_dict):
        # Create a tensor that can accommodate the largest index in the dictionary
        max_index = max(index_dict.values())
        lookup_tensor = torch.full((max_index + 1,), -1, dtype=torch.long)
        for key, value in index_dict.items():
            lookup_tensor[value] = key
        return lookup_tensor
    
    def calculate_loss(self, pred_action, target_action):
        # cross entropy for buttons and mse for sticks
        buttons = pred_action[:, :self.action_dim//2]
        sticks = pred_action[:, self.action_dim//2:]
        buttons_loss = F.binary_cross_entropy_with_logits(buttons, target_action[:, :self.action_dim//2])
        sticks_loss = F.mse_loss(sticks, target_action[:, self.action_dim//2:])
        total_loss = buttons_loss + sticks_loss
        return total_loss, buttons_loss, sticks_loss
    
    def get_action(self, src):
        if len(src.shape) < 3:
            src = src.unsqueeze(0) 

        out = self.forward(torch.as_tensor(src).float())
        out = F.sigmoid(out)
        digital, analog = out[:, :5], out[:, 5:]

        # Return digital as multi-hot encoding of button predictions (up to two)
        values, indices = torch.sort(digital, descending=True)
        valid_indices = indices[values > 0.5]

        multi_hot = torch.zeros_like(digital)
        if valid_indices.numel() > 0:
            if valid_indices.ndim == 1:
                valid_indices = valid_indices.unsqueeze(0)

            # Get up to top two valid indices
            top_indices = valid_indices[:, :min(2, valid_indices.size(1))]
            for idx in top_indices.flatten():
                multi_hot[:, idx] = 1

        return multi_hot, analog
    
    def embed_gamestate(self, src):
        """ 
        Embeds the gamestate into a higher dimensional space for the transformer.                
                    
        The parsed gamestate is shape (B,S,21) so each sequence has some important features embedded
        before all features are transformed to model_dim."""

        # Each of the s in (B,S,C) contain info related to playerstate, or nanastate, or projectiles, or misc
        # The misc info is distance (btwn players), frame, and stage.
        B, S, C = src.shape 

        all_sequence_embeddings = torch.zeros(B, S, self.embed_dim, device=src.device)

        # Process MISC_TYPE - always 0th sequence
        stage_indices = self.stage_lookup_tensor[src[:,0,3].long()] # stage is 3rd feature
        embedded_stage = self.stage_embedding(stage_indices)
        all_sequence_embeddings[:,0,:] = torch.cat([src[:,0,:13], embedded_stage], dim=-1)

        # Process PLAYER_TYPE / NANA_TYPE
        player_types_mask = (torch.abs(src[:, :, 0]) == PLAYER_TYPE) | (torch.abs(src[:, :, 0]) == NANA_TYPE)
        if player_types_mask.any():
            action_indices = src[:,:,1][player_types_mask]
            character_indices = src[:,:,3][player_types_mask]
            action_indices = self.action_lookup_tensor[action_indices.long()]
            character_indices = self.character_lookup_tensor[character_indices.long()] 
            embedded_actions = self.action_embedding(action_indices)
            embedded_characters = self.character_embedding(character_indices)  
            rest_features = src[:, :, self.non_embedded_player_feats][player_types_mask]
            all_sequence_embeddings[player_types_mask] = torch.cat([rest_features, embedded_actions, embedded_characters], dim=-1)
      
        # Process PROJECTILE_TYPE
        projectile_mask = (src[:, :, 0] == PROJECTILE_TYPE)
        if projectile_mask.any():
            proj_values = src[:,:,8][projectile_mask]
            proj_indices = self.projectile_lookup_tensor[proj_values.long()]
            embedded_projectile_type = self.projectile_embedding(proj_indices)
            projectile_rest = src[:,:,:8][projectile_mask]
            all_sequence_embeddings[projectile_mask] = torch.cat([projectile_rest, embedded_projectile_type], dim=-1)

        all_sequence_embeddings = self.embed_norm(all_sequence_embeddings)
        all_sequence_embeddings = self.embed_linear(all_sequence_embeddings)
        return all_sequence_embeddings
    
    def extract_features(self, src):
        B, S, C = src.shape

        # Embed and concat learnable pred tokens
        src_embedded = self.embed_gamestate(src.float())
        src_embedded = torch.cat([self.pred_token.expand(B, 1, -1), src_embedded], dim=1)
        
        output = self.transformer_encoder(src_embedded)
        output = output[:, :self.pred_sequences, :].view(B, -1)
        return output

    def forward(self, src):
        B, S, C = src.shape

        # Embed and concat learnable pred tokens
        src_embedded = self.embed_gamestate(src.float())
        src_embedded = torch.cat([self.pred_token.expand(B, 1, -1), src_embedded], dim=1)
        
        output = self.transformer_encoder(src_embedded)
        output = self.policy_head(output[:, :self.pred_sequences, :].view(B, -1))
        return output
    
