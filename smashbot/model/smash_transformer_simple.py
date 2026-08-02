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




stage_to_index      =  {stage.value: index for index, stage in enumerate(Stage)}
action_to_index     =  {action.value: index for index, action in enumerate(Action)}
character_to_index  =  {character.value: index for index, character in enumerate(Character)}
projectile_to_index =  {projectile.value: index for index, projectile in enumerate(ProjectileType)}


    
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
        encoder_layer = nn.TransformerEncoderLayer(model_dim, nhead, model_dim*4, dropout, norm_first=True, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Input is (B, 3, 42)
        S = 3
        C = 42

        # Learnable pred token sequence. Fill up first sequence with this token so stage doesn't dominate        
        self.pred_token = nn.Parameter(torch.empty(1, model_dim))
        nn.init.kaiming_normal_(self.pred_token, mode='fan_in', nonlinearity='relu')
        
        # Embeddings for the enums
        # action and character are both part of {player/nana}state so they split the model_dim
        self.stage_embedding = nn.Embedding(len(stage_to_index), embed_dim//2) # //2 for filling rest with pred_token
        self.action_embedding = nn.Embedding(len(action_to_index), (embed_dim-18)//2) # -20 for rest of {player/nana}state. //2 for character +1 for action importance vs char
        self.character_embedding = nn.Embedding(len(character_to_index), (embed_dim-18)//2) 

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

        # Pred/linear layers and norm
        self.policy_head = PolicyHead(S*model_dim, action_dim, dropout=dropout)
        self.embed_linear = nn.Linear(embed_dim, model_dim)
        self.embed_norm = nn.LayerNorm(embed_dim)

    
    def calculate_loss(self, pred_action, target_action):
        # cross entropy for buttons and mse for sticks
        buttons = pred_action[:, :self.action_dim//2]
        sticks = pred_action[:, self.action_dim//2:]
        buttons_loss = F.binary_cross_entropy_with_logits(buttons, target_action[:, :self.action_dim//2])
        sticks_loss = F.mse_loss(sticks, target_action[:, self.action_dim//2:])
        total_loss = buttons_loss + sticks_loss
        return total_loss, buttons_loss, sticks_loss
    
    def embed_gamestate(self, src):
        """ 
        Embeds the gamestate into a higher dimensional space for the transformer.                
                
        The parsed gamestate is shape (B, 42) so each sequence has some important features embedded
        before all features are transformed to model_dim."""

        # The C = 42 comes from 2 features (distance, stage) + 20 features (player1) + 20 features (player2)
        B, C = src.shape 
        
        # Need to embed: stage (index 1), action (index 2), character (index 4)
        all_sequence_embeddings = torch.zeros(B, 3, self.embed_dim, device=src.device)

        # Embed stage
        stage_indices = self.stage_lookup_tensor[src[:,1].long()]
        embedded_stage = self.stage_embedding(stage_indices)

        # Rest of zeros will be filled with learnable self.pred_token
        all_sequence_embeddings[:, 0, 0] = src[:, 0]  # distance
        all_sequence_embeddings[:, 0, 1:1+self.stage_embedding.embedding_dim] = embedded_stage

        # Embedding for both players
        for i in range(2):
            offset = 20 * i
            action_indices = self.action_lookup_tensor[src[:, 2 + offset].long()]
            character_indices = self.character_lookup_tensor[src[:, 4 + offset].long()] 
            embedded_actions = self.action_embedding(action_indices)
            embedded_characters = self.character_embedding(character_indices)  
            
            # Gather additional features and embed
            additional_features = src[:, 3 + offset].unsqueeze(-1)  # Specific player feature
            other_features = src[:, 5 + offset:22 + offset]  # Remaining features
            
            all_sequence_embeddings[:, i + 1, :] = torch.cat(
                [embedded_actions, embedded_characters, additional_features, other_features], dim=1
            )

        # Normalize and linearly transform embeddings
        all_sequence_embeddings = self.embed_norm(all_sequence_embeddings)
        all_sequence_embeddings = self.embed_linear(all_sequence_embeddings)
        return all_sequence_embeddings
    
    def extract_features(self, src):
        B, C = src.shape

        # Embed and concat learnable pred tokens
        src_embedded = self.embed_gamestate(src.float())
        # src_embedded = torch.cat([src_embedded, self.pred_token.expand(B, 1, -1), ], dim=1)
        src_embedded[:, 0] += self.pred_token.expand(B, -1)
        
        output = self.transformer_encoder(src_embedded)
        output = output.view(B, -1)
        return output

    def forward(self, src):
        B, C = src.shape

        # Embed and concat learnable pred tokens
        src_embedded = self.embed_gamestate(src.float())
        # src_embedded = torch.cat([self.pred_token.expand(B, 1, -1), src_embedded], dim=1)
        src_embedded[:, 0] += self.pred_token.expand(B, -1)
        
        output = self.transformer_encoder(src_embedded)
        output = self.policy_head(output.view(B, -1))
        return output
    
