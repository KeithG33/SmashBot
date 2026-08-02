import wandb

from melee.enums import Stage, Action, Character, ProjectileType

import torch
import torch.nn as nn
import torch.nn.functional as F


stage_to_index = {stage.value: index for index, stage in enumerate(Stage)}
action_to_index = {action.value: index for index, action in enumerate(Action)}
character_to_index = {
    character.value: index for index, character in enumerate(Character)
}
projectile_to_index = {
    projectile.value: index for index, projectile in enumerate(ProjectileType)
}


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
        out = out + x  # skip
        out = self.final_norm(out)
        out = self.fc(out)
        return out


class SmashTransformer(nn.Module):
    def __init__(
        self,
        action_dim,
        embed_dim=224,
        model_dim=384,
        act_embed_dim=72,
        nhead=8,
        num_layers=6,
        dropout=0.1,
    ):
        

        super().__init__()
        self.model_dim = model_dim
        self.action_dim = action_dim
        # self.embed_dim = embed_dim
        src_encoder_layer = nn.TransformerEncoderLayer(
            model_dim,
            nhead,
            model_dim * 4,
            dropout, 
            norm_first=True, 
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            src_encoder_layer, 
            num_layers)

        self.action_linear = nn.Linear(action_dim, act_embed_dim)
        act_encoder_layer = nn.TransformerEncoderLayer(
            act_embed_dim,
            nhead,
            act_embed_dim * 4,
            dropout,
            norm_first=True,
            batch_first=True,
            activation='gelu'
        )
        self.action_encoder = nn.TransformerEncoder(act_encoder_layer, 3)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            kdim=act_embed_dim,
            vdim=act_embed_dim,
            num_heads=nhead,
            batch_first=True,
        )

        # Input is (B, S, C)
        S = 10
        C = 52

        self.pos_encoding = nn.Parameter(torch.empty((1, S, model_dim)))
        torch.nn.init.kaiming_normal_(self.pos_encoding)

        embed_features = round(
            (model_dim - act_embed_dim - C + 15) / (1 + 2 + 2)
        )  # 1xstage, 2xaction, 2xcharacter

        # Embeddings for the enums
        # action and character are both part of {player/nana}state so they split the model_dim
        self.stage_embedding = nn.Embedding(len(stage_to_index), embed_features - 1)
        self.action_embedding = nn.Embedding(len(action_to_index), embed_features)
        self.character_embedding = nn.Embedding(len(character_to_index), embed_features)

        # Stage embeddings
        max_stage_val = max(stage_to_index.keys())
        stage_lookup_tensor = torch.full((max_stage_val + 1,), -1)
        for stage, idx in stage_to_index.items():
            stage_lookup_tensor[stage] = idx
        self.register_buffer('stage_lookup_tensor', stage_lookup_tensor)

        # Action embeddings
        max_action_val = max(action_to_index.keys())
        action_lookup_tensor = torch.full((max_action_val + 1,), -1)
        for action, idx in action_to_index.items():
            action_lookup_tensor[action] = idx
        self.register_buffer('action_lookup_tensor', action_lookup_tensor)

        # Character embeddings
        max_character_val = max(character_to_index.keys())
        character_lookup_tensor = torch.full((max_character_val + 1,), -1)
        for character, idx in character_to_index.items():
            character_lookup_tensor[character] = idx
        self.register_buffer('character_lookup_tensor', character_lookup_tensor)

        # Pred/linear layers and norm
        self.policy_head = PolicyHead(S // 2 * model_dim, action_dim, dropout=dropout)
        self.embed_linear = nn.Linear(model_dim, model_dim)
        self.embed_norm = nn.LayerNorm(model_dim)

    def calculate_loss(self, pred_action, target_action):
        # cross entropy for buttons and mse for sticks
        buttons = pred_action[:, : self.action_dim // 2]
        sticks = pred_action[:, self.action_dim // 2 :]
        buttons_loss = F.binary_cross_entropy_with_logits(
            buttons, target_action[:, : self.action_dim // 2]
        )
        sticks_loss = F.mse_loss(sticks, target_action[:, self.action_dim // 2 :])
        total_loss = buttons_loss + sticks_loss
        return total_loss, buttons_loss, sticks_loss

    def embed_gamestate(self, src):
        """
        Embeds the gamestate into a higher dimensional space for the transformer.

        The parsed gamestate is shape (B, S, 52) so each sequence has some important features embedded
        before all features are transformed to model_dim."""

        # The C = 52 comes from 2 (distance, stage) + 20 (player1) + 20 (player2) + 10 (actions) features
        # The S = 25 comes from the sequence length
        # Order is prev_action + p1_info + p2_info + misc_info
        B, S, C = src.shape

        # Need to embed: prev_actions, stage, action_type (index 0 of players), character_type (index 2 of players)
        all_sequence_embeddings = torch.zeros(B, S, self.model_dim, device=src.device)

        prev_actions = src[:, :, :10]
        prev_actions_embedded = self.action_linear(prev_actions)

        stage_indices = self.stage_lookup_tensor[
            src[:, :, -1].long()
        ] # stage 
        action_indices1 = self.action_lookup_tensor[
            src[:, :, 10].long()
        ]  # player1 action
        action_indices2 = self.action_lookup_tensor[
            src[:, :, 10 + 20].long()
        ]  # player2 action
        character_indices1 = self.character_lookup_tensor[
            src[:, :, 12].long()
        ]  # player1 character
        character_indices2 = self.character_lookup_tensor[
            src[:, :, 12 + 20].long()
        ]

        embedded_stage = self.stage_embedding(stage_indices)
        embedded_action1 = self.action_embedding(action_indices1)
        embedded_action2 = self.action_embedding(action_indices2)

        embedded_character1 = self.character_embedding(character_indices1)
        embedded_character2 = self.character_embedding(character_indices2)

        all_sequence_embeddings = torch.cat(
            [
                prev_actions_embedded,  # prev_action
                embedded_action1,
                embedded_character1,
                src[:, :, 11].unsqueeze(-1),
                src[:, :, 13:30],
                embedded_action2,
                embedded_character2,
                src[:, :, 31].unsqueeze(-1),
                src[:, :, 33:52],
                embedded_stage,
            ],
            dim=-1,
        )

        # Normalize, add pos encodings, and embed
        all_sequence_embeddings = self.embed_norm(all_sequence_embeddings)
        all_sequence_embeddings += self.pos_encoding.expand(B, -1, -1)
        all_sequence_embeddings = self.embed_linear(all_sequence_embeddings)
        return all_sequence_embeddings, prev_actions_embedded

    def extract_features(self, src):
        B, S, C = src.shape

        # Encode src and prev_actions on its own
        src_embedded, prev_actions_embedded = self.embed_gamestate(src.float())
        src_encoded = self.transformer_encoder(src_embedded)
        prev_actions_encoded = self.action_encoder(prev_actions_embedded.float())

        # Merge prev_actions and src_encoded with cross-attention
        output, _ = self.cross_attention(
            src_encoded, prev_actions_encoded, prev_actions_encoded
        )
        output += src_encoded
        output = output[:, 5:, :].reshape(B, -1)
        return output

    def forward(self, src):
        B, S, C = src.shape

        # Encode src and prev_actions on its own
        src_embedded, prev_actions_embedded = self.embed_gamestate(src.float())
        src_encoded = self.transformer_encoder(src_embedded)
        prev_actions_encoded = self.action_encoder(prev_actions_embedded.float())

        # Merge prev_actions and src_encoded with cross-attention
        output, _ = self.cross_attention(
            src_encoded, prev_actions_encoded, prev_actions_encoded
        )
        output += src_encoded

        output = self.policy_head(output[:, 5:, :].reshape(B, -1))
        return output
