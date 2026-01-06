"""
RCRG-2R-11C-conc-Temp-GAT Model
================================
Temporal relational model with 2 Graph Attention layers (2R) and 1 Clique of all 12 players (11C).
Uses concatenation and LSTM for temporal modeling across 9 frames.
Includes Multi-Head Self-Attention with Query, Key, Value mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.attention_model.RelationalGATLayer import RelationalGATLayer, clique_adjacency


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention Layer with Q, K, V.
    
    Q = X @ W_Q
    K = X @ W_K  
    V = X @ W_V
    Attention = softmax(Q @ K^T / sqrt(d_k)) @ V
    """
    
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Q, K, V projections
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.W_o = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x, return_attention=False):
        """
        Args:
            x: (B, N, D) - B batches, N tokens/players, D dimensions
            return_attention: whether to return attention weights
        Returns:
            out: (B, N, D)
            attn_weights: (B, num_heads, N, N) if return_attention
        """
        B, N, D = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x)  # (B, N, D)
        K = self.W_k(x)  # (B, N, D)
        V = self.W_v(x)  # (B, N, D)
        
        # Reshape for multi-head: (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (B, num_heads, N, N)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values: (B, num_heads, N, head_dim)
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads: (B, N, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        
        # Output projection
        out = self.W_o(attn_output)
        
        # Residual connection + layer norm
        out = self.layer_norm(out + x)
        
        if return_attention:
            return out, attn_weights
        return out


class RCRG_2R_11C_conc_Temp_GAT(nn.Module):
    def __init__(self, person_classifier, num_classes=8, feature_dim=2048):
        super(RCRG_2R_11C_conc_Temp_GAT, self).__init__()

        self.person_feature_extractor = person_classifier.resnet50
        for param in self.person_feature_extractor.parameters():
            param.requires_grad = False  # Freeze person feature extractor

        self.gat_layer1 = RelationalGATLayer(in_dim=feature_dim, out_dim=2048, dropout=0.5)
        
        self.self_attn1 = MultiHeadSelfAttention(embed_dim=2048, num_heads=4, dropout=0.3)
        
        self.gat_layer2 = RelationalGATLayer(in_dim=2048, out_dim=1024, dropout=0.4)
        
        self.self_attn2 = MultiHeadSelfAttention(embed_dim=1024, num_heads=4, dropout=0.3)

        self.proj = nn.Linear(1024, 512)
        self.layer_norm1= nn.LayerNorm(512)
        self.layer_norm2= nn.LayerNorm(512)

        self.hidden_size = 512
        self.lstm = nn.LSTM(1024, self.hidden_size, batch_first=True)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=12*512, out_features=256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=num_classes)
        )
        
        # Adjacency matrix for all players (1 clique)
        self.register_buffer('adj', clique_adjacency(K=12, num_cliques=1))

    def forward(self, x, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 12, 9, C, H, W)
            return_attention: If True, return attention weights for visualization
            
        Returns:
            out: Class logits (B, num_classes)
            attention_weights: (optional) Dict with attention weights from each layer
        """
        b, num_people, num_frames, c, h, w = x.size()
        x = x.view(b * num_frames * num_people, c, h, w)  # (B*9*12, C, H, W)

        # Extract person features
        x = self.person_feature_extractor(x)  # (B*9*12, 2048, 1, 1)
        x = x.view(b * num_frames, num_people, -1)  # (B*9, 12, 2048)

        # First GAT layer
        if return_attention:
            x, gat_attn1 = self.gat_layer1(x, self.adj, return_attention=True)
        else:
            x = self.gat_layer1(x, self.adj)  # (B*9, 12, 2048)
        
        # Self-Attention after first GAT
        if return_attention:
            x, self_attn1 = self.self_attn1(x, return_attention=True)
        else:
            x = self.self_attn1(x)  # (B*9, 12, 2048)

        # Second GAT layer - uses output from self_attn1
        if return_attention:
            x, gat_attn2 = self.gat_layer2(x, self.adj, return_attention=True)
        else:
            x = self.gat_layer2(x, self.adj)  # (B*9, 12, 1024)
        
        # Self-Attention after second GAT
        if return_attention:
            x, self_attn2 = self.self_attn2(x, return_attention=True)
        else:
            x = self.self_attn2(x)  # (B*9, 12, 1024)

        x = x.view(b, num_frames, num_people, -1)  # (B, 9, 12, 1024)
        x = x.permute(0, 2, 1, 3)  # (B, 12, 9, 1024)
        x = x.contiguous().view(b * num_people, num_frames, -1)  # (B*12, 9, 1024)

        lstm_out, _ = self.lstm(x)  # (B*12, 9, 512)
        lstm_out = self.layer_norm1(lstm_out)  # (B*12, 9, 512)

        x = x[:, -1, :]  # (B*12, 1024)
        lstm_out = lstm_out[:, -1, :]  # (B*12, 512)
        
        x = self.proj(x)  # (B*12, 512)
        x = self.layer_norm2(x + lstm_out)  # (B*12, 512)

        x = x.contiguous().view(b, -1)  # (B, 12*512)

        out = self.classifier(x)  # (B, num_classes)
        
        if return_attention:
            return out, {
                'gat_layer1': gat_attn1, 
                'self_attn1': self_attn1,
                'gat_layer2': gat_attn2,
                'self_attn2': self_attn2
            }
        return out


def collate_group_fn(batch):
    """Collate function to pad bounding boxes to 12 per frame."""
    clips, labels = zip(*batch)

    max_bboxes = 12
    padded_clips = []

    for clip in clips:
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros(
                (max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4))
            )
            clip = torch.cat((clip, clip_padding), dim=0)
        padded_clips.append(clip)

    padded_clips = torch.stack(padded_clips)  # (B, 12, T, C, H, W)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_clips, labels
