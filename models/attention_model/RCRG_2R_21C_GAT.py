"""
RCRG-2R-21C-GAT Model
======================
Relational model with 2 Graph Attention layers (2R) and 2+1 Cliques (21C).
Layer 1: 2 cliques (teams) with attention
Layer 2: 1 clique (all players) with attention

Uses single-head GAT instead of simple message passing.
"""

import torch
import torch.nn as nn
from models.attention_model.RelationalGATLayer import RelationalGATLayer, clique_adjacency


class RCRG_2R_21C_GAT(nn.Module):
    def __init__(self, person_classifier, num_classes=8, feature_dim=2048):
        super(RCRG_2R_21C_GAT, self).__init__()

        self.person_feature_extractor = person_classifier.resnet50
        for param in self.person_feature_extractor.parameters():
            param.requires_grad = False  # Freeze person feature extractor

        # First GAT layer: 2048 -> 256 (2 cliques - teams)
        self.gat_layer1 = RelationalGATLayer(in_dim=feature_dim, out_dim=256, dropout=0.5)
        
        # Second GAT layer: 256 -> 128 (1 clique - all players)
        self.gat_layer2 = RelationalGATLayer(in_dim=256, out_dim=128, dropout=0.3)

        # Pool across team members
        self.scene_pool = nn.AdaptiveMaxPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=num_classes)
        )
        
        # Store adjacency matrices
        self.register_buffer('adj1', clique_adjacency(K=12, num_cliques=2))
        self.register_buffer('adj2', clique_adjacency(K=12, num_cliques=1))

    def forward(self, x, return_attention=False):
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 12, 1, C, H, W)
            return_attention: If True, return attention weights for visualization
            
        Returns:
            out: Class logits (B, num_classes)
            attention_weights: (optional) Dict with attention weights from each layer
        """
        b, k, t, c, h, w = x.size()
        x = x.squeeze(2)  # (B, 12, C, H, W)
        x = x.view(b * k, c, h, w)  # (B*12, C, H, W)

        # Extract person features
        x = self.person_feature_extractor(x)  # (B*12, 2048, 1, 1)
        x = x.view(b, k, -1)  # (B, 12, 2048)

        # First GAT layer: team-level attention (2 cliques)
        if return_attention:
            x, attn1 = self.gat_layer1(x, self.adj1, return_attention=True)
        else:
            x = self.gat_layer1(x, self.adj1)  # (B, 12, 256)

        # Second GAT layer: scene-level attention (1 clique)
        if return_attention:
            x, attn2 = self.gat_layer2(x, self.adj2, return_attention=True)
        else:
            x = self.gat_layer2(x, self.adj2)  # (B, 12, 128)
        
        # Pool teams separately then concatenate
        x = x.permute(0, 2, 1)  # (B, 128, 12)
        team1 = x[:, :, :6]  # (B, 128, 6)
        team2 = x[:, :, 6:]  # (B, 128, 6)

        team1 = self.scene_pool(team1).squeeze(-1)  # (B, 128)
        team2 = self.scene_pool(team2).squeeze(-1)  # (B, 128)

        x = torch.cat([team1, team2], dim=1)  # (B, 256)

        out = self.classifier(x)  # (B, num_classes)
        
        if return_attention:
            return out, {'layer1': attn1, 'layer2': attn2}
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
