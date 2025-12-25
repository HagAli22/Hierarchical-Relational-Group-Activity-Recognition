"""
RCRG-2R-21C-Temporal Model
===========================
Temporal relational model with 2 Relational layers (2R) and 2+1 Cliques (21C).
Layer 1: 2 cliques (teams), Layer 2: 1 clique (all players).
Uses LSTM for temporal modeling across 9 frames.
"""

import torch
import torch.nn as nn
from models.non_temporal_model.RelationalGNNLayer import clique_adjacency, RelationalGNNLayer


class RCRG_2R_21C_Temporal(nn.Module):
    def __init__(self, person_classifier, num_classes=8, feature_dim=2048):
        super(RCRG_2R_21C_Temporal, self).__init__()

        self.person_feature_extractor = person_classifier.resnet50
        for param in self.person_feature_extractor.parameters():
            param.requires_grad = False  # Freeze person feature extractor

        # First relational layer: 2048 -> 256 (2 cliques - teams)
        self.relation_layer1 = RelationalGNNLayer(in_dim=feature_dim, out_dim=256)
        
        # Second relational layer: 256 -> 128 (1 clique - all players)
        self.relation_layer2 = RelationalGNNLayer(in_dim=256, out_dim=128)

        self.hidden_size = 512
        self.num_layers = 2

        # LSTM input: 256 (team1: 128 + team2: 128)
        self.lstm = nn.LSTM(256, self.hidden_size, num_layers=self.num_layers, batch_first=True, dropout=0.5)

        # Pool across team members
        self.scene_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):  # x: (B, 12, 9, c, h, w)
        b, k, t, c, h, w = x.size()
        x = x.view(b * t * k, c, h, w)  # (B*9*12, c, h, w)

        x = self.person_feature_extractor(x)  # (B*9*12, 2048, 1, 1)
        x = x.view(b * t, k, -1)  # (B*9, 12, 2048)

        x = self.relation_layer1(x, clique_adjacency(K=12, num_cliques=2))  # (B*9, 12, 256)

        x = self.relation_layer2(x, clique_adjacency(K=12, num_cliques=1))  # (B*9, 12, 128)
        
        x = x.permute(0, 2, 1)  # (B*9, 128, 12)
        team1 = x[:, :, :6]  # (B*9, 128, 6)
        team2 = x[:, :, 6:]  # (B*9, 128, 6)

        team1 = self.scene_pool(team1).squeeze(-1)  # (B*9, 128)
        team2 = self.scene_pool(team2).squeeze(-1)  # (B*9, 128)

        x = torch.cat([team1, team2], dim=1)  # (B*9, 256)
        x = x.view(b, t, -1)  # (B, 9, 256)

        x, _ = self.lstm(x)  # (B, 9, 512)

        x = x[:, -1, :]  # (B, 512)

        out = self.classifier(x)  # (B, num_classes)
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

    padded_clips = torch.stack(padded_clips)  # (B, 12, T, c, h, w)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_clips, labels
