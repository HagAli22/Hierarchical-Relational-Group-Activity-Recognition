"""
RCRG-3R-421C-conc Model
====================
Relational model with 3 Relational layers (3R) and 4+2+1 Cliques (421C).
Uses concatenation instead of pooling for scene representation.
Layer 1: 4 cliques (3 players each), Layer 2: 2 cliques (teams), Layer 3: 1 clique (all players).
"""

import torch
import torch.nn as nn
from models.non_temporal_model.RelationalGNNLayer import clique_adjacency, RelationalGNNLayer


class RCRG_3R_421C_conc(nn.Module):
    def __init__(self, person_classifier, num_classes=8, feature_dim=2048):
        super(RCRG_3R_421C_conc, self).__init__()

        self.person_feature_extractor = person_classifier.resnet50
        for param in self.person_feature_extractor.parameters():
            param.requires_grad = False  # Freeze person feature extractor

        # First relational layer: 4096 -> 512
        self.relation_layer1 = RelationalGNNLayer(in_dim=feature_dim, out_dim=512)
        
        # Second relational layer: 1024 -> 256
        self.relation_layer2 = RelationalGNNLayer(in_dim=512, out_dim=256)

        # third relational layer: 512 -> 128
        self.relation_layer3 = RelationalGNNLayer(in_dim=256, out_dim=128)

        # Pool across all persons
        self.scene_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=12*128, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=num_classes)
        )

    def forward(self, x):  # x: (B, 12, 1, c, h, w)
        b, k, t, c, h, w = x.size()
        x = x.squeeze(2)  # (B, 12, c, h, w)
        x = x.view(b * k, c, h, w)  # (B*12, c, h, w)

        x = self.person_feature_extractor(x)  # (B*12, 2048, 1, 1)
        x = x.view(b, k, -1)  # (B, 12, 2048)

        x = self.relation_layer1(x, clique_adjacency(K=12,num_cliques=4)) # (B, 12, 512)

        x= self.relation_layer2(x, clique_adjacency(K=12,num_cliques=2)) # (B, 12, 256)

        x= self.relation_layer3(x, clique_adjacency(K=12,num_cliques=1)) # (B, 12, 128)

        x = x.view(b,-1) # (B, 12*128=1536)
        

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
