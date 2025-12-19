"""
RCRG-2R-11C Model
====================
Relational model with 2 Relational layers (2R) and 1 Clique of all 11 people (11C) and concatenation not max pooling.
"""

import torch
import torch.nn as nn


class RCRG_2R_11C_conc(nn.Module):
    def __init__(self, person_classifier, num_classes=8, feature_dim=2048):
        super(RCRG_2R_11C_conc, self).__init__()

        self.person_feature_extractor = person_classifier.resnet50
        for param in self.person_feature_extractor.parameters():
            param.requires_grad = False  # Freeze person feature extractor

        # First relational layer: 4096 -> 256
        self.relation_layer1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=feature_dim * 2, out_features=256),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Second relational layer: 512 -> 128
        self.relation_layer2 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=256 * 2, out_features=128),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=128 * 12, out_features=1024),
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

        # ============ First Relational Layer ============
        # Create all pairs
        Pi = x.unsqueeze(2)  # (B, 12, 1, 2048)
        Pj = x.unsqueeze(1)  # (B, 1, 12, 2048)
        Pi_expanded = Pi.expand(-1, -1, k, -1)  # (B, 12, 12, 2048)
        Pj_expanded = Pj.expand(-1, k, -1, -1)  # (B, 12, 12, 2048)

        pairs1 = torch.cat([Pi_expanded, Pj_expanded], dim=-1)  # (B, 12, 12, 4096)
        pairs1 = pairs1.view(b * k * k, -1)  # (B*144, 4096)
        pairs1 = self.relation_layer1(pairs1)  # (B*144, 256)
        pairs1 = pairs1.view(b, k, k, -1)  # (B, 12, 12, 256)

        # Mask self-relations (i == j)
        mask = (1 - torch.eye(k, device=x.device)).view(1, k, k, 1)
        pairs1 = pairs1 * mask  # (B, 12, 12, 256)

        # Aggregate: sum over j to get per-person relational features
        rel_features1 = pairs1.sum(dim=2)  # (B, 12, 256)

        # ============ Second Relational Layer ============
        Pi2 = rel_features1.unsqueeze(2)  # (B, 12, 1, 256)
        Pj2 = rel_features1.unsqueeze(1)  # (B, 1, 12, 256)
        Pi2_expanded = Pi2.expand(-1, -1, k, -1)  # (B, 12, 12, 256)
        Pj2_expanded = Pj2.expand(-1, k, -1, -1)  # (B, 12, 12, 256)

        pairs2 = torch.cat([Pi2_expanded, Pj2_expanded], dim=-1)  # (B, 12, 12, 512)
        pairs2 = pairs2.view(b * k * k, -1)  # (B*144, 512)
        pairs2 = self.relation_layer2(pairs2)  # (B*144, 128)
        pairs2 = pairs2.view(b, k, k, -1)  # (B, 12, 12, 128)

        # Mask self-relations
        pairs2 = pairs2 * mask  # (B, 12, 12, 128)

        # Aggregate: sum over j
        rel_features2 = pairs2.sum(dim=2)  # (B, 12, 128)

        # ============ Scene-level concatenation ============
        x = rel_features2.view(b, -1)  # (B, 12*128=1536)

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
