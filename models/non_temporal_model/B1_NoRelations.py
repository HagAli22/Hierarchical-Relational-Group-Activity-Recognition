"""
B1-NoRelations Model
====================
Baseline model without relational reasoning for group activity recognition.
"""

import torch
import torch.nn as nn


class B1_NoRelations(nn.Module):
    def __init__(self, person_classifier, num_classes=8, feature_dim=2048):
        super(B1_NoRelations, self).__init__()

        self.person_feature_extractor = person_classifier.resnet50
        for param in self.person_feature_extractor.parameters():
            param.requires_grad = False  # Freeze person feature extractor
            
        self.shared = nn.Sequential(
            # nn.Dropout(0.2),
            # nn.Linear(in_features=feature_dim, out_features=1024),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(in_features=feature_dim, out_features=128)
        )
        # pool across the each team dimension (6 players) -> keep channel dim = 128
        self.scene_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=128, out_features=num_classes)
        )

    def forward(self, x):  # x: (B, 12, 1, c,h,w) 

        b, k, t, c , h , w = x.size()
        x = x.squeeze(2)  # (B, 12, c,h,w)
        x = x.view(b * k, c,h,w)  # (B*12, c,h,w)

        x = self.person_feature_extractor(x)  # (B*12, 2048, 1, 1)
        x = x.view(b * k, -1)  # (B*12, 2048)

        x = self.shared(x)  # (B*12, 128)

        x = x.view(b, k, 128).permute(0, 2, 1)  # (B, 128, 12)

        team1 = x[:, :, :6]  # (B, 128, 6)
        team2 = x[:, :, 6:]  # (B, 128, 6)

        team1 = self.scene_pool(team1).squeeze(-1)  # (B, 128)
        team2 = self.scene_pool(team2).squeeze(-1)  # (B, 128)

        x = torch.cat([team1, team2], dim=1)  # (B, 256)

        out = self.classifier(x)  # (B, num_classes)
        return out


def collate_group_fn(batch):
    """Collate function to pad bounding boxes to 12 per frame."""
    clips, labels = zip(*batch)
    
    max_bboxes = 12

    padded_clips = []

    for clip in clips:
        # clip: (num_bboxes, num_frames, c ,h ,c)
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.zeros(
                (max_bboxes - num_bboxes, clip.size(1), clip.size(2), clip.size(3), clip.size(4))
            )
            clip = torch.cat((clip, clip_padding), dim=0)
        padded_clips.append(clip)

    padded_clips = torch.stack(padded_clips)  # (B, 12, T, c ,h ,w)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_clips, labels
