"""
B1-NoRelations Model
====================
Baseline model without relational reasoning for group activity recognition.
"""

import torch
import torch.nn as nn


class B1_NoRelations(nn.Module):
    def __init__(self, num_classes=8):
        super(B1_NoRelations, self).__init__()
        
        self.shared = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=4096, out_features=2048),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=2048, out_features=128)
        )
        # pool across the player dimension (N players) -> keep channel dim = 128
        self.scene_pool = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, x):  # x: (B, 12, 1, 4096)
        b, k, t, f = x.size()
        x = x.squeeze(2)  # (B, 12, 4096)
        x = x.view(b * k, f)  # (B*12, 4096)
        x = self.shared(x)  # (B*12, 128)

        x = x.view(b, k, 128).permute(0, 2, 1)  # (B, 128, 12)
        x = self.scene_pool(x).squeeze(-1)  # (B, 128)

        out = self.classifier(x)  # (B, num_classes)
        return out


def collate_group_fn(batch):
    """Collate function to pad bounding boxes to 12 per frame."""
    clips, labels = zip(*batch)
    
    max_bboxes = 12
    NEG_FILL = -1e9  # ensures padded players never affect max-pool

    padded_clips = []

    for clip in clips:
        # clip: (num_bboxes, num_frames, feature_dim), feature_dim=4096
        num_bboxes = clip.size(0)
        if num_bboxes < max_bboxes:
            clip_padding = torch.full(
                (max_bboxes - num_bboxes, clip.size(1), clip.size(2)), 
                NEG_FILL
            )
            clip = torch.cat((clip, clip_padding), dim=0)
        padded_clips.append(clip)

    padded_clips = torch.stack(padded_clips)  # (B, 12, T, 4096)
    labels = torch.tensor(labels, dtype=torch.long)

    return padded_clips, labels
