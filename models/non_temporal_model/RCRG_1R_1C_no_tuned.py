"""
RCRG-1R-1C-no_tuned Model
====================
Same as RCRG-1R-1C model with 1 Relational layer and 1 Clique for group activity recognition but ImageNet-pretrained ResNet50 without fine-tuning.
"""

import torch
import torch.nn as nn
import torchvision.models as models



class RCRG_1R_1C_no_tuned(nn.Module):
    def __init__(self, num_classes=8, feature_dim=2048):
        super(RCRG_1R_1C_no_tuned, self).__init__()

        self.person_feature_extractor = nn.Sequential(
            *list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1]
        )
        for param in self.person_feature_extractor.parameters():
            param.requires_grad = False  # Freeze person feature extractor
            
        self.shared = nn.Sequential(
            nn.Linear(in_features=feature_dim * 2, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=512, out_features=128)
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
        x = x.view(b , k, -1)  # (B, 12, 2048)

        # Apply broadcasted
        Pi=x.unsqueeze(1) # (B,1,12,2048)
        Pj=x.unsqueeze(2) # (B,12,1,2048)

        Pi_expanded = Pi.expand(-1, k, -1, -1)  # (B,12,12,2048)
        Pj_expanded = Pj.expand(-1, -1, k, -1)  # (B,12,12,2048)

        # Concatenate along the feature dimension
        pairs = torch.cat([Pi_expanded, Pj_expanded], dim=-1)  # (B,12,12,4096)

        # applay MLP for relational reasoning between all pairs
        pairs = pairs.view(b * k * k, -1)  # (B*12*12, 4096)
        pairs = self.shared(pairs)  # (B*12*12, 128)
        pairs = pairs.view(b, k, k, -1)  # (B,12,12,128)


        # the paper does not use self-relations, so we mask at i==j
        mask = torch.ones((k, k), device=x.device) - torch.eye(k, device=x.device)  # (12,12)
        mask = mask.view(1, k, k, 1)  # (1,12,12,1)
        pairs = pairs * mask  # (B,12,12,128)

        # sum over j dimension to get relational feature for each person
        relational_features = pairs.sum(dim=2)  # (B,12,128)

        # Now pool over all persons to get scene-level feature
        relational_features = relational_features.permute(0, 2, 1)  # (B,128,12)

        team1 = relational_features[:, :, :6]  # (B, 128, 6)
        team2 = relational_features[:, :, 6:]  # (B, 128, 6)

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
