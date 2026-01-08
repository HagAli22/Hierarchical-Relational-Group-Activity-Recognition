"""
RCRG-2R-11C-conc-Temp-GAT Model
================================
Temporal relational model with 2 Relational Units and LSTM for temporal modeling.
Uses PyTorch Geometric MessagePassing for graph-based attention.
"""

import torch
import torch.nn as nn
import itertools

try:
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import softmax
    PYGEOMETRIC_AVAILABLE = True
except ImportError:
    PYGEOMETRIC_AVAILABLE = False
    print("Warning: torch_geometric not available, using fallback implementation")


class GraphRelationalAttention(MessagePassing):
    """
    Graph-based Relational Attention layer with multi-head attention and FFN.
    Processes player interactions using pairwise feature attention mechanism.
    """
    def __init__(self, in_channels, out_channels, num_heads=4, hidden_size=1024, dropout_rate=0.5):
        super(GraphRelationalAttention, self).__init__(aggr='add')
        
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        self.scale = self.head_dim ** 0.5
        
        self.query = nn.Linear(in_channels, in_channels)
        self.key = nn.Linear(in_channels, in_channels)
        self.value = nn.Linear(2 * in_channels, in_channels)  # Pairwise features!
        
        self.ln1 = nn.LayerNorm(in_channels)
        self.dr1 = nn.Dropout(dropout_rate)
        
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, out_channels),
        )
        
        self.ln2 = nn.LayerNorm(out_channels)
        self.dr2 = nn.Dropout(dropout_rate)
    
    def forward(self, x, edge_index):
        x_att = self.propagate(edge_index, x=x)
        x_att = x + self.dr1(x_att)
        x_att = self.ln1(x_att)
        
        x_ffn = self.ffn(x_att)
        out = x_att + self.dr2(x_ffn)
        out = self.ln2(out)
        return out
    
    def message(self, x_i, x_j, index, ptr, size_i):
        """
        x_i: Features of the receiving node
        x_j: Features of the sending node
        """
        batch, edges, _ = x_i.shape
        
        query = self.query(x_i).view(batch, edges, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.key(x_j).view(batch, edges, self.num_heads, self.head_dim).transpose(1, 2)
        value = self.value(torch.cat([x_i, x_j], dim=-1)).view(batch, edges, self.num_heads, self.head_dim).transpose(1, 2)
        
        e_ij = (query * key).sum(dim=-1) / self.scale
        a_ij = softmax(e_ij, index, ptr, num_nodes=size_i, dim=-1)
        
        return (a_ij.unsqueeze(-1) * value).view(batch, edges, self.num_heads * self.head_dim)
    
    def update(self, aggr_out):
        return aggr_out


class RCRG_2R_11C_conc_Temp_GAT(nn.Module):
    def __init__(self, person_classifier, num_classes=8, feature_dim=2048):
        super(RCRG_2R_11C_conc_Temp_GAT, self).__init__()
        
        self.resnet50 = person_classifier.resnet50
        for param in self.resnet50.parameters():
            param.requires_grad = False
        
        self.gra1 = GraphRelationalAttention(
            in_channels=feature_dim,
            out_channels=feature_dim,
            num_heads=4,
            dropout_rate=0.5
        )
        
        self.gra2 = GraphRelationalAttention(
            in_channels=feature_dim,
            out_channels=feature_dim,
            num_heads=4,
            dropout_rate=0.5
        )
        
        self.proj_layer = nn.Linear(2048, 512)
        self.norm_temporal = nn.LayerNorm(512)
        self.norm_fusion = nn.LayerNorm(512)
        
        self.temporal_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=512,
            batch_first=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(12 * 512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )
        
        # Store edge_index as buffer (will be created on first forward)
        self.register_buffer('edge_index', None)
    
    def _get_edge_index(self, num_nodes, device):
        """Generate fully connected edge index for all players."""
        if self.edge_index is None or self.edge_index.device != device:
            edges = [(i, j) for i, j in itertools.permutations(range(num_nodes), 2)]
            self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(device)
        return self.edge_index
    
    def forward(self, x):
        b, bb, seq, c, h, w = x.shape  # batch, bbox, frames, channels, height, width
        
        x = x.view(b * bb * seq, c, h, w)
        x = self.resnet50(x)  # (b*bb*seq, 2048, 1, 1)
        x = x.view(b * seq, bb, -1)  # (b*seq, bb, 2048)
        
        # Get edge index for graph
        edge_index = self._get_edge_index(bb, x.device)
        
        # Graph relational attention layers
        x_gra1 = self.gra1(x, edge_index)  # (b*seq, bb, 2048)
        x_gra2 = self.gra2(x_gra1, edge_index)  # (b*seq, bb, 2048)
        
        # Reshape for LSTM: (b*bb, seq, 2048)
        x_gra2 = x_gra2.view(b, seq, bb, -1).permute(0, 2, 1, 3).contiguous()
        x_gra2 = x_gra2.view(b * bb, seq, -1)
        
        # LSTM temporal modeling
        x_temporal, (h, c) = self.temporal_lstm(x_gra2)  # (b*bb, seq, 512)
        x_temporal = self.norm_temporal(x_temporal)
        
        # Take last timestep
        x_spatial_last = x_gra2[:, -1, :]  # (b*bb, 2048)
        x_temporal_last = x_temporal[:, -1, :]  # (b*bb, 512)
        
        # Combine with projection
        x = self.norm_fusion(self.proj_layer(x_spatial_last) + x_temporal_last)  # (b*bb, 512)
        
        x = x.contiguous().view(b, -1)  # (b, bb*512)
        x = self.classifier(x)  # (b, num_classes)
        
        return x


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
    
    padded_clips = torch.stack(padded_clips)
    labels = torch.tensor(labels, dtype=torch.long)
    
    return padded_clips, labels