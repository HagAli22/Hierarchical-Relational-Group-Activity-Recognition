"""
Relational Graph Attention Layer (Single-Head GAT)
===================================================
Implements attention mechanism for relational reasoning between players.
Instead of treating all neighbors equally, learns attention weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGATLayer(nn.Module):
    """
    Single-head Graph Attention Layer for relational reasoning.
    
    Computes attention weights between player pairs based on their features,
    then aggregates neighbor features weighted by attention scores.
    """
    
    def __init__(self, in_dim, out_dim, dropout=0.3, negative_slope=0.2):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Linear transformation for features
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
        # Attention mechanism: a^T [Wh_i || Wh_j]
        self.attention = nn.Linear(2 * out_dim, 1, bias=False)
        
        # LeakyReLU for attention scores
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for output
        self.layer_norm = nn.LayerNorm(out_dim)
        
        # Residual projection if dimensions differ
        self.residual = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attention.weight)
    
    def forward(self, x, adj, return_attention=False):
        """
        Forward pass with attention mechanism.
        
        Args:
            x: Node features (B, K, in_dim)
            adj: Adjacency matrix (K, K) - defines which nodes can attend to each other
            return_attention: If True, also return attention weights for visualization
            
        Returns:
            out: Updated node features (B, K, out_dim)
            attention_weights: (optional) Attention weights (B, K, K)
        """
        B, K, _ = x.shape
        device = x.device
        
        # Transform features: (B, K, out_dim)
        h = self.W(x)
        
        # Prepare for attention computation
        # h_i: (B, K, 1, out_dim) -> (B, K, K, out_dim)
        h_i = h.unsqueeze(2).expand(-1, -1, K, -1)
        # h_j: (B, 1, K, out_dim) -> (B, K, K, out_dim)
        h_j = h.unsqueeze(1).expand(-1, K, -1, -1)
        
        # Concatenate pairs: (B, K, K, 2*out_dim)
        pairs = torch.cat([h_i, h_j], dim=-1)
        
        # Compute attention scores: (B, K, K, 1) -> (B, K, K)
        e = self.attention(pairs).squeeze(-1)
        e = self.leaky_relu(e)
        
        # Mask attention scores based on adjacency matrix
        # adj: (K, K) -> (1, K, K)
        adj = adj.to(device)
        mask = adj.unsqueeze(0)
        
        # Set non-neighbors to -inf so softmax gives 0
        e = e.masked_fill(mask == 0, float('-inf'))
        
        # Softmax over neighbors: (B, K, K)
        attention_weights = F.softmax(e, dim=-1)
        
        # Handle case where a node has no neighbors (all -inf -> nan after softmax)
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
        
        # Apply dropout to attention weights
        attention_weights = self.dropout(attention_weights)
        
        # Aggregate neighbor features weighted by attention: (B, K, out_dim)
        out = torch.bmm(attention_weights, h)
        
        # Residual connection + layer norm
        out = self.layer_norm(out + self.residual(x))
        
        if return_attention:
            return out, attention_weights
        return out


def clique_adjacency(K=12, num_cliques=1):
    """
    Create adjacency matrix for clique-based graph structure.
    
    Args:
        K: Number of nodes (players)
        num_cliques: Number of cliques to divide players into
        
    Returns:
        adj: Adjacency matrix (K, K)
    """
    adj = torch.zeros(K, K)
    clique_size = K // num_cliques

    for i in range(num_cliques):
        start = i * clique_size
        end = start + clique_size
        adj[start:end, start:end] = 1

    # Remove self-loops
    adj.fill_diagonal_(0)
    return adj
