"""
Relational Graph Attention Layer (Multi-Head GAT)
==================================================
Implements multi-head attention mechanism for relational reasoning between players.
Supports attention entropy regularization to prevent attention collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalGATLayer(nn.Module):
    """
    Multi-Head Graph Attention Layer for relational reasoning.
    
    Uses 2 attention heads with averaging (not concat) for stability.
    Supports attention entropy regularization.
    """
    
    def __init__(self, in_dim, out_dim, num_heads=2, dropout=0.3, negative_slope=0.2):
        super().__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        # Linear transformation for features (shared across heads)
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        
        # Separate attention mechanisms for each head
        self.attentions = nn.ModuleList([
            nn.Linear(2 * out_dim, 1, bias=False) for _ in range(num_heads)
        ])
        
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
        for attn in self.attentions:
            nn.init.xavier_uniform_(attn.weight)
    
    def forward(self, x, adj, return_attention=False):
        """
        Forward pass with multi-head attention mechanism.
        
        Args:
            x: Node features (B, K, in_dim)
            adj: Adjacency matrix (K, K) - defines which nodes can attend to each other
            return_attention: If True, also return attention weights for visualization
            
        Returns:
            out: Updated node features (B, K, out_dim)
            attention_weights: (optional) Attention weights (B, num_heads, K, K)
        """
        B, K, _ = x.shape
        device = x.device
        
        # Transform features: (B, K, out_dim)
        h = self.W(x)
        
        # Prepare for attention computation
        h_i = h.unsqueeze(2).expand(-1, -1, K, -1)  # (B, K, K, out_dim)
        h_j = h.unsqueeze(1).expand(-1, K, -1, -1)  # (B, K, K, out_dim)
        
        # Concatenate pairs: (B, K, K, 2*out_dim)
        pairs = torch.cat([h_i, h_j], dim=-1)
        
        # Mask based on adjacency matrix
        adj = adj.to(device)
        mask = adj.unsqueeze(0)  # (1, K, K)
        
        # Compute attention for each head
        all_attention_weights = []
        head_outputs = []
        
        for head_idx, attn_layer in enumerate(self.attentions):
            # Compute attention scores: (B, K, K)
            e = attn_layer(pairs).squeeze(-1)
            e = self.leaky_relu(e)
            
            # Mask non-neighbors
            e = e.masked_fill(mask == 0, float('-inf'))
            
            # Softmax over neighbors
            attention_weights = F.softmax(e, dim=-1)
            attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
            
            # Apply dropout
            attention_weights_dropped = self.dropout(attention_weights)
            
            # Aggregate: (B, K, out_dim)
            head_out = torch.bmm(attention_weights_dropped, h)
            head_outputs.append(head_out)
            all_attention_weights.append(attention_weights)
        
        # Average heads (not concat) for stability
        out = torch.stack(head_outputs, dim=0).mean(dim=0)  # (B, K, out_dim)
        
        # Residual connection + layer norm
        out = self.layer_norm(out + self.residual(x))
        
        if return_attention:
            # Stack attention weights: (B, num_heads, K, K)
            attn_stack = torch.stack(all_attention_weights, dim=1)
            return out, attn_stack
        return out
    
    def compute_attention_entropy(self, attention_weights):
        """
        Compute entropy of attention weights for regularization.
        Higher entropy = more uniform attention = better diversity.
        
        Args:
            attention_weights: (B, num_heads, K, K)
            
        Returns:
            entropy: Scalar entropy value (negative for loss minimization)
        """
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(
            attention_weights * torch.log(attention_weights + eps),
            dim=-1
        )  # (B, num_heads, K)
        
        # Average over all
        return entropy.mean()


def attention_entropy_loss(attention_weights, lambda_entropy=0.01):
    """
    Compute attention entropy regularization loss.
    Encourages diverse attention patterns (prevents collapse to single node).
    
    Args:
        attention_weights: (B, num_heads, K, K)
        lambda_entropy: Regularization strength (default: 0.01)
        
    Returns:
        loss: Negative entropy (to maximize entropy via minimization)
    """
    eps = 1e-8
    
    # Compute entropy
    entropy = -torch.sum(
        attention_weights * torch.log(attention_weights + eps),
        dim=-1
    )  # (B, num_heads, K)
    
    # We want to MAXIMIZE entropy, so return NEGATIVE entropy as loss
    return -lambda_entropy * entropy.mean()


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
