

import torch
import torch.nn as nn

class RelationalGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_dim, out_dim),
            # nn.ReLU()
        )

    def forward(self, x, adj):
        """
        x   : (B, K, D_in)
        adj : (K, K)  binary adjacency matrix
        """
        B, K, D = x.shape

        Pi = x.unsqueeze(2)        # (B,K,1,D)
        Pj = x.unsqueeze(1)        # (B,1,K,D)

        pairs = torch.cat([
            Pi.expand(-1, K, K, -1),
            Pj.expand(-1, K, K, -1)
        ], dim=-1)                 # (B,K,K,2D)

        messages = self.mlp(pairs) # (B,K,K,D_out)

        mask = adj.view(1, K, K, 1).to(x.device)
        messages = messages * mask

        out = messages.sum(dim=2)  # (B,K,D_out)
        return out
    

def clique_adjacency(K=11, num_cliques=1):
    """
    K: number of nodes (players)
    num_cliques: number of cliques
    """
    adj = torch.zeros(K, K)
    clique_size = K // num_cliques

    for i in range(num_cliques):
        start = i * clique_size
        end = start + clique_size
        adj[start:end, start:end] = 1

    adj.fill_diagonal_(0)
    return adj
