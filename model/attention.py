import torch
import torch.nn as nn
from torchtyping import TensorType
import math
class SingleHeadAttention(nn.Module):

    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        #pass
        self.e_dim = embedding_dim
        self.a_dim = attention_dim
        self.k = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.q = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.v = nn.Linear(embedding_dim, attention_dim, bias=False)
    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        B, L , D = embedded.shape
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        Q = self.q(embedded)
        K = self.k(embedded)
        V = self.v(embedded)
        scores = torch.matmul(Q, K.transpose(-1,-2))/ math.sqrt(self.a_dim)
        mask = torch.tril(torch.ones(L, L))
        #print(mask)
        scores = scores.masked_fill(mask==0, value =float('-inf'))
        scores = torch.softmax(scores, dim = -1)
        output = scores @ V
        return torch.round(output, decimals = 4)
