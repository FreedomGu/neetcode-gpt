import torch
import torch.nn as nn
from torchtyping import TensorType

class Solution(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.layer1 = nn.Embedding(vocabulary_size, 16)
        self.layer2 = nn.Linear(16, 1)
        self.act = nn.Sigmoid()
        #pass

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places

        #pass
        x1 = self.layer1(x).mean(dim=1)
        x2 = self.act(self.layer2(x1))
        return torch.round(x2,decimals=4)

