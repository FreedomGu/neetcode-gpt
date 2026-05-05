import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch),
        # sample batches from data, run forward/backward, update weights.
        # Return the final loss rounded to 4 decimals.
        optim = AdamW(model.parameters(), lr= lr)
        for epoch in range(epochs):
            torch.manual_seed(epoch)
            indx = torch.randint(len(data)-context_length, (batch_size,))
            offset = torch.arange(context_length)
            indx = indx[:, None] + offset[None, :]

            X = data[indx]
            Y = data[indx+1]
            output = model(X) # batch, context_length, vocab_size

            B, T, C = output.shape
            output = output.view(-1,C)
            Y = Y.view(-1)

            loss = F.cross_entropy(output, Y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        return round(loss.detach().item(),4)
