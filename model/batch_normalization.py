import numpy as np
from typing import Tuple, List
import torch

class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        x = torch.tensor(x)
        #print(x.shape)
        running_mean = torch.tensor(running_mean).view(1,-1)
        running_var = torch.tensor(running_var).view(1,-1)
        #print("runningmean:", running_mean.shape)
        #gamma = torch.tensor(gamma) # torch learnable parameters
        #beta = torch.tensor(beta)
        if training:
            mean_x = torch.tensor(x).mean(dim=0, keepdims=True)#.item()
            #print(x.shape, mean_x.shape)

            var_x = torch.tensor(x).var(dim=0, keepdims=True, unbiased=False)#.item(
            running_mean = (1 - momentum) * running_mean + momentum * mean_x 
            running_var = (1 - momentum) * running_var + momentum * var_x
            x_ = (x - mean_x) / torch.sqrt(var_x + eps)
            out = torch.tensor(gamma) * x_ + torch.tensor(beta)
        else:
            x_ = (x - running_mean) / torch.sqrt(running_var + eps)
            out = torch.tensor(gamma) * x_ + torch.tensor(beta)
        return(
                        [[round(float(v), 4) for v in row] for row in out],
                        [round(float(v), 4) for v in running_mean.squeeze(0)],
                        [round(float(v), 4) for v in running_var.squeeze(0)]
                    )
