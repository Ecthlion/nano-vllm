import torch
from torch import nn


class Sampler(nn.Module):

    YES_TOKEN_ID = 9693
    NO_TOKEN_ID = 2152

    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        del temperatures  # only greedy yes/no sampling is supported
        logits = logits.float()
        yes_scores = logits[..., self.YES_TOKEN_ID]
        no_scores = logits[..., self.NO_TOKEN_ID]
        yes_token = torch.full_like(yes_scores, self.YES_TOKEN_ID, dtype=torch.long)
        no_token = torch.full_like(no_scores, self.NO_TOKEN_ID, dtype=torch.long)
        # Greedy decision between yes/no for every row
        return torch.where(yes_scores >= no_scores, yes_token, no_token)
