import torch
from torch import nn


class Sampler(nn.Module):

    YES_TOKEN_ID = 9693
    NO_TOKEN_ID = 2152
    # YES_TOKEN_ID = 30487
    # NO_TOKEN_ID = 42224

    def __init__(self):
        super().__init__()

    # def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
    #     logits = logits.float()
    #     greedy_tokens = logits.argmax(dim=-1)
    #     logits.div_(temperatures.unsqueeze(dim=1))
    #     probs = torch.softmax(logits, dim=-1, dtype=torch.float)
    #     sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1) + 1e-10).argmax(dim=-1)  
    #     return torch.where(temperatures == 0, greedy_tokens, sample_tokens)

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        del temperatures  # only greedy yes/no sampling is supported
        logits = logits.float()
        # log_probs = torch.log_softmax(logits, dim=-1)
        yes_scores = logits[..., self.YES_TOKEN_ID]
        no_scores = logits[..., self.NO_TOKEN_ID]
        yes_token = torch.full_like(yes_scores, self.YES_TOKEN_ID, dtype=torch.long)
        no_token = torch.full_like(no_scores, self.NO_TOKEN_ID, dtype=torch.long)
        # Greedy decision between yes/no for every row
        sampled = torch.where(yes_scores >= no_scores, yes_token, no_token)

        # Compute perplexity from selected tokens' probabilities
        # selected_log_probs = log_probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
        # perplexity = torch.exp(-selected_log_probs)
        # print(f"[sampler] perplexity: {perplexity.mean().item():.4f}")

        return sampled
