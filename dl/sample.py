import torch

def sample_discrete_action_from_logit(logits):
    return torch.multinominal(logits,num_samples = 1).squeeze(1)
