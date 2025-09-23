import torch

a = torch.zeros(10, 20, 30)
print(a.shape)
print(a[:, 0, :].shape)
