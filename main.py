import torch

with torch.no_grad():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 10, bias=False),
        torch.nn.ReLU()
    )