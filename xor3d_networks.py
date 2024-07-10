import torch
import torch.nn as nn


def xor3d_net(checkpoint_path: str = "", hid_size = 16, score: bool=False, non_linearity: nn.Module=nn.ReLU()) -> nn.Module:
    net = nn.Sequential(
        nn.Linear(3, hid_size),
        non_linearity,
        nn.Linear(hid_size, 2),
        nn.LogSoftmax(dim=1) if not score else nn.Sequential(),
    )
    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path)) #, map_location=device))
    return net
