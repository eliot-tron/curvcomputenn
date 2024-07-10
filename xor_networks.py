import torch
import torch.nn as nn


def oneD_probability_to_twoD_class(input_proba):
    return torch.stack((1 - input_proba, input_proba), dim=-1)

def xor_net(checkpoint_path: str = "", hid_size = 8, score: bool=False, non_linearity: nn.Module=nn.ReLU()) -> nn.Module:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = nn.Sequential(
        nn.Linear(2, hid_size),
        non_linearity,
        nn.Linear(hid_size, 2),
        nn.LogSoftmax(dim=1) if not score else nn.Sequential(),
    )
    # net = net.to(device)
    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path)) #, map_location=device))
    return net

def xor_net_old(checkpoint_path: str = "", hid_size = 8, score: bool=False, non_linearity=nn.Sigmoid()) -> nn.Module:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = nn.Sequential(
        nn.Linear(2, hid_size),
        non_linearity,
        nn.Linear(hid_size, 1),
        nn.Sigmoid() if not score else nn.Sequential(),
    )
    # net = net.to(device)
    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path)) #, map_location=device))
    return net
