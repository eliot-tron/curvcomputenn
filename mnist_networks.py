import torch
import torch.nn as nn
from copy import deepcopy

def medium_cnn(checkpoint_path: str = "", num_classes: int=10, score: bool=False, non_linearity:nn.Module=nn.ReLU(), maxpool=False) -> nn.Module:
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
    net = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        deepcopy(non_linearity),
        nn.Conv2d(32, 64, 3, 1),
        deepcopy(non_linearity),
        nn.AvgPool2d(2) if not maxpool else nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * (28 - 2 * 2) * (28 - 2 * 2) // (2**2), 128),
        deepcopy(non_linearity),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1) if not score else nn.Sequential(),
    )
    # net = net.to(device)
    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path)) # , map_location=device))
    return net
