import torch
import torch.nn as nn


def medium_cnn(checkpoint_path: str = "", num_classes: int=10, score: bool=False) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = nn.Sequential(
        nn.Conv2d(1, 32, 3, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, 1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(9216, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_classes),
        nn.LogSoftmax(dim=1) if not score else nn.Sequential(),
    )
    net = net.to(device)
    if checkpoint_path:
        net.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return net