import argparse
import random
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cifar10_networks import medium_cnn

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: Optimizer, epoch: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_interval = len(loader) // 10
    device = next(model.parameters()).device
    model.train()
    steps = []
    for batch_idx, (data, target) in enumerate(loader, start=1):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        if batch_idx % max(log_interval, 1) == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            steps.append(batch_idx)
        optimizer.step()
    steps = torch.tensor(steps)
    return steps


def test(model: nn.Module, loader: DataLoader) -> float:
    device = next(model.parameters()).device
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(loader.dataset),
            100.0 * correct / len(loader.dataset),
        )
    )
    return test_loss


def cifar10_loader(batch_size: int, train: bool) -> DataLoader:
    loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(
            "data",
            train=train,
            download=True,
            transform=transform
        ),
        batch_size=batch_size,
        shuffle=train,
        num_workers=1,
        pin_memory=True,
    )
    return loader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a basic model on CIFAR10",
        usage="python3 mnist_training.py [--batch-size BATCH-SIZE "
        "--epochs EPOCHS --lr LR --seed SEED --output-dir OUTPUT-DIR]",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoint",
        help="Model checkpoint output directory",
    )

    args = parser.parse_args(sys.argv[1:])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    non_linearity = 'ReLU'
    non_linearity_dict = {
        'ReLU': nn.ReLU(),
        'Sigmoid': nn.Sigmoid(),
    }
    maxpool = True
    pool = 'maxpool' if maxpool else 'avgpool'

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = medium_cnn(num_classes=10, non_linearity=non_linearity_dict[non_linearity], maxpool=maxpool)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader = cifar10_loader(args.batch_size, train=True)
    test_loader = cifar10_loader(args.batch_size, train=False)

    global_steps = []
    for epoch in range(args.epochs):
        epoch_steps = train_epoch(
            model, train_loader, optimizer, epoch + 1
        )
        global_steps.append(epoch_steps + epoch * len(train_loader))
        test(model, test_loader)
        torch.save(model.state_dict(), output_dir / f"cifar10_medium_cnn_{epoch + 1:02d}_{pool}_{non_linearity}.pt")

    global_steps = torch.cat(global_steps, dim=0)