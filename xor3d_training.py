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

from xor3d_datasets import Xor3dDataset
from xor3d_networks import xor3d_net


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
            pred = output.argmax(dim=1, keepdim=True) # TODO: check choice
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


def xor3d_loader(batch_size: int, train: bool) -> DataLoader:
    dataset = Xor3dDataset(nsample=100000, test=not train, discrete=False)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=1,
        pin_memory=True,
    )

    return loader


def exemplar_batch(batch_size: int, train: bool) -> torch.Tensor:
    dataset = Xor3dDataset()
    examples = []
    for i in range(batch_size):
        examples.append(dataset[i][0])
    batch = torch.stack(examples, dim=0)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a basic model on 3D XOR",
        usage="python3 xor3d_training.py [--batch-size BATCH-SIZE "
        "--epochs EPOCHS --lr LR --seed SEED --output-dir OUTPUT-DIR]",
    )
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--activation", type=str, default="sigmoid", help="Activation function [sigmoid|relu]")
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

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.activation.lower() == "sigmoid":
        non_linearity = nn.Sigmoid()
    elif args.activation.lower() == "relu":
        non_linearity = nn.ReLU()
    model = xor3d_net(non_linearity=non_linearity) 
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader = xor3d_loader(args.batch_size, train=True)
    test_loader = xor3d_loader(args.batch_size, train=False)

    global_steps = []
    for epoch in range(args.epochs):
        epoch_steps = train_epoch(
            model, train_loader, optimizer, epoch + 1
        )
        global_steps.append(epoch_steps + epoch * len(train_loader))
        test(model, test_loader)
        torch.save(model.state_dict(), output_dir / f"xor3d_net_{args.activation.lower()}_{epoch + 1:02d}.pt")

    global_steps = torch.cat(global_steps, dim=0)
