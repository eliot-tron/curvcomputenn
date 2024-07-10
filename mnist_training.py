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

from mnist_networks import medium_cnn
# from model_manifold.data_matrix import batch_data_matrix_trace_rank
# from model_manifold.plot import save_ranks, save_mean_trace, save_images


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: Optimizer, epoch: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_interval = len(loader) // 10
    device = next(model.parameters()).device
    model.train()
    steps = []
    ranks = []
    traces = []
    reference_batch = exemplar_batch(50, train=True).to(device)
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
            # batch_traces, batch_ranks = batch_data_matrix_trace_rank(
            #     model, reference_batch
            # )
            # batch_FIM_traces, batch_FIM_ranks = batch_fisher_matrix_trace_rank(
            #     model, reference_batch
            # )
            # traces.append(batch_traces)
            # ranks.append(batch_ranks)
        optimizer.step()
    steps = torch.tensor(steps)
    # ranks = torch.stack(ranks, dim=1)
    # traces = torch.stack(traces, dim=1)
    return steps, ranks, traces


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


def mnist_loader(batch_size: int, train: bool) -> DataLoader:
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data",
            train=train,
            download=False,
            transform=transforms.Compose(
                [transforms.ToTensor(), 
                #  transforms.Normalize((0.1307,), (0.3081,))
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=train,
        num_workers=1,
        pin_memory=True,
    )
    return loader

def emnist_loader(batch_size: int, train: bool) -> DataLoader:
    dataset = datasets.EMNIST(
                "data_augmented",
                train=train,
                download=False,
                split='letters',
                transform=transforms.Compose(
                    [transforms.ToTensor(), 
                    #  transforms.Normalize((0.1307,), (0.3081,))
                    ]
                ),
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=1,
        pin_memory=True,
    )
    idx = torch.randint(dataset.data.shape[0]-1, (16,))
    idx = torch.arange(0, 31 ,3) + 31*10
    # save_images(dataset.data[idx,...], f'./data_augmented/visualisation_{"train" if train else "test"}', predictions=[dataset.classes[i] for i in dataset.targets[idx,...].int()])

    return loader


def exemplar_batch(batch_size: int, train: bool) -> torch.Tensor:
    dataset = datasets.MNIST(
        "data",
        train=train,
        download=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), 
            #  transforms.Normalize((0.1307,), (0.3081,))
            ]
        ),
    )
    examples = []
    for i in range(batch_size):
        examples.append(dataset[i][0])
    batch = torch.stack(examples, dim=0)
    return batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a basic model on MNIST",
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
    parser.add_argument(
        "--nl",
        type=str,
        metavar='f',
        default="ReLU",
        choices=['Sigmoid', 'ReLU', 'GELU'],
        help="Non linearity used by the network."
    )
    parser.add_argument(
        "--maxpool",
        action="store_true",
        help="Use the legacy architecture with maxpool2D instead of avgpool2d."
    )

    args = parser.parse_args(sys.argv[1:])

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    # model = medium_cnn(num_classes=27)  # 26 letters and 1 N/A
    non_linearity = args.nl
    if non_linearity == 'Sigmoid':
        non_linearity_function = nn.Sigmoid()
    elif non_linearity == 'ReLU':
        non_linearity_function = nn.ReLU()
    elif non_linearity == 'GELU':
        non_linearity_function = nn.GELU()
    else:
        raise NotImplementedError
    model = medium_cnn(num_classes=10, non_linearity=non_linearity_function, maxpool=args.maxpool)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    train_loader = mnist_loader(args.batch_size, train=True)
    test_loader = mnist_loader(args.batch_size, train=False)

    global_steps = []
    global_ranks = []
    global_traces = []
    for epoch in range(args.epochs):
        epoch_steps, epoch_ranks, epoch_traces = train_epoch(
            model, train_loader, optimizer, epoch + 1
        )
        global_steps.append(epoch_steps + epoch * len(train_loader))
        global_ranks.append(epoch_ranks)
        global_traces.append(epoch_traces)
        test(model, test_loader)
        torch.save(model.state_dict(), output_dir / f"mnist_medium_cnn_{epoch + 1:02d}_{'maxpool' if args.maxpool else 'avgpool'}_{non_linearity}.pt")

    global_steps = torch.cat(global_steps, dim=0)
    global_ranks = torch.cat(global_ranks, dim=1)
    global_traces = torch.cat(global_traces, dim=1)
    # save_mean_trace(
    #     global_steps,
    #     global_traces,
    #     output_dir / "traces_medium_cnn.pdf",
    # )
    # save_ranks(
    #     global_steps,
    #     global_ranks,
    #     output_dir / "ranks_medium_cnn.pdf",
    # )
