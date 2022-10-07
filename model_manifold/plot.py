from email.mime import image
import enum
from math import ceil, floor, sqrt
from pathlib import Path
from typing import Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import seaborn as sns


def denormalize(x: torch.Tensor, normalization: transforms.Normalize) -> torch.Tensor:
    mean = torch.tensor(normalization.mean, device=x.device).view(-1, 1, 1)
    std = torch.tensor(normalization.std, device=x.device).view(-1, 1, 1)
    x *= std
    x += mean
    return x


def to_gif(
    images: torch.Tensor,
    output_path: Union[str, Path],
    step: int = 1,
    scale_factor: float = 1.0,
) -> None:
    # noinspection PyArgumentList
    images = images[::step]
    images = images.cpu()
    images = F.interpolate(images, scale_factor=scale_factor)
    images = images.permute(0, 2, 3, 1).squeeze_(-1)
    images = torch.round(images * 255).to(torch.uint8)
    images = torch.unbind(images)
    images = [im.numpy() for im in images]
    imageio.mimsave(str(output_path), images)


def save_strip(
    images: torch.Tensor,
    output_path: Union[str, Path],
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    steps: int = 9,
) -> None:
    images = images.permute(0, 2, 3, 1).squeeze_(-1)
    image_indices = torch.linspace(0, images.shape[0] - 1, steps).tolist()
    fig, axes = plt.subplots(1, steps, figsize=(10, 1.8))
    for plot_idx, image_idx in enumerate(image_indices):
        iteration = round(image_idx)
        image = images[iteration].cpu()
        axes[plot_idx].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[plot_idx].set_title(
            f"Iteration {iteration}:\n"
            f"predicted label {predictions[iteration]} with\n"
            f"probability {probabilities[iteration]:0.4f}",
            fontsize=7,
        )
        axes[plot_idx].axis("off")
    fig.tight_layout()
    plt.savefig(str(output_path))

def save_images(
    images: torch.Tensor,
    output_path: Union[str, Path],
    probabilities: torch.Tensor = None,
    predictions: torch.Tensor = None,
) -> None:
    # images = images.permute(0, 2, 3, 1).squeeze_(-1)
    nb_images = images.shape[0]
    image_indices = torch.linspace(0, nb_images - 1, nb_images).tolist()
    n_row, n_col = floor(sqrt(nb_images)), ceil(sqrt(nb_images))
    fig, axes = plt.subplots(n_row, n_col)
    for plot_idx, image_idx in enumerate(image_indices):
        iteration = round(image_idx)
        row = plot_idx // n_col
        col = plot_idx % n_col
        image = images[iteration].cpu()
        axes[row, col].imshow(image, cmap="gray", vmin=0, vmax=1)
        axes[row, col].set_title(
            f"Image n°{iteration}:\n" +
            ("" if predictions is None else f"predicted label {predictions[iteration]} with\n") +
            ("" if probabilities is None else f"probability {probabilities[iteration]:0.4f}"),
            fontsize=7,
        )
    [ax.set_axis_off() for ax in axes.ravel()]
    fig.tight_layout()
    plt.savefig(str(output_path))
    plt.show()
    
    
def save_matrices(
    matrices: torch.Tensor,
    output_path: Union[str, Path],
) -> None:
    # images = images.permute(0, 2, 3, 1).squeeze_(-1)
    nb_images = matrices.shape[0]
    image_indices = torch.linspace(0, nb_images - 1, nb_images).tolist()
    n_row, n_col = floor(sqrt(nb_images)), ceil(sqrt(nb_images))
    fig, axes = plt.subplots(n_row, n_col)
    for plot_idx, image_idx in enumerate(image_indices):
        iteration = round(image_idx)
        row = plot_idx // n_col
        col = plot_idx % n_col
        image = matrices[iteration].cpu()
        im = axes[row, col].matshow(image)
        axes[row, col].set_xticks(range(10))
        axes[row, col].set_yticks(range(10))
        axes[row, col].set_xticklabels([])
        axes[row, col].set_yticklabels([])
        fig.colorbar(im, ax=axes[row, col])
        # axes[row, col].set_title(
        #     f"Matrix n°{iteration}:\n",
        #     fontsize=7,
        # )
    # [ax.set_axis_off() for ax in axes.ravel()]
    fig.tight_layout()
    # fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(str(output_path) + ".pdf")
    plt.show()



def save_ranks(
    steps: torch.Tensor, ranks: torch.Tensor, output_path: Union[str, Path]
) -> None:
    steps = steps.cpu().numpy()
    ranks = ranks.cpu().numpy()
    steps = np.tile(steps, (ranks.shape[0], 1))
    points = np.stack([steps, ranks], axis=-1).reshape(-1, 2)
    unique_points, counts = np.unique(points, axis=0, return_counts=True)
    data = {
        "Steps": unique_points[:, 0],
        r"Rank of $G(x, w)$": unique_points[:, 1],
        "Count": counts,
    }
    plt.clf()
    with plt.style.context("seaborn"):
        sns.scatterplot(
            data=data,
            x="Steps",
            y=r"Rank of $G(x, w)$",
            hue="Count",
            size="Count",
            sizes=(20, 200),
            legend="full",
            palette="coolwarm",
        )
        plt.yticks(range(np.max(ranks) + 2))
        plt.xlabel("Steps")
        plt.ylabel(r"Rank of $G(x, w)$")
        plt.title(r"Distribution of rank $G(x, w)$ during training")
    plt.tight_layout()
    plt.savefig(str(output_path))


def save_traces(
    steps: torch.Tensor, traces: torch.Tensor, output_path: Union[str, Path]
) -> None:
    steps = steps.cpu().numpy()
    traces = traces.cpu().numpy()
    colors = (
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    )
    plt.clf()
    with plt.style.context("seaborn"):
        for i in range(len(traces)):
            plt.plot(steps, traces[i], color=colors[i % len(colors)])
        plt.xlabel("Steps")
        plt.ylabel(r"Trace of $G(x, w)$")
        plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(str(output_path))


def save_mean_trace(
    steps: torch.Tensor, traces: torch.Tensor, output_path: Union[str, Path]
) -> None:
    steps = steps.cpu().numpy()
    traces = traces.cpu().numpy()
    mean_trace = np.mean(traces, axis=0)
    plt.clf()
    with plt.style.context("seaborn"):
        plt.plot(steps, mean_trace)
        plt.xlabel("Steps")
        plt.ylabel(r"Mean trace of $G(x, w)$")
        plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(str(output_path))
