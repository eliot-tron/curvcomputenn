from datetime import datetime
from math import ceil, floor, sqrt
from os import makedirs, path
from tkinter import W
from typing import Union
from pathlib import Path
import torch
import matplotlib.pyplot as plt
from curvature import model_curvature_computer

from matplotlib.colors import SymLogNorm


def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

def plot_debug(
    curvature: model_curvature_computer,
    eval_point: torch.Tensor,
    ) -> None:

    j = curvature.jac_proba(eval_point)
    fim_on_data = curvature.fim_on_data(eval_point).detach()
    euclidean_product_on_data = torch.einsum('zai, zki, zkj, zbj-> zab',
                                             j, j, j, j)
    bs, C, x = j.shape
    bs, _, px_row, px_col = eval_point.shape
    j = j.reshape(bs, C, px_row, px_col)
    date = datetime.now().strftime("%y%m%d-%H%M%S")
    output_dir = f"output/{date}/"
    save_matrices(
        matrices=eval_point,
        titles=["Data point"],
        output_dir=output_dir,
        output_name="debug_data_point",
    )
    save_matrices(
        matrices=torch.cat((fim_on_data.unsqueeze(1), euclidean_product_on_data.unsqueeze(1)), dim=1),
        titles=[r"$G(e_i,e_j)$", r"$e_i ⋅ e_j$"],
        output_dir=output_dir,
        output_name="debug_metric_grad_proba",
        log_scales=True,
    )
    save_matrices(
        matrices=j,
        titles=[f"Class n°{i}" for i in range(C)],
        output_dir=output_dir,
        output_name="debug_jacobians",
        no_ticks=True,
        )

# TODO: plot matrices of different shapes on same plot
def save_matrices(
    matrices: torch.Tensor,
    titles: Union[list[list[str]], list[str]],
    output_dir: Union[str, Path]='output/',
    output_name: str="matrices",
    log_scales: Union[bool, list[bool]]=False,
    no_ticks: Union[bool, list[bool]]=False,
) -> None:
    """Plot on a smart grid the different matrices given in input.

    Args:
        matrices (torch.Tensor): Matrices to plot in format (bs, n, x, y).
                                 Creates one pdf for each bach (bs) and plot the n matrices on a grid of subplots.
        titles (Union[list[list[str]], list[str]])
        output_dir (Union[str, Path]): Directory in which the output will be stored.
        output_name (str): name to add to the file.
    """
    if not path.isdir(output_dir):
        makedirs(output_dir)
    
    if len(matrices.shape) == 3:
        matrices = matrices.unsqueeze(0)
    
    number_of_batch = matrices.shape[0]
    
    if titles is None:
        titles = [["" for _ in mb] for mb in matrices]
    elif isinstance(titles[0], str):
        titles = [titles] * number_of_batch

    if isinstance(log_scales, bool):
        log_scales = [[log_scales for _ in mb] for mb in matrices]
    elif isinstance(log_scales[0], bool):
        log_scales = [log_scales] * number_of_batch
        
   
    if isinstance(no_ticks, bool):
        no_ticks = [[no_ticks for _ in mb] for mb in matrices]
    elif isinstance(no_ticks, bool):
        no_ticks = [no_ticks] * number_of_batch

    for index_batch, matrices_batch, titles_batch, no_ticks_batch, log_scales_batch in zip(range(number_of_batch), matrices, titles, no_ticks, log_scales):
        number_of_matrices = matrices_batch.shape[0]
        n_row, n_col = floor(sqrt(number_of_matrices)) , ceil(sqrt(number_of_matrices))
        figure, axes = plt.subplots(n_row, n_col, figsize=(n_col*3, n_row*3), squeeze=False)

        for index, matrix, title, no_tick, log_scale in zip(range(number_of_matrices), matrices_batch, titles_batch, no_ticks_batch, log_scales_batch):
            row = index // n_col
            col = index % n_col
            if log_scale:
                matrix_subplot = axes[row, col].matshow(matrix, norm=SymLogNorm(matrix.abs().min().numpy()))
            else:
                matrix_subplot = axes[row, col].matshow(matrix)
            if no_tick:
                axes[row, col].tick_params(left = False, right = False, top=False, labeltop=False, labelleft = False , labelbottom = False, bottom = False)
            axes[row, col].set_title(title)
            colorbar(matrix_subplot)
        
        for axes_to_remove in range(number_of_matrices, n_row*n_col):
            row = axes_to_remove // n_col
            col = axes_to_remove % n_col
            axes[row, col].axis("off")

        figure.tight_layout()
        saving_path = f"{output_dir}"
        if number_of_batch > 1:
            saving_path = f"{saving_path}batch_{index_batch}_"
        saving_path = f"{saving_path}{output_name}.pdf"

        plt.savefig(saving_path, transparent=True, dpi=None)


def save_rank(
    matrices: torch.Tensor,
    titles: Union[list[str], str]="",
    output_dir: Union[str, Path]='output/',
    output_name: str="rank",
    hermitian: bool=False,
) -> None:
    """ Save plot of the evolution of the rank of the matrices.

    Args:
        matrices (torch.Tensor): matrices to compute the rank of with shape (bs, n, i, j)
        titles (Union[list[str], str]): titles of the plot for each batch.
        output_dir (Union[str, Path], optional): Directory where to save the results. Defaults to 'output/'.
        output_name (str, optional): Name of the file. Defaults to "rank".
    """

    if not path.isdir(output_dir):
        makedirs(output_dir)
    
    if len(matrices.shape) == 3:
        matrices = matrices.unsqueeze(0)
    
    number_of_batch = matrices.shape[0]
    matrices = matrices.double()
    
    C = matrices.shape[-2]
        
    if isinstance(titles, str):
        titles = [titles for _ in matrices]
    
    ranks = torch.linalg.matrix_rank(matrices, atol=None, hermitian=hermitian)
    
    figure, axes = plt.subplots()
    
    for index_batch, (rank_batch, title_batch) in enumerate(zip(ranks, titles)):
        axes.hist(rank_batch.int().numpy(), bins=torch.arange(-0.5, C+0.5, 1), rwidth=0.6)
        axes.set_xticks(range(C+1))
        axes.set_xlim((0, rank_batch.max()+1))
        axes.set_title(title_batch)
        saving_path = f"{output_dir}"
        if number_of_batch > 1:
            saving_path = f"{saving_path}batch_{index_batch}_"
        saving_path = f"{saving_path}{output_name}.pdf"
        plt.savefig(saving_path, transparent=True, dpi=None)


def save_eigenvalues(
    matrices: torch.Tensor,
    titles: Union[list[str], str]="",
    symetric: bool=True,
    output_dir: Union[str, Path]='output/',
    output_name: str="eigenvalues",
) -> None:
    """ Save plot of the norm of the eigenvalues meaned over the matrices.

    Args:
        matrices (torch.Tensor): matrices to compute the eigenvalues of with shape (bs, n, i, j)
        titles (Union[list[str], str]): titles of the plot for each batch.
        output_dir (Union[str, Path], optional): Directory where to save the results. Defaults to 'output/'.
        output_name (str, optional): Name of the file. Defaults to "eigenvalues".
    """

    if not path.isdir(output_dir):
        makedirs(output_dir)
    
    if len(matrices.shape) == 3:
        matrices = matrices.unsqueeze(0)
    
    number_of_batch = matrices.shape[0]
    
    matrices = matrices.double()
    C = matrices.shape[-2]
        
    if isinstance(titles, str):
        titles = [titles for _ in matrices]
    
    if symetric:
        eigenvalues = torch.linalg.eigvalsh(matrices) 
    else:
        eigenvalues = torch.linalg.eigvals(matrices)
    
    eigenvalues = eigenvalues.mean(1).abs().sort(descending=True).values[...,:19,...]
    
    _, axes = plt.subplots()
    
    for index_batch, (eigenvals_batch, title_batch) in enumerate(zip(eigenvalues, titles)):
        axes.plot(range(9), eigenvals_batch.detach().numpy()[:9], "bo")
        axes.plot(range(9, len(eigenvals_batch)), eigenvals_batch.detach().numpy()[9:], "ro")
        axes.set_xticks(range(len(eigenvals_batch)))
        axes.set_yscale('log')
        axes.set_title(title_batch)
        saving_path = f"{output_dir}"
        if number_of_batch > 1:
            saving_path = f"{saving_path}batch_{index_batch}_"
        saving_path = f"{saving_path}{output_name}.pdf"
        plt.savefig(saving_path, transparent=True, dpi=None)