from datetime import datetime
from math import ceil, floor, sqrt
from os import makedirs, path
from tkinter import W
from typing import Optional, Union, Tuple
from pathlib import Path
from matplotlib import cm
import torch
import matplotlib.pyplot as plt

from matplotlib.colors import SymLogNorm, Normalize
from tqdm import tqdm
from geometry import GeometricModel


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

#  def plot_debug(
    #  curvature: model_curvature_computer,
    #  eval_point: torch.Tensor,
    #  ) -> None:

    #  j = curvature.jac_proba(eval_point)
    #  fim_on_data = curvature.fim_on_data(eval_point).detach()
    #  euclidean_product_on_data = torch.einsum('zai, zki, zkj, zbj-> zab',
                                             #  j, j, j, j)
    #  bs, C, x = j.shape
    #  bs, _, px_row, px_col = eval_point.shape
    #  j = j.reshape(bs, C, px_row, px_col)
    #  date = datetime.now().strftime("%y%m%d-%H%M%S")
    #  output_dir = f"output/{date}/"
    #  save_matrices(
        #  matrices=eval_point,
        #  titles=["Data point"],
        #  output_dir=output_dir,
        #  output_name="debug_data_point",
    #  )
    #  save_matrices(
        #  matrices=torch.cat((fim_on_data.unsqueeze(1), euclidean_product_on_data.unsqueeze(1)), dim=1),
        #  titles=[r"$G(e_i,e_j)$", r"$e_i ⋅ e_j$"],
        #  output_dir=output_dir,
        #  output_name="debug_metric_grad_proba",
        #  log_scales=True,
    #  )
    #  save_matrices(
        #  matrices=j,
        #  titles=[f"Class n°{i}" for i in range(C)],
        #  output_dir=output_dir,
        #  output_name="debug_jacobians",
        #  no_ticks=True,
        #  )

# TODO: plot matrices of different shapes on same plot
def save_matrices(
    matrices: torch.Tensor,
    titles: Optional[Union[list[list[str]], list[str]]]=None,
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
    rtol: float=1e-5,
    atol: float=None,
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
    
    C = matrices.shape[-2]
        
    if isinstance(titles, str):
        titles = [titles for _ in matrices]
    
    ranks = torch.linalg.matrix_rank(matrices, atol=atol, hermitian=hermitian, rtol=rtol)
    
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


def save_vectors_norm(
    vectors: torch.Tensor,
    titles: Union[list[str], str]="",
    output_dir: Union[str, Path]='output/',
    output_name: str="vectors_norm",
) -> None:
    """ Save plot of the norm of the eigenvalues averaged over the matrices.

    Args:
        vectors (torch.Tensor): vectors to compute the norm of with shape (bs, c, i)
        titles (Union[list[str], str]): titles of the plot for each batch.
        output_dir (Union[str, Path], optional): Directory where to save the results. Defaults to 'output/'.
        output_name (str, optional): Name of the file. Defaults to "vectors_norm".
    """

    if not path.isdir(output_dir):
        makedirs(output_dir)
    
    if len(vectors.shape) == 3:
        vectors = vectors.unsqueeze(0)
    
    number_of_batch = vectors.shape[0]
    
    C = vectors.shape[-2]
        
    if isinstance(titles, str):
        titles = [titles for _ in vectors]
    
    vectors_norm = torch.norm(vectors, p=2, dim=-1).sort(descending=True).values
    max_norms = vectors_norm.max(dim=-1, keepdims=True).values
    vectors_norm = vectors_norm / max_norms
    
    _, axes = plt.subplots()
    
    for index_batch, (vectors_norm_batch, title_batch) in enumerate(zip(vectors_norm, titles)):
        for vn in vectors_norm_batch:
            axes.plot(vn.detach().numpy(), "b.-", alpha=.7, lw=0.5)  # TODO: color chosen by the class
            axes.set_xticks(range(len(vn)))
        axes.set_yscale('log')
        axes.set_title(title_batch)
        saving_path = f"{output_dir}"
        if number_of_batch > 1:
            saving_path = f"{saving_path}batch_{index_batch}_"
        saving_path = f"{saving_path}{output_name}.pdf"
        plt.savefig(saving_path, transparent=True, dpi=None)


def save_eigenvalues(
    matrices_list: Union[list[torch.Tensor], torch.Tensor],
    titles: Union[list[str], str]="",
    symetric: bool=False,
    output_dir: Union[str, Path]='output/',
    output_name: str="eigenvalues",
    singular_values: bool=False,
    known_rank: Optional[int]=None,
) -> None:
    """ Save plot of the norm of the eigenvalues averaged over the matrices.

    Args:
        matrices (torch.Tensor): matrices to compute the eigenvalues of with shape (bs, n, i, j)
        titles (Union[list[str], str]): titles of the plot for each batch.
        output_dir (Union[str, Path], optional): Directory where to save the results. Defaults to 'output/'.
        output_name (str, optional): Name of the file. Defaults to "eigenvalues".
    """

    if not path.isdir(output_dir):
        makedirs(output_dir)

    if not isinstance(matrices_list, list):
        matrices_list = [matrices_list]
    
    for matrices in matrices_list:
        if len(matrices.shape) == 3:
            matrices = matrices.unsqueeze(0)
        
        number_of_batch = matrices.shape[0]
        
        if known_rank is None:
            known_rank = min(matrices.shape[1:])
            
        if isinstance(titles, str):
            titles = [titles for _ in matrices]
        

        if singular_values:
            eigenvalues = torch.linalg.svdvals(matrices)
        elif symetric:
            eigenvalues = torch.linalg.eigvalsh(matrices) 
        else:
            eigenvalues = torch.linalg.eigvals(matrices)
        
        eigenvalues = eigenvalues.abs().sort(descending=True).values[...,:19]
        # max_eigenvalues = eigenvalues.max(dim=-1, keepdims=True).values
        # eigenvalues = eigenvalues / max_eigenvalues
        
        _, axes = plt.subplots()
        
        for index_batch, (eigenvals_batch, title_batch) in enumerate(zip(eigenvalues, titles)):
            for ev in eigenvals_batch:
                axes.plot(range(known_rank), ev.detach().numpy()[:known_rank], "b.-", alpha=.7, lw=0.5)  # TODO: color chosen by the class
                axes.plot(range(known_rank - 1, min(known_rank + 1, len(ev))), ev.detach().numpy()[known_rank - 1:known_rank + 1], "m:", alpha=.7, lw=0.5)
                axes.plot(range(known_rank, len(ev)), ev.detach().numpy()[known_rank:], "r.-", alpha=.7, lw=0.5)
                axes.set_xticks(range(len(ev)))
            axes.set_yscale('log')
            axes.set_title(title_batch)
            saving_path = f"{output_dir}"
            if number_of_batch > 1:
                saving_path = f"{saving_path}batch_{index_batch}_"
            saving_path = f"{saving_path}{output_name}.pdf"
            plt.savefig(saving_path, transparent=True, dpi=None)


def kmeans(
    points: torch.Tensor,
    k: int=2,
    max_iterations: int=100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Naive implementation of the k-means algorithm.

    Args:
        points (torch.Tensor): points to clusterize with shape (bs, d)
        k (int, optional): number of means. Defaults to 2.
        max_iterations (int, optional): max number of iterations. Defaults to 10.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: tuple of index of the nearest mean for each point: shape (bs), and centroids tensor: shape (k, d).
    """

    points = points.flatten(1)
    centroids = points[:k].clone()

    for i in range(max_iterations):
        # E step: assign points to the closest cluster
        distance_to_centroids = torch.cdist(points, centroids, p=1)
        nearest_cluster = distance_to_centroids.argmin(dim=1, keepdim=True).long()
        
        # M step: update the centroids to the normalized cluster average
        centroids = torch.zeros_like(centroids)
        centroids.scatter_add_(0, nearest_cluster, points)
        num_points_by_cluster = nearest_cluster.flatten().bincount(minlength=k).type_as(centroids).view(k, 1)
        centroids /= num_points_by_cluster
        
    return nearest_cluster, centroids


def save_spectral_clustering(
    input_points: torch.Tensor,
    matrices: torch.Tensor,
    titles: Union[list[str], str]="",
    symetric: bool=False,
    output_dir: Union[str, Path]='output/',
    output_name: str="image",
    singular_values: bool=False,
    known_rank: int=None,
    n_ev: int=None,
    n_cluster: int=2,
) -> None:
    """ Save plot of the norm of the eigenvalues averaged over the matrices.

    Args:
        input_points (torch.Tensor): Points corresponding to the matrices with shape (bs, d)
        matrices (torch.Tensor): matrices to compute the eigenvalues of with shape (bs, i, j)
        titles (Union[list[str], str]): titles of the plot for each batch.
        output_dir (Union[str, Path], optional): Directory where to save the results. Defaults to 'output/'.
        output_name (str, optional): Name of the file. Defaults to "eigenvalues".
        n_ev (int, optional): Number of eigenvalues to consider for the clustering.
        n_cluster (int, optional): Number of cluster for the clustering.
    """
    number_of_points = matrices.shape[0]
    colors = plt.cm.rainbow(torch.linspace(0, 1, n_cluster))
    
    if known_rank is None:
        known_rank = min(matrices.shape[2:])
        
    if isinstance(titles, str):
        titles = [titles for _ in matrices]
    

    if singular_values:
        eigenvalues = torch.linalg.svdvals(matrices)
    elif symetric:
        eigenvalues = torch.linalg.eigvalsh(matrices) 
    else:
        eigenvalues = torch.linalg.eigvals(matrices)
    
    eigenvalues = eigenvalues.abs().sort(descending=True).values[..., :known_rank+1]
    #  max_eigenvalues = eigenvalues.max(dim=-1, keepdims=True).values
    #  eigenvalues = eigenvalues / max_eigenvalues
    selected_eigenvalues = eigenvalues[..., known_rank - n_ev:known_rank]
    
    # clusters_indexes, centroids = kmeans(selected_eigenvalues.log10(), k=n_cluster)

    _, axes = plt.subplots()
    # for (i, ev), cluster in zip(enumerate(eigenvalues), clusters_indexes):
    for (i, ev) in enumerate(eigenvalues):
            axes.plot(range(known_rank), ev.detach().numpy()[:known_rank], ".-", c=colors[int(i > len(eigenvalues)//2)], alpha=.7, lw=0.5)  # TODO: color chosen by the class
            axes.plot(range(known_rank - 1, min(known_rank + 1, len(ev))), ev.detach().numpy()[known_rank - 1:known_rank + 1], "m:", alpha=.7, lw=0.5)
            axes.plot(range(known_rank, len(ev)), ev.detach().numpy()[known_rank:], "r.-", alpha=.7, lw=0.5)
            axes.set_xticks(range(len(ev)))
    axes.set_yscale('log')
    saving_path = f"{output_dir}{output_name}.pdf"
    plt.savefig(saving_path, transparent=True, dpi=None)
    
    # for cluster, (index_image, image) in zip(clusters_indexes, enumerate(input_points)):
    #     saving_path = f"{output_dir}{'/' if output_dir[-1] != '/' else ''}cluster-{cluster.data[0]}/"
    #     if not path.isdir(saving_path):
    #         makedirs(saving_path)
    #     saving_path = f"{saving_path}{output_name}_{index_image}.pdf"
    #     #  save_matrices(input_points, output_dir=, output_name=, no_ticks=True)
    #     # plt.image(image)
    #     plt.matshow(image.squeeze(0))
    #     plt.tick_params(left = False, right = False, top=False, labeltop=False, labelleft = False , labelbottom = False, bottom = False)
    #     plt.savefig(saving_path, transparent=True, dpi=None)


def save_function_neighborhood(
    geo_model: GeometricModel,
    input_points: torch.Tensor,
    function: str="rank",
    steps: int=20,
    plot_range: float=10,
    rank_rtol: float=1e-9,
    rank_atol: float=None,
) -> None:
    function_options = ["rank", "proba", "trace", "gradproba"]    
    if function not in function_options:
        raise ValueError(f"{function} not implemented yet, please select one in {function_options}.")
    G = geo_model.local_data_matrix(input_points)
    # compute the basis vectors for the plane
    if G.is_cuda:
        _, _, ev = torch.linalg.svd(G)
        e_1 = ev[..., 0, :]  # be careful, it isn't intuitive -> RTD
        e_2 = ev[..., 1, :]  # be careful, it isn't intuitive -> RTD
    else:
        _, ev = torch.linalg.eigh(G)  # value, vector, in ascending order
        e_1 = ev[..., -1]  # be careful, it isn't intuitive -> RTD
        e_2 = ev[..., -2]  # be careful, it isn't intuitive -> RTD
    norm_1 = torch.linalg.vector_norm(e_1, ord=2, dim=-1, keepdim=True)
    e_1 = e_1 / norm_1
    e_1 = e_1.reshape(input_points.shape)
    norm_2 = torch.linalg.vector_norm(e_2, ord=2, dim=-1, keepdim=True)
    e_2 = e_2 / norm_2
    e_2 = e_2.reshape(input_points.shape)
    grid1D = torch.linspace(-plot_range, plot_range, steps)
    xs = torch.einsum("zi..., i -> zi...", e_1.unsqueeze(1), grid1D)
    ys = torch.einsum("zi..., i -> zi...", e_2.unsqueeze(1), grid1D)
    grid_vectors = (xs.unsqueeze(2) + ys.unsqueeze(1)).reshape(xs.shape[0], -1, *xs.shape[2:])
    grid_points = input_points.unsqueeze(1) + grid_vectors
    number_of_classes = geo_model.proba(input_points[0].unsqueeze(0)).shape[-1]

    for points in tqdm(grid_points):
        if function == 'rank':
            G_on_grid = geo_model.local_data_matrix(points)
            rank_on_grid = torch.linalg.matrix_rank(G_on_grid, atol=rank_atol, rtol=rank_rtol, hermitian=True)
            rank_on_grid = rank_on_grid.reshape((*grid1D.shape, *grid1D.shape))
            cmap = cm.get_cmap('viridis', number_of_classes + 1)  
            plt.pcolormesh(grid1D, grid1D, rank_on_grid.detach().numpy(), cmap=cmap, vmin=-0.5, vmax=number_of_classes + 0.5) 
            print(rank_on_grid)
            plt.colorbar(ticks=range(number_of_classes + 2))
            plt.show()
        elif function == 'proba':
            grid2Dx, grid2Dy = torch.meshgrid(grid1D, grid1D)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.set(xlabel=r'e_1', ylabel=r'e_2', zlabel=r'proba')
            proba_on_grid, class_on_grid = torch.max(geo_model.proba(points), dim=1)
            proba_on_grid = proba_on_grid.reshape((*grid1D.shape, *grid1D.shape))
            class_on_grid = class_on_grid.reshape((*grid1D.shape, *grid1D.shape))
            cmap = cm.get_cmap('jet', number_of_classes)
            scamap = cm.ScalarMappable(norm=Normalize(vmin=-0.5, vmax=number_of_classes-0.5), cmap=cmap)
            color_from_class = scamap.to_rgba(class_on_grid)
            ax.plot_surface(grid2Dx, grid2Dy, proba_on_grid.detach(), facecolors=color_from_class, cmap=cmap, rstride=1, cstride=1, lw=0, alpha=0.6)
            fig.colorbar(scamap, ticks=range(number_of_classes))
            plt.show()
        elif function == 'trace':
            grid2Dx, grid2Dy = torch.meshgrid(grid1D, grid1D)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.set(xlabel=r'e_1', ylabel=r'e_2', zlabel=r'$tr(G)$')
            G_on_grid = geo_model.local_data_matrix(points)
            trace_G_on_grid = torch.einsum('...ii -> ...', G_on_grid)
            trace_G_on_grid = trace_G_on_grid.reshape((*grid1D.shape, *grid1D.shape))
            _, class_on_grid = torch.max(geo_model.proba(points), dim=1)
            class_on_grid = class_on_grid.reshape((*grid1D.shape, *grid1D.shape))
            cmap = cm.get_cmap('jet', number_of_classes)
            scamap = cm.ScalarMappable(norm=Normalize(vmin=-0.5, vmax=number_of_classes-0.5), cmap=cmap)
            color_from_class = scamap.to_rgba(class_on_grid)
            ax.plot_surface(grid2Dx, grid2Dy, trace_G_on_grid.detach(), facecolors=color_from_class, cmap=cmap, rstride=1, cstride=1, lw=0, alpha=0.6)
            fig.colorbar(scamap, ticks=range(number_of_classes))
            plt.show()
        elif function == 'gradproba':
            grid2Dx, grid2Dy = torch.meshgrid(grid1D, grid1D)
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.set(xlabel=r'e_1', ylabel=r'e_2', zlabel=r'$min_i|∇p_i|$')
            jac_proba_on_grid = geo_model.jac_proba(points)
            min_norm_grad_proba_on_grid = jac_proba_on_grid.norm(p=2, dim=-1).min(dim=-1).values.log10()
            _, class_on_grid = torch.max(geo_model.proba(points), dim=1)
            min_norm_grad_proba_on_grid = min_norm_grad_proba_on_grid.reshape((*grid1D.shape, *grid1D.shape))
            class_on_grid = class_on_grid.reshape((*grid1D.shape, *grid1D.shape))
            cmap = cm.get_cmap('jet', number_of_classes)
            scamap = cm.ScalarMappable(norm=Normalize(vmin=-0.5, vmax=number_of_classes-0.5), cmap=cmap)
            color_from_class = scamap.to_rgba(class_on_grid)
            ax.plot_surface(grid2Dx, grid2Dy, min_norm_grad_proba_on_grid.detach(), facecolors=color_from_class, cmap=cmap, rstride=1, cstride=1, lw=0, alpha=0.6)
            fig.colorbar(scamap, ticks=range(number_of_classes))
            plt.show()
        
        
def save_drop_eigenvalue(
    matrices: torch.Tensor,
    symetric: bool=False,
    titles: Union[list[str], str]="",
    output_dir: Union[str, Path]='output/',
    output_name: str="drop_eigenvalue",
    singular_values: bool=False,
    max_rank: int=None,
    method: str="derivative",
) -> None:
    """ Save plot of the norm of the eigenvalues averaged over the matrices.

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
    
    C = matrices.shape[-2]
        
    if isinstance(titles, str):
        titles = [titles for _ in matrices]
    

    if singular_values:
        eigenvalues = torch.linalg.svdvals(matrices)
    elif symetric:
        eigenvalues = torch.linalg.eigvalsh(matrices) 
    else:
        eigenvalues = torch.linalg.eigvals(matrices)
    
    eigenvalues = eigenvalues.abs().sort(descending=True).values[...,:19]
    # max_eigenvalues = eigenvalues.max(dim=-1, keepdims=True).values
    # eigenvalues = eigenvalues / max_eigenvalues
    
    _, axes = plt.subplots()

    for index_batch, (eigenvals_batch, title_batch) in enumerate(zip(eigenvalues, titles)):
        raise NotImplementedError
