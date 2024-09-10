import argparse
from copy import deepcopy
from datetime import datetime
from os import makedirs, path
import random
import time
from matplotlib import pyplot as plt
import numpy as np

import torch
from tqdm import tqdm
from experiment import *



if __name__ == "__main__":
    plt.rcParams.update({'font.size': 14})
    parser = argparse.ArgumentParser(
        description="CIDRE: Comparison based on Information between Datasets with degenerate Riemannian metric's Eigenvalues",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs='+',
        default="MNIST",
        choices=['MNIST', 'Letters', 'FashionMNIST', 'KMNIST', 'QMNIST', 'CIFARMNIST', 'XOR', 'XOR3D', 'CIFAR10', 'Noise', 'Adversarial'],
        metavar='name',
        help="Dataset name to be used.",
    )
    parser.add_argument(
        "--restrict",
        type=int,
        metavar="class",
        default=None,
        help="Class to restrict the main dataset to if needed.",
    )
    parser.add_argument(
        "--nsample",
        type=int,
        metavar='N',
        default=2,
        help="Number of initial points to consider."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="curvature",
        choices=['curvature', 'foliation', 'rank2D', 'proba2D', 'trace2D', 'gradproba', 'connection-forms'],
        help="Task."
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Permutes randomly the inputs."
    )
    parser.add_argument(
        "--savedirectory",
        type=str,
        metavar='path',
        default='./output/',
        help="Path to the directory to save the outputs in."
    )
    parser.add_argument(
        "--double",
        action="store_true",
        help="Use double precision (1e-16) for the computations (recommended)."
    )
    parser.add_argument(
        "--maxpool",
        action="store_true",
        help="Use the legacy architecture with maxpool2D instead of avgpool2d."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force device to cpu."
    )
    parser.add_argument(
        "--nl",
        type=str,
        metavar='f',
        nargs='+',
        default="ReLU",
        choices=['Sigmoid', 'ReLU', 'GELU'],
        help="Non linearity used by the network."
    )

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cpu:
        device = torch.device('cpu')
    print(f"Device: {device}")

    dataset_names = args.datasets
    num_samples = args.nsample
    task = args.task
    non_linearities =  args.nl
    adversarial_budget = 0
    if not isinstance(dataset_names, list):
        dataset_names = [dataset_names] * len(non_linearities)
    elif len(dataset_names) == 1:
        dataset_names = dataset_names * len(non_linearities)
    if not isinstance(non_linearities, list):
        non_linearities = [non_linearities] * len(dataset_names)
    elif len(non_linearities) == 1:
        non_linearities = non_linearities * len(dataset_names)
    dtype = torch.double if args.double else torch.float
    restrict_to_class = args.restrict
    pool = "maxpool" if args.maxpool else "avgpool"
    date = datetime.now().strftime("%y%m%d-%H%M%S")
    savedirectory = args.savedirectory + \
        ("" if args.savedirectory[-1] == '/' else '/') + \
        f"{'-'.join(dataset_names)}/{task}/{dtype}/" + \
        f"{date}_nsample={num_samples}{f'_class={restrict_to_class}' if restrict_to_class is not None else ''}_{pool}_{'-'.join(non_linearities)}/"
    if not path.isdir(savedirectory):
        makedirs(savedirectory)

    if not args.random:
        seed = 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)  # Numpy module.
        random.seed(seed)  # Python random module.
        torch.manual_seed(seed)
        torch.backends.cudnn.benchmark = False  # type: ignore
        torch.backends.cudnn.deterministic = True # type: ignore

    experiment_list = []
    base_experiment = None
    for (dataset, non_linearity) in zip(dataset_names, non_linearities):
        print(dataset, non_linearity)
        # TODO: faire un main_network <15-04-24, eliot> #
        if dataset == 'Adversarial':
            adversarial_budget = 2
        
        if len(experiment_list) > 0:
            print('entering comparison')
            base_space = None # TODO: how to do cleaner?
            if dataset in ['Noise', 'Adversarial']:
                base_space = deepcopy(base_experiment.input_space)
            experiment = implemented_experiment_dict[dataset](
                non_linearity=non_linearity,
                adversarial_budget=adversarial_budget,
                dtype=dtype,
                device=device,
                num_samples=num_samples,
                pool=pool,
                random=args.random,
                restrict_to_class=restrict_to_class,
                input_space=base_space,
                checkpoint_path=base_experiment.checkpoint_path,
                network=deepcopy(base_experiment.network),
                network_score=deepcopy(base_experiment.network_score),
            )
        else:
            experiment = implemented_experiment_dict[dataset](
                non_linearity=non_linearity,
                adversarial_budget=adversarial_budget,
                dtype=dtype,
                device=device,
                num_samples=num_samples,
                restrict_to_class=restrict_to_class,
                pool=pool,
                random=args.random,
            )
            
        experiment_list.append(experiment)
        if base_experiment is None:
            base_experiment = experiment_list[0]

    base_output_dimension = base_experiment.get_output_dimension()
    
    nb_experiments = len(experiment_list)

    print(f'Task {task} with dataset {dataset_names}, non linearities {non_linearities} and {num_samples} samples.')

    if task == "curvature":
        for i, experiment in enumerate(tqdm(experiment_list)):
            experiment.plot_curvature(type='scalar')
            saving_path = savedirectory + 'curvature.pdf'
            plt.tight_layout()
            plt.savefig(saving_path, transparent=True, dpi=None)
    
    elif task == 'foliation':
        transverse = True
        for experiment in tqdm(experiment_list):
            experiment.plot_foliation(transverse=transverse)
            saving_path = savedirectory + f"{'transverse' if transverse else 'kernel'}_foliations.pdf"
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.savefig(saving_path, transparent=True, dpi=None)
    

    elif task == 'rank2D':
        for i, experiment in enumerate(tqdm(experiment_list)):
            experiment.save_function_neighborhood(
                function='rank',
                steps=10,
                plot_range=1.,
            )

    elif task == 'proba2D':
        for i, experiment in enumerate(tqdm(experiment_list)):
            experiment.save_function_neighborhood(
                function='proba',
                steps=50,
                plot_range=10,
            )

    elif task == 'trace2D':
        for i, experiment in enumerate(tqdm(experiment_list)):
            experiment.save_function_neighborhood(
                function='trace',
                steps=10,
                plot_range=10,
            )

    elif task == 'gradproba':
        # jac_probas = geo_model.jac_proba(input_points)
        # grad_norms = jac_probas.norm(p=2, dim=-1, keepdim=True)
        # jac_normalized = jac_probas / grad_norms
        for i, experiment in enumerate(tqdm(experiment_list)):

            experiment.save_gradproba(
                savedirectory=savedirectory
            )
            # experiment.save_function_neighborhood(
            #     function='gradproba',
            #     steps=10,
            #     plot_range=10,
            # )
        print('wait')

    elif task == 'connection-forms':
        for i, experiment in enumerate(tqdm(experiment_list)):
            experiment.plot_connection_forms(savedirectory=savedirectory)


    for experiment in experiment_list:
        experiment.save_info_to_txt(savedirectory)

    print("Done.")
