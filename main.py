from curvature import model_curvature_computer
from datetime import datetime
import plot
import mnist_networks
import xor_networks
import xor_datasets
import torch
import matplotlib.pyplot as plt

import torchvision.datasets as datasets
import torchvision.transforms as transforms

if __name__ == "__main__":
    DATASET = ['MNIST', 'XOR', 'EMNIST'][0]
    RANDOM = False
    SEED = 42
    number_of_points = 100
    TASK = ["rank", "curvature"][0]
    restrict_to_class = 9

    if DATASET == 'MNIST':
        checkpoint_path = './checkpoint/medium_cnn_10_ReLU.pt'
        network = mnist_networks.medium_cnn(checkpoint_path)
        network_score = mnist_networks.medium_cnn(checkpoint_path, score=True)

        normalize = transforms.Normalize((0.1307,), (0.3081,))

        input_space = datasets.MNIST(
            "data",
            train=False,  # TODO: True ?
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
    elif DATASET == 'EMNIST':
        checkpoint_path = './checkpoint/medium_cnn_10_ReLU.pt'
        network = mnist_networks.medium_cnn(checkpoint_path)
        network_score = mnist_networks.medium_cnn(checkpoint_path, score=True)

        normalize = transforms.Normalize((0.1307,), (0.3081,))

        input_space = datasets.EMNIST(
            "data",
            split="letters",
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
    elif DATASET == 'XOR':
        checkpoint_path = './checkpoint/xor_net_relu_30.pt'
        network = xor_networks.xor_net(checkpoint_path)
        network_score = xor_networks.xor_net(checkpoint_path, score=True)

        input_space = xor_datasets.XorDataset(nsample=100000, test=True, discrete=False)
    
    if restrict_to_class is not None:
        restriction_indices = input_space.targets == restrict_to_class
        input_space.targets = input_space.targets[restriction_indices]
        input_space.data = input_space.data[restriction_indices]


    device = next(network.parameters()).device
            
    curvature = model_curvature_computer(network, network_score, input_space, verbose=False)
    
    images = torch.cat([curvature.get_point()[0].unsqueeze(0) for _ in range(number_of_points)])
    print(f"Shape of points: {images.shape}")
    random_points = torch.rand_like(images)
    if RANDOM:
        points = random_points
    else:
        points = images

    probas = curvature.jac_proba(points)
    fim_on_data = curvature.fim_on_data(points)
    local_data_matrix = curvature.local_data_matrix(points)
    
    date = datetime.now().strftime("%y%m%d-%H%M%S")
    output_dir = f"output/{DATASET}/{date}_nsample={number_of_points}{f'_class={restrict_to_class}' if restrict_to_class is not None else ''}/"
    
    if TASK == "rank":
        plot.save_rank(fim_on_data, 
                    r"Rank of $G(e_i,e_j)$",
                    output_name=f"rank_G_on_data_{'random' if RANDOM else 'images'}", output_dir=output_dir)
        plot.save_rank(local_data_matrix, 
                    r"Rank of $G$",
                    output_name=f"rank_G_{'random' if RANDOM else 'images'}", output_dir=output_dir)
        plot.save_rank(probas, 
                    r"Rank of $(∇p_i)_i$",
                    output_name=f"rank_jac_{'random' if RANDOM else 'images'}", output_dir=output_dir)

        plot.save_eigenvalues(local_data_matrix, 
                            r"Eigenvalues of $G$",
                            output_name=f"eigenvalues_G_{'random' if RANDOM else 'images'}", output_dir=output_dir)
        plot.save_eigenvalues(fim_on_data,
                            r"Eigenvalues of $G(e_i,e_j)$",
                            output_name=f"eigenvalues_G_on_data_{'random' if RANDOM else 'images'}", output_dir=output_dir)

    if TASK == "curvature":
        # omega = curvature.connection_form(points)
        # print(omega)
        Omega = curvature.curvature_form(points)
        print(Omega.max(), Omega.min())
        print(f"Mean curvature tensors: {Omega.mean()}")
        print(f"Proportion of nan: {Omega.isnan().sum() / Omega.numel():.6f}")
        print(Omega)
    # print(probas.norm(dim=2).min(dim=1).values.mean())
    # eigenvalues = torch.linalg.eigvalsh(local_data_matrix).mean(0).detach()
    # print(eigenvalues)
    # plt.plot(range(len(eigenvalues.numpy())), eigenvalues.numpy())
    # plt.yscale('log')
    # plt.show()
    # plot.plot_debug(curvature, random_points)
    # for img, metric, proba in zip(point, fim_on_data, probas):
    #     plt.matshow(img.squeeze(0))
    #     plt.matshow(proba, aspect="auto")
    #     plt.matshow(metric.detach().numpy())
    #     plt.show()