import argparse
from http.client import NOT_IMPLEMENTED
from random import random
from typing import Tuple, Union
import torch
import torch.nn as nn

from model_manifold.data_matrix import jacobian

class model_curvature_computer:
    """A class to compute curvature and associated measures.
    """
    
    def __init__(
        self,
        network: nn.model,
        verbose: bool=True,
        ) -> None:
        """Initialize the curvature computer for a given trained network.

        Args:
            network (nn.model): trained network
            verbose (bool, optional): print more information meant for debugging.
                                      Defaults to True.
        """
        
        self.network = network
        self.verbose = verbose
        self.input_space = NOT_IMPLEMENTED
        
    def get_point(
        self,
        index: int=-1,
    ) -> Tuple[torch.Tensor, Union[str, int]]:
        """Get an image from the database given a index.

        Args:
            image_idx (int): index of the image to get
            If -1 is passed, the image is random.

        Returns:
            torch.Tensor: image chosen.
            str | int: label
        """
        
        if index == -1:
            index = random.randrange(len(self.input_space))
        point = self.input_space[index][0].to(self.device)
        label = self.input_space[index][1]
        print()
        
        return point, label

    def proba(
        self,
        eval_point: torch.Tensor,
    ) -> None:

        if len(eval_point.shape) == 3:  # TODO: trouver un truc plus propre
            eval_point = eval_point.unsqueeze(0)
        p = torch.exp(self.network(eval_point))
        return p

    def score(
        self,
        eval_point: torch.Tensor,
    ) -> None:
        
        if len(eval_point.shape) == 3:  # TODO: trouver un truc plus propre
            eval_point = eval_point.unsqueeze(0)
        
        return self.network_score(eval_point)


    def grad_proba(
        self,
        eval_point: torch.Tensor,
        wanted_class: int, 
    ) -> torch.Tensor:

        j = jacobian(self.proba, eval_point).squeeze(0)

        grad_proba = j[wanted_class, :]

        return grad_proba


    def jac_proba(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:

        print(f"shape of eval_point = {eval_point.shape}")
        print(f"shape of output = {self.proba(eval_point).shape}")
        j = jacobian(self.proba, eval_point) # TODO: vérifier dans le cadre non batched
        print(f"shape of j before reshape = {j.shape}")
        j = j.sum(2)
        j = j.reshape(*(j.shape[:-3]), -1)
        print(f"shape of j after reshape = {j.shape}")
        

        return j
    
    
    def jac_score(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        
        print(f"shape of eval_point = {eval_point.shape}")
        print(f"shape of output = {self.proba(eval_point).shape}")
        j = jacobian(self.score, eval_point)
        print(f"shape of j before reshape = {j.shape}")
        
        j = j.sum(2)
        j = j.reshape(*(j.shape[:-3]), -1)
        print(f"shape of j after reshape = {j.shape}")
        
        return j


    def dot_product_matrix(
        self,
        eval_point: torch.Tensor,
        score: bool=False,
    ) -> torch.Tensor:
        
        if score:
            j = self.jac_score(eval_point)
        else:
            j = self.jac_proba(eval_point)

        U = j @ j.transpose(-1, -2)
        
        # U = torch.nn.functional.normalize(U)

        return U