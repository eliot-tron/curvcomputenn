import argparse
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
        self.input_space = NotImplemented
        
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
    
    
    def local_data_matrix(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        
        J_s = self.jac_score(eval_point)
        p = self.proba(eval_point)
        P = torch.diag(p)
        pp = torch.einsum("i,j -> ij", p, p)
        
        return torch.einsum("ji, jk, kl -> il", J_s, (P - pp), J_s)

    
    def hessian_gradproba(
        self, 
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing H(p_a)𝛁p_b 

        Args:
            eval_point (torch.Tensor): point of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor (H(p_a)𝛁p_b)_l with dimensions (a,b,l).
        """

        J_p = self.jac_proba(eval_point)
        J_s = self.jac_score(eval_point)
        P = self.proba(eval_point)
        C = P.shape[-1]
        I = torch.eye(C)
        N = P.expand(C, -1)
        
        """Compute """
        first_term = torch.einsum("bi, ki, ak, al -> abl", J_p, J_s, (I-N), J_p) 
        
        """Compute """
        second_term = torch.einsum("a, bi, ki, kl -> abl", P, J_p, J_s, J_p )
        
        return first_term - second_term

    
    
    def lie_bracket(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing [𝛁p_a, 𝛁p_b] = H(p_b)𝛁p_a - H(p_a)𝛁p_b

        Args:
            eval_point (torch.Tensor): point of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor [𝛁p_a, 𝛁p_b]_l with dimensions (a,b,l)
        """

        H_grad = self.hessian_gradproba(eval_point)
        
        return H_grad.transpose(-2, -3) - H_grad

    
    def bra_grad_lie(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing 〈𝛁p_a, [𝛁p_b, 𝛁p_c]〉

        Args:
            eval_point (torch.Tensor): point of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor 〈𝛁p_a, [𝛁p_b, 𝛁p_c]〉 with dimensions (a, b, c)
        """

        G = self.local_data_matrix(eval_point)
        J_p = self.jac_proba(eval_point)
        lie = self.lie_bracket(eval_point)

        return torch.einsum("ai, ij, bcj -> abc", J_p, G, lie) 
        
    
    def grad_bra_grad(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing 𝛁p_a〈𝛁p_b, 𝛁p_c〉

        Args:
            eval_point (torch.Tensor): point of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor 𝛁p_a〈𝛁p_b, 𝛁p_c〉with dimensions (a, b, c)
        """
        
        U = self.dot_product_matrix(eval_point)
        p = self.proba(eval_point)
        J_p = self.jac_proba(eval_point)
        J_s = self.jac_score(eval_point)
        
        """Compute p_k ∇p_l"""
        p_gradp = torch.einsum("l, ik -> ikl", p, J_p)
        
        """Compute δ_kl 𝛁p_k"""
        delta_gradp = torch.eye(J_p.shape[-2]) * J_p.unsqueeze(-1).transpose(-2, -3)
    
        return torch.einsum("bk, ai, ikl, cl -> abc" J_s, J_p, delta_gradp - p_gradp - p_gradp.transpose(-1,-2), J_s)
    
    
    def bra_connection(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing ⟨∇_(e_a) e_b, e_c⟩ with e_a = ∇p_a

        Args:
            eval_point (torch.Tensor): point of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor ⟨∇_(e_a) e_b, e_c⟩ with dimensions (a, b, c)
        """
        
        elmt_1 = self.grad_bra_grad(eval_point)
        elmt_2 = self.bra_grad_lie(eval_point)
        
        return ( elmt_1 + elmt_1.permute(1, 2, 0) - elmt_1.permute(2, 0, 1) - elmt_2 + elmt_2.permute(1, 2, 0) + elmt_2.permute(2, 0, 1) ) / 2 

    
    
    def connection_form(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the connection form ω(e_k) on the basis e_k = ∇p_k.

        Args:
            eval_point (torch.Tensor): point of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor ω^i_j(e_k) with dimensions (i, j, k) 
        """
        
        C = self.bra_connection(eval_point)
        G = self.local_data_matrix(eval_point)
        G_inv = G.inverse()
        
        return torch.einsum("il, kjl -> ijk", G_inv, C)
    
    
    def d_connection_form(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the exterior derivative of the connection form: dω(e_a, e_b).

        Args:
            eval_point (torch.Tensor): point of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor dω(e_a, e_b)^i_j with dimensions (i, j, a, b).
        """
        
        return NotImplemented
        
        
    def wedge_connection_forms(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the exterior product of the connection forms: ω(e_a) ∧ ω(e_b).

        Args:
            eval_point (torch.Tensor): point of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor (ω(e_a) ∧ ω(e_b))^i_j with dimensions (i, j, a, b).
        """
        
        return NotImplemented