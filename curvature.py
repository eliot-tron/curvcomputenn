from datetime import datetime
from math import ceil, floor, sqrt
from pathlib import Path
import random
from threading import local
from typing import Tuple, Union
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from matplotlib.colors import SymLogNorm
import matplotlib.pyplot as plt

import mnist_networks
import xor_networks
import xor_datasets

from model_manifold.data_matrix import jacobian, local_data_matrix

class model_curvature_computer:
    """A class to compute curvature and associated measures.
    """
    
    def __init__(
        self,
        network: nn.Module,
        network_score: nn.Module,
        input_space: datasets,
        verbose: bool=True,
        ) -> None:
        """Initialize the curvature computer for a given trained network.

        Args:
            network (nn.model): trained network
            verbose (bool, optional): print more information meant for debugging.
                                      Defaults to True.
        """
        
        self.network = network
        self.network_score = network_score
        self.verbose = verbose
        self.input_space = input_space
        self.device = next(self.network.parameters()).device
        
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
        if self.verbose: print(f"point: {point}, label: {label}")
        
        return point, label

    def proba(
        self,
        eval_point: torch.Tensor,
    ) -> None:

        if len(eval_point.shape) == 3:  # TODO: trouver un truc plus propre
            eval_point = eval_point.unsqueeze(0)
        p = torch.exp(self.network(eval_point))
        if self.verbose: print(f"proba: {p}")
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

        if self.verbose:
            print(f"shape of eval_point = {eval_point.shape}")
            print(f"shape of output = {self.proba(eval_point).shape}")
        j = jacobian(self.proba, eval_point) # TODO: vérifier dans le cadre non batched
        if self.verbose: print(f"shape of j before reshape = {j.shape}")
        j = j.sum(2)
        j = j.reshape(*(j.shape[:2]), -1)
        if self.verbose: print(f"shape of j after reshape = {j.shape}")

        return j
    
    
    def jac_score(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        
        if self.verbose:
            print(f"shape of eval_point = {eval_point.shape}")
            print(f"shape of output = {self.proba(eval_point).shape}")
        j = jacobian(self.score, eval_point)
        if self.verbose: print(f"shape of j before reshape = {j.shape}")
        
        j = j.sum(2)
        j = j.reshape(*(j.shape[:2]), -1)
        if self.verbose: print(f"shape of j after reshape = {j.shape}")
        
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
        P = torch.diag_embed(p, dim1=1)
        pp = torch.einsum("zi,zj -> zij", p, p)
        
        return torch.einsum("zji, zjk, zkl -> zil", J_s, (P - pp), J_s)

    
    def fim_on_data(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        
        J_p = self.jac_proba(eval_point)
        G = self.local_data_matrix(eval_point)
        G_on_data = torch.einsum("zai, zij, zbj -> zab", J_p, G, J_p)

        return G_on_data

    
    def hessian_gradproba(
        self, 
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing H(p_a)𝛁p_b 

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor (H(p_a)𝛁p_b)_l with dimensions (bs, a,b,l).
        """

        J_p = self.jac_proba(eval_point)
        J_s = self.jac_score(eval_point)
        P = self.proba(eval_point)
        C = P.shape[-1]
        I = torch.eye(C).unsqueeze(0)
        N = P.unsqueeze(-2).expand(-1, C, -1)
        
        """Compute """
        first_term = torch.einsum("zbi, zki, zak, zal -> zabl", J_p, J_s, (I-N), J_p) 
        
        """Compute """
        second_term = torch.einsum("za, zbi, zki, zkl -> zabl", P, J_p, J_s, J_p )
        
        return first_term - second_term

    
    
    def lie_bracket(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing [𝛁p_a, 𝛁p_b] = H(p_b)𝛁p_a - H(p_a)𝛁p_b

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor [𝛁p_a, 𝛁p_b]_l with dimensions (bs, a,b,l)
        """

        H_grad = self.hessian_gradproba(eval_point)
        
        return H_grad.transpose(-2, -3) - H_grad

    
    def bra_grad_lie(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing 〈𝛁p_a, [𝛁p_b, 𝛁p_c]〉

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor 〈𝛁p_a, [𝛁p_b, 𝛁p_c]〉 with dimensions (bs, a, b, c)
        """

        G = self.local_data_matrix(eval_point)
        J_p = self.jac_proba(eval_point)
        lie = self.lie_bracket(eval_point)

        return torch.einsum("zai, zij, zbcj -> zabc", J_p, G, lie) 
        
    
    def grad_metric(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing 𝛁p_a(G_x) = J(s)^T A_a J(s).

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor 𝛁p_a(G_x)_kl with dimensions (bs, a, k, l)
        """

        J_p = self.jac_proba(eval_point)
        J_s = self.jac_proba(eval_point)
        p = self.proba(eval_point)
        """Compute p_l ∇p_k"""
        p_gradp = torch.einsum("zl, zki -> zikl", p, J_p)
        
        """Compute δ_kl 𝛁p_k"""
        delta_gradp = torch.eye(J_p.shape[-2]) * J_p.unsqueeze(-1).transpose(-2, -3)

        return torch.einsum("zai, zbk, zibc, zcl -> zakl", 
                            J_p, J_s, delta_gradp - p_gradp - p_gradp.transpose(-1,-2), J_s)
   
    
    def grad_bra_grad(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing 𝛁p_a〈𝛁p_b, 𝛁p_c〉

        Args:
            eval_point (torch.Tensor): Batch of points of the 
            input space at which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor 𝛁p_a〈𝛁p_b, 𝛁p_c〉with dimensions (bs, a, b, c)
        """
        
        p = self.proba(eval_point)
        P = torch.diag_embed(p, dim1=1)
        pp = torch.einsum("zi,zj -> zij", p, p)
        J_p = self.jac_proba(eval_point)
        J_s = self.jac_score(eval_point)
        grad_G = self.grad_metric(eval_point)

        H_grad = self.hessian_gradproba(eval_point)
        elmt_1 = torch.einsum("zbai, zdi, zde, zej, zcj -> zabc", H_grad, J_s, (P - pp), J_s, J_p)
        elmt_2 = torch.einsum("zbk, zakl, zcl -> zabc", J_p, grad_G, J_p)

         
        if self.verbose: 
            print(f"Shape of elmt_1: {elmt_1.shape}")
            print(f"Shape of elmt_2: {elmt_2.shape}")

         
        return elmt_1 + elmt_2 + elmt_1.permute(0, 1, 3, 2) 
    
    def bra_connection(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Function computing ⟨∇_(e_a) e_b, e_c⟩ with e_a = ∇p_a

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor ⟨∇_(e_a) e_b, e_c⟩ with dimensions (bs, a, b, c)
        """
        
        elmt_1 = self.grad_bra_grad(eval_point)
        elmt_2 = self.bra_grad_lie(eval_point)
        
        return ( elmt_1 + elmt_1.permute(0, 2, 3, 1) - elmt_1.permute(0, 3, 1, 2) - elmt_2 + elmt_2.permute(0, 2, 3, 1) + elmt_2.permute(0, 3, 1, 2) ) / 2 

    
    def connection_form(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the connection form ω(e_k) on the basis e_k = ∇p_k.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor ω^i_j(e_k) with dimensions (bs, i, j, k) 
        """
        
        C = self.bra_connection(eval_point)
        J_p = self.jac_proba(eval_point)
        G = self.local_data_matrix(eval_point)
        G_on_data = torch.einsum("zai, zij, zbj -> zab", J_p, G, J_p)
        print(f"G: {G} \nĜ {G_on_data}\nJ: {J_p}")
        G_inv = torch.cholesky_inverse(torch.linalg.cholesky(G_on_data))
        # if self.verbose:
        #     print("plotting")
        #     plt.matshow(G_on_data[0].detach().numpy()) 
        #     plt.show()
        #     plt.matshow(G_inv[0].detach().numpy())
        #     plt.show()
        
        return torch.einsum("zil, zkjl -> zijk", G_inv, C)
    
    
    def jac_connection(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the jacobian of the connection form.

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor ∂_l ω^i_j(e_k) with dimensions (bs, i, j, k, l)
        """

        if not self.verbose:
            print(f"GC: shape of eval_point = {eval_point.shape}")
            print(f"GC: shape of output = {self.proba(eval_point).shape}")
        j = jacobian(self.connection_form, eval_point)
        if not self.verbose: print(f"GC: shape of j before reshape = {j.shape}")
        
        j = j.sum(4)  # TODO: vérifier pourquoi on somme sur les batchs de l'entrée
        j = j.reshape(*(j.shape[:4]), -1)
        if not self.verbose: print(f"GC: shape of j after reshape = {j.shape}")
        
        
        return j
    
    
    def connection_lie(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ω^i_j([e_a, e_b])

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor ω^i_j([e_a, e_b]) with dimensions (bs, i, j, a, b).
        """
        
        omega = self.connection_form(eval_point)
        J_p = self.jac_proba(eval_point)
        J_s = self.jac_score(eval_point)
        P = self.proba(eval_point)
        C = P.shape[-1]
        I = torch.eye(C).unsqueeze(0)
        N = P.unsqueeze(-2).expand(-1, C, -1)
        # print(N)
        
        elmt_1 = torch.einsum("zbl, zcl, zac, zija -> zijab", J_p, J_s, (I-N), omega) 

        elmt_2 = torch.einsum("za, zbl, zkl, zijk -> zijab", P, J_p, J_s, omega)
        
        return (elmt_1 - elmt_2).transpose(-1, -2) - (elmt_1 - elmt_2)


    def grad_hessian_gradproba(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute e_a (H(p_b) ∇p_c)

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor e_a (H(p_b) ∇p_c)_l with dimension (bs, a, b, c, l)
        """
        
        J_p = self.jac_proba(eval_point)
        J_s = self.jac_score(eval_point)
        H_grad = self.hessian_gradproba(eval_point)
        P = self.proba(eval_point)
        C = P.shape[-1]
        I = torch.eye(C).unsqueeze(0)
        N = P.unsqueeze(-2).expand(-1, C, -1)
        
        result = - torch.einsum('zai, zki, zkj, zbj, zcl -> zabcl',
                                 J_p, J_p, J_s, J_p, J_p) \
                + torch.einsum('zck, zki, zbai, zcl -> zabcl',
                               (I-N), J_s, H_grad, J_p) \
                + torch.einsum('zck, zki, zbi, zcal -> zabcl', 
                               (I-N), J_s, J_p, H_grad) \
                - torch.einsum('zai, zci, zkj, zbj, zkl -> zabcl',
                               J_p, J_p, J_s, J_p, J_p) \
                - torch.einsum('zc, zki, zbai, zkl -> zabcl',
                               P, J_s, H_grad, J_p) \
                - torch.einsum('zc, zki, zbi, zkal -> zabcl',
                               P, J_s, J_p, H_grad)
        
        return result
        

    def grad_grad_ang(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute e_a (e_b ⟨e_c, e_d⟩)

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor e_a (e_b ⟨e_c, e_d⟩) with dimensions (bs, a, b, c, d).
        """
        grad_H_grad = self.grad_hessian_gradproba(eval_point)
        H_grad = self.hessian_gradproba(eval_point)
        G = self.local_data_matrix(eval_point)
        J_p = self.jac_proba(eval_point)
        grad_G = self.grad_metric(eval_point)
        
        # elmt_1 := e_a(M_{b,c,d})
        elmt_1 = torch.einsum('zacbl, zlk, zdk -> zabcd', 
                              grad_H_grad, G, J_p) \
                + torch.einsum('zcbk, zakl, zdl -> zabcd',
                               H_grad, grad_G, J_p) \
                + torch.einsum('zcbk, zkl, zdal -> zabcd',
                               H_grad, G, H_grad)  

        """ elmt_2_1 := ∇p_a^T H(p_c)^T e_b(G_x) ∇p_d """
        elmt_2_1 = torch.einsum("zcak, zbkl, zdl -> zabcd",
                                H_grad, grad_G, J_p)

        J_s = self.jac_proba(eval_point)
        p = self.proba(eval_point)
        """Compute p_l ∇p_k"""
        p_gradp = torch.einsum("zl, zki -> zikl", p, J_p)
        
        """Compute δ_kl 𝛁p_k"""
        delta_gradp = torch.eye(J_p.shape[-2]) * J_p.unsqueeze(-1).transpose(-2, -3)
        """Compute δ_kl H(p_k) 𝛁p_a (bs, i, a, l, k)"""
        delta_H_grad = torch.eye(H_grad.shape[-3]) * H_grad.unsqueeze(-1).transpose(-2, -4)
        """Compute ∇p_a^T ∇p_b"""
        gradp_gradp = torch.einsum("zai, zbi -> zab", J_p, J_p)
        """Compute p_k∇p_b^T ∇p_l"""
        p_gradp_gradp = torch.einsum("zk, zbl -> zbkl", p, gradp_gradp).unsqueeze(-4)
        """Compute (∇p_a^T ∇p_l)(∇p_b^T ∇p_k)"""
        four_gradp = torch.einsum("zal, zbk -> zabkl", gradp_gradp, gradp_gradp)
        """Compute e_a(A_b)_kl"""
        grad_A = torch.einsum("zbai, zikl  -> zabkl",
                                   H_grad, delta_gradp - p_gradp - p_gradp.transpose(-1, -2)) \
                    + four_gradp \
                    - 2 * p_gradp_gradp \
                    + torch.einsum("zbi, zialk -> zabkl",
                                   J_p, delta_H_grad) \
                    - torch.einsum("zbi, zl, zkai -> zabkl",
                                   J_p, p, H_grad) \
                    + torch.einsum("zbi, zikl -> zbkl",
                                   J_p, delta_gradp).unsqueeze(-4) \
                    - p_gradp_gradp.transpose(-1, -2) \
                    - four_gradp.transpose(-1, -2) 

        """Compute e_a(e_b(G_x))"""
        grad_grad_G = torch.einsum("zki, zabkl, zlj -> zabij",
                                   J_s, grad_A, J_s)
        """ elmt_2_2 := ∇p_c^T e_a(e_b(G_x)) ∇p_d """
        elmt_2_2 = torch.einsum("zck, zabkl, zdl -> zabcd",
                                J_p, grad_grad_G, J_p)
        """ elmt_2 := e_a(∇p_c^T e_b(G_x) ∇p_d) """
        elmt_2 =  elmt_2_1 + elmt_2_1.permute(0, 1, 2, 4, 3) + elmt_2_2
        
        return elmt_1 + elmt_1.transpose(-1, -2) + elmt_2


    def grad_ang_lie(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute e_a (⟨e_b, [e_c, e_d]⟩)

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor e_a (⟨e_b, [e_c, e_d]⟩) with dimensions (bs, a, b, c, d).
        """

        J_p = self.jac_proba(eval_point)
        H_grad = self.hessian_gradproba(eval_point)
        lie = H_grad.transpose(1, 2) - H_grad
        G = self.local_data_matrix(eval_point)        
        grad_G = self.grad_metric(eval_point)
        grad_H_gradp = self.grad_hessian_gradproba(eval_point)
        grad_lie = grad_H_gradp.transpose(2, 3) - grad_H_gradp
        
        return torch.einsum("zbal, zlk, zcdk -> zabcd",
                            H_grad, G, lie) \
             + torch.einsum("zbi, zaij, zcdi -> zabcd",
                            J_p, grad_G, lie) \
             + torch.einsum("zbi, zij, zacdj -> zabcd",
                            J_p, G, grad_lie)

    def grad_connection_ang(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute e_a (⟨∇_{e_b} e_c, e_d⟩)
        It uses the formula 2 e_a (⟨∇_{e_b} e_c, e_d⟩) = e_a (e_b ⟨e_c, e_d⟩ 
        + e_c ⟨e_d, e_b⟩ - e_d ⟨e_b, e_c⟩ - ⟨e_b, [e_c, e_d]⟩ + ⟨e_c, [e_d, e_b]⟩
        + ⟨e_d, [e_b, e_c]⟩).

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor e_a (⟨∇_{e_b} e_c, e_d⟩) with dimensions (bs, a, b, c, d).
        """

        elmt_1 = self.grad_grad_ang(eval_point)
        elmt_2 = self.grad_ang_lie(eval_point)
        
        return 0.5 * (elmt_1
                      + elmt_1.permute(0, 1, 3, 4, 2)
                      - elmt_1.permute(0, 1, 4, 2, 3)
                      - elmt_2 
                      + elmt_2.permute(0, 1, 3, 4, 2)
                      + elmt_2.permute(0, 1, 4, 2, 3)) 


    def grad_connection(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute e_a(ω^i_j(e_b)).
        It uses the formula dω^i_j(X,Y) = Xω^i_j(Y) - Yω^i_j(X) - ω^i_j([X,Y]).

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor e_a(ω^i_j(e_b)) with dimensions (bs, i, j, a, b).
        """

        grad_connection_ang = self.grad_connection_ang(eval_point)
        grad_ang = self.grad_bra_grad(eval_point)
        connection = self.connection_form(eval_point)
        J_p = self.jac_proba(eval_point)
        G = self.local_data_matrix(eval_point)
        G_on_data = torch.einsum("zai, zij, zbj -> zab", J_p, G, J_p)
        G_inv = torch.cholesky_inverse(torch.linalg.cholesky(G_on_data))

        N = grad_connection_ang - torch.einsum("zicb, zaid -> zabcd", connection, grad_ang)
        
        
        return torch.einsum("zdi, zabci -> zabcd", G_inv, N)

    
    def d_connection_form(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the exterior derivative of the connection form: dω^i_j(e_a, e_b).
        It uses the formula dω^i_j(X,Y) = Xω^i_j(Y) - Yω^i_j(X) - ω^i_j([X,Y]).

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor dω^i_j(e_a, e_b) with dimensions (bs, i, j, a, b).
        """
        
        J_omega = self.jac_connection(eval_point)
        print(f"J_omega: {J_omega.shape}")
        J_p = self.jac_proba(eval_point)
        elmt_1_old = torch.einsum("zak, zijbk -> zijab", J_p, J_omega)
        elmt_1 = self.grad_connection(eval_point)
        elmt_2 = self.connection_lie(eval_point)
        mask = ~elmt_1_old.isnan() * ~ elmt_1.isnan()
        i = 2
        # print(f"Elmt_1_old =\n {elmt_1_old[0,i,i,:4,:4]}")
        # print(f"Elmt_1 =\n {elmt_1[0,i,i,:4,:4]}")
        print(f"Is it a good estimate for domaga? {'Yes' if torch.allclose(elmt_1[mask], elmt_1_old[mask], equal_nan = True) else 'No'}\n \
                Error mean = {(elmt_1_old[mask]-elmt_1[mask]).pow(2).mean()}\n \
                Max error = {(elmt_1_old[mask]-elmt_1[mask]).abs().max()} out of {max(elmt_1_old[mask].abs().max(), elmt_1[mask].abs().max())}")
        
        return elmt_1 - elmt_1.transpose(-1, -2) - elmt_2
        
        
    def wedge_connection_forms(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the exterior product of the connection forms: ∑_k ω^i_k(e_a) ∧ ω^k_j(e_b).

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor (∑_k ω^i_k(e_a) ∧ ω^k_j(e_b)) with dimensions (bs, i, j, a, b).
        """
        
        omega = self.connection_form(eval_point)
        
        elmt = torch.einsum("zika, zkjb -> zijab", omega, omega)

        return elmt - elmt.transpose(-1, -2)

    
    def curvature_form(
        self,
        eval_point: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the curvature forms Ω^i_j(e_a, e_b).

        Args:
            eval_point (torch.Tensor): Batch of points of the input space at
            which the expression is evaluated.

        Returns:
            torch.Tensor: Tensor (Ω^i_j(e_a, e_b)) with dimensions (bs, i, j, a, b)
        """

        domega = self.d_connection_form(eval_point)
        wedge = self.wedge_connection_forms(eval_point)
        
        return domega + wedge


    def plot_debug(
        self,
        eval_point: torch.Tensor,
        output_dir: Union[str, Path]="./data",
    ) -> None:
        saving_idx = datetime.now().strftime("%y%m%d-%H%M%S")
        j = self.jac_proba(eval_point)
        fim_on_data = self.fim_on_data(eval_point).detach()
        bs, C, x = j.shape
        bs, _, px_row, px_col = eval_point.shape
        n_row, n_col = floor(sqrt(C)) - 1 , ceil(sqrt(C)) + 1
        for img, jac_img, fim_img in zip(eval_point, j, fim_on_data):
            fig_2, axes_2 = plt.subplots(1, 2)
            
            img_subplot = axes_2[0].matshow(img.squeeze(0))
            fig_2.colorbar(img_subplot, ax=axes_2[0])
            axes_2[0].set_title(f'Image.')
            
            fim_img_subplot = axes_2[1].matshow(fim_img, norm=SymLogNorm(fim_img.abs().min().numpy()))
            fig_2.colorbar(fim_img_subplot, ax=axes_2[1])
            axes_2[1].set_title(r'$G(e_i,e_j)$')

            plt.savefig(f"{output_dir}/{saving_idx}_img_and_metric.pdf", transparent=True)
            fig, axes = plt.subplots(n_row, n_col)

            for classe in range(C):
                row = classe // n_col
                col = classe % n_col
                im = axes[row, col].matshow(jac_img[classe].reshape(px_row, px_col))
                axes[row, col].tick_params(left = False, right = False, top=False, labeltop=False, labelleft = False , labelbottom = False, bottom = False)
                axes[row, col].set_title(f'Direction {classe}.')
                fig.colorbar(im, ax=axes[row, col])
            plt.savefig(f"{output_dir}/{saving_idx}_plot_grad_proba.pdf", transparent=True)
        
        
        
if __name__ == "__main__":
    mnist = True
    if mnist:
        checkpoint_path = './checkpoint/medium_cnn_10.pt'
        network = mnist_networks.medium_cnn(checkpoint_path)
        network_score = mnist_networks.medium_cnn(checkpoint_path, score=True)

        normalize = transforms.Normalize((0.1307,), (0.3081,))

        input_space = datasets.MNIST(
            "data",
            train=False,  # TODO: True ?
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), normalize]),
        )
    else:
        checkpoint_path = './checkpoint/xor_net_05.pt'
        network = xor_networks.xor_net(checkpoint_path)
        network_score = xor_networks.xor_net(checkpoint_path, score=True)

        input_space = xor_datasets.XorDataset(nsample=100000, test=True, discrete=False)

    device = next(network.parameters()).device
            
    curvature = model_curvature_computer(network, network_score, input_space, verbose=False)
    
    points = torch.cat([curvature.get_point()[0].unsqueeze(0) for _ in range(1)])
    print(f"Shape of points: {points.shape}")
    random_points = torch.rand_like(points)
    probas = curvature.jac_proba(points)
    fim_on_data = curvature.fim_on_data(points)
    curvature.plot_debug(points)
    # for img, metric, proba in zip(point, fim_on_data, probas):
    #     plt.matshow(img.squeeze(0))
    #     plt.matshow(proba, aspect="auto")
    #     plt.matshow(metric.detach().numpy())
    #     plt.show()
    # Omega = curvature.curvature_form(point)
    # print(Omega.max(), Omega.min())
    # print(f"Mean curvature tensors: {Omega.mean()}")
    # print(f"Proportion of nan: {Omega.isnan().sum() / Omega.numel():.6f}")
