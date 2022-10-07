from typing import Tuple

import torch
import torch.nn as nn
from torch.autograd.functional import jacobian


def local_data_matrix(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    training_state = model.training
    if training_state:
        model.eval()
    # noinspection PyTypeChecker
    j = jacobian(model, x.unsqueeze(0)).squeeze(0)
    j = j.reshape(j.size(0), -1)  # concatenates (bash, -1)
    with torch.no_grad():
        p = torch.exp(model(x.unsqueeze(0)))
        jacobian_product = torch.bmm(j.unsqueeze(2), j.unsqueeze(1)).permute(1, 2, 0)
        g_matrix = torch.sum(p * jacobian_product, dim=-1)
    if training_state:
        model.train()
    return g_matrix


def local_data_matrix_trace_rank(
    model: nn.Module, x: torch.Tensor
) -> Tuple[float, float]:
    training_state = model.training
    if training_state:
        model.eval()
    # noinspection PyTypeChecker
    j = jacobian(model, x.unsqueeze(0)).squeeze(0)
    j = j.reshape(j.size(0), -1)
    rank = torch.clamp(torch.linalg.matrix_rank(j), max=j.size(0) - 1)
    with torch.no_grad():
        p = torch.exp(model(x.unsqueeze(0)))
        trace = torch.sum(p * torch.pow(j, 2).sum(dim=1))
    if training_state:
        model.train()
    return trace.item(), rank.item()


def batch_data_matrix_trace_rank(
    model: nn.Module, batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    traces, ranks = torch.tensor(
        [local_data_matrix_trace_rank(model, x) for x in batch]
    ).T
    ranks = ranks.to(torch.int64)
    return traces, ranks