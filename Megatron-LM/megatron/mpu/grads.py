# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch


import torch
from torch._six import inf

from apex.multi_tensor_apply import multi_tensor_applier
import amp_C

from .initialize import get_model_parallel_group, model_parallel_is_initialized
from .initialize import get_model_parallel_rank


def l2_grad_clipper(parameters, max_norm):
    """Efficient L2 norm gradient clipping."""

    overflow_buf = torch.zeros(1, dtype=torch.int, device='cuda')
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    # Filter parameters with gradients.
    parameters_with_grads = list(filter(
        lambda p: p.grad is not None, parameters))
    # Filter parameters for norm calculations.
    mp_rank_is_zero = (not model_parallel_is_initialized()) or (get_model_parallel_rank() == 0)
    parameters_for_norm = list(filter(
        lambda p: mp_rank_is_zero or p.model_parallel, parameters_with_grads))
    # Calculate L2 norm.
    norm, _ = multi_tensor_applier(
        amp_C.multi_tensor_l2norm,
        overflow_buf,
        [parameters_for_norm],
        False # no per-parameter norm
    )
    # Sum across all model parallel GPUs.
    norm_2 = norm * norm
    if model_parallel_is_initialized():
        torch.distributed.all_reduce(norm_2,
                                    op=torch.distributed.ReduceOp.SUM,
                                    group=get_model_parallel_group())
    total_norm = norm_2.item() ** 0.5
    # Scale to get max_norm.
    clip_coef = float(max_norm) / (total_norm + 1.0e-6)
    grads = [p.grad for p in parameters_with_grads]
    if clip_coef < 1.0:
        multi_tensor_applier(
            amp_C.multi_tensor_scale,
            overflow_buf,
            [grads, grads],
            clip_coef)
    return total_norm


def clip_grad_norm(parameters, max_norm, norm_type=2):
    """Clips gradient norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    model_parallel_enabled = model_parallel_is_initialized()
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        if model_parallel_enabled:
            # Take max across all GPUs.
            torch.distributed.all_reduce(total_norm_cuda,
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    #elif norm_type == 2:
    #    total_norm = l2_grad_clipper(parameters, max_norm)

    else:
        total_norm = 0
        for p in parameters:
            if not model_parallel_enabled or p.model_parallel or (get_model_parallel_rank() == 0):
                param_norm = p.grad.data.norm(norm_type)
                total_norm += param_norm.item() ** norm_type
        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        if model_parallel_enabled:
            torch.distributed.all_reduce(total_norm_cuda,
                                        op=torch.distributed.ReduceOp.SUM,
                                        group=get_model_parallel_group())
        total_norm = total_norm_cuda[0].item() ** (1. / norm_type)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for p in parameters:
                p.grad.data.mul_(clip_coef)
    return total_norm