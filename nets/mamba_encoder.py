"""Pure PyTorch Mamba-1 blocks and an MRTA encoder adapter.

Reference: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
https://arxiv.org/abs/2312.00752

This reference implementation uses a sequential scan, without custom CUDA
kernels or an einops dependency. MRTA tasks are sorted by (x, y, deadline)
before scanning, and their embeddings are restored to the original task IDs.
"""
from dataclasses import dataclass
import math
from typing import Union

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int  # Historical name: continuous input feature dimension.
    d_state: int = 8
    expand: int = 1
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 2
    pad_vocab_size_multiple: int = 1
    conv_bias: bool = True
    bias: bool = False

    def __post_init__(self):
        for name in ('d_model', 'n_layer', 'vocab_size', 'd_state', 'expand',
                     'd_conv', 'pad_vocab_size_multiple'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError('{} must be a positive integer'.format(name))
        self.d_inner = self.expand * self.d_model
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        if not isinstance(self.dt_rank, int) or self.dt_rank < 1:
            raise ValueError('dt_rank must be auto or a positive integer')
        multiple = self.pad_vocab_size_multiple
        self.vocab_size = math.ceil(self.vocab_size / multiple) * multiple


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        # Accumulate low-precision squares in float32 to avoid overflow.
        values = x.float() if x.dtype in (torch.float16, torch.bfloat16) else x
        normalized = values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class MambaBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, 2 * args.d_inner, bias=args.bias)
        self.conv1d = nn.Conv1d(args.d_inner, args.d_inner, args.d_conv,
                                groups=args.d_inner, padding=args.d_conv - 1,
                                bias=args.conv_bias)
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + 2 * args.d_state, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        rates = torch.arange(1, args.d_state + 1, dtype=torch.float32)
        self.A_log = nn.Parameter(rates.log().repeat(args.d_inner, 1))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        length = x.size(1)
        values, gate = self.in_proj(x).chunk(2, dim=-1)
        values = self.conv1d(values.transpose(1, 2))[:, :, :length].transpose(1, 2)
        output = self.ssm(F.silu(values)) * F.silu(gate)
        return self.out_proj(output)

    def ssm(self, x):
        delta, B, C = self.x_proj(x).split(
            [self.args.dt_rank, self.args.d_state, self.args.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, -self.A_log.float().exp(), B, C, self.D.float())

    def selective_scan(self, u, delta, A, B, C, D):
        delta_A = torch.exp(torch.einsum('bld,dn->bldn', delta, A))
        delta_B_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
        state = delta_A.new_zeros((u.size(0), u.size(2), A.size(1)))
        outputs = []
        for index in range(u.size(1)):
            state = delta_A[:, index] * state + delta_B_u[:, index]
            outputs.append(torch.einsum('bdn,bn->bd', state, C[:, index]))
        return torch.stack(outputs, dim=1) + u * D


class ResidualBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)

    def forward(self, x):
        return x + self.mixer(self.norm(x))


class Mamba_encoder(nn.Module):
    """Tensor encoder; input is [batch, length, args.vocab_size]."""

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.init_embed = nn.Linear(args.vocab_size, args.d_model)
        self.layers = nn.Sequential(*(ResidualBlock(args) for _ in range(args.n_layer)))
        self.norm_f = RMSNorm(args.d_model)

    def encode_embeddings(self, embeddings):
        return self.norm_f(self.layers(embeddings))

    def forward(self, x, mask=None):
        if mask is not None:
            raise ValueError('Mamba expects unpadded task batches; masks are not supported')
        if x.ndim != 3 or x.size(-1) != self.args.vocab_size:
            raise ValueError('Expected [batch, length, {}] input features'.format(self.args.vocab_size))
        h = self.encode_embeddings(self.init_embed(x))
        return h, h.mean(dim=1)


class MambaEncoder(Mamba_encoder):
    """MRTA adapter with canonical ordering and depot at output index 0.

    Single-direction Mamba remains sensitive to sequence order. A lexicographic
    sort makes that order depend on task features instead of arbitrary IDs.
    Equal-feature tasks share the average of their encoded representations so
    stable-sort ties cannot introduce an ID-dependent representation.
    """

    def __init__(self, n_dim=128, n_layers=3):
        super().__init__(ModelArgs(d_model=n_dim, n_layer=n_layers, vocab_size=3))
        self.init_embed_depot = nn.Linear(2, n_dim)

    def forward(self, data, mask=None):
        if mask is not None:
            raise ValueError('Mamba expects unpadded task batches; masks are not supported')
        deadline = data['deadline']
        features = torch.cat((data['loc'],
                              (deadline / deadline.amax(dim=1, keepdim=True))[:, :, None]), dim=-1)
        batch, tasks, _ = features.shape
        order = torch.arange(tasks, device=features.device).expand(batch, tasks)
        # Sort least significant keys first, yielding lexicographic (x, y, d).
        for feature in (2, 1, 0):
            values = features[:, :, feature].gather(1, order)
            order = order.gather(1, torch.sort(values, dim=1, stable=True).indices)
        sorted_features = features.gather(1, order[:, :, None].expand(-1, -1, 3))
        embeddings = torch.cat((self.init_embed_depot(data['depot']),
                                self.init_embed(sorted_features)), dim=1)
        encoded = self.encode_embeddings(embeddings)
        task_embeddings = encoded[:, 1:]

        different = (sorted_features[:, 1:] != sorted_features[:, :-1]).any(dim=-1)
        if not different.all():
            starts = torch.cat((different.new_ones((batch, 1)), different), dim=1)
            groups = starts.long().cumsum(dim=1) - 1
            indices = groups[:, :, None].expand_as(task_embeddings)
            sums = torch.zeros_like(task_embeddings).scatter_add(1, indices, task_embeddings)
            counts = task_embeddings.new_zeros((batch, tasks, 1)).scatter_add(
                1, groups[:, :, None], task_embeddings.new_ones((batch, tasks, 1)))
            task_embeddings = sums.gather(1, indices) / counts.gather(1, groups[:, :, None])

        restored = torch.zeros_like(task_embeddings).scatter(
            1, order[:, :, None].expand_as(task_embeddings), task_embeddings)
        h = torch.cat((encoded[:, :1], restored), dim=1)
        return h, h.mean(dim=1)
