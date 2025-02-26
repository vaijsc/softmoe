from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def softmax(x: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """
    Compute the softmax along the specified dimensions.
    This function adds the option to specify multiple dimensions

    Args:
        x (torch.Tensor): Input tensor.
        dims (int or tuple[int]): The dimension or list of dimensions along which the softmax probabilities are computed.

    Returns:
        torch.Tensor: Output tensor containing softmax probabilities along the specified dimensions.
    """
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


class SoftMoELayerWrapper(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
        self,
        dim: int,
        num_experts: int,
        slots_per_expert: int,
        layer: Callable,
        normalize: bool = True,
        **layer_kwargs,
    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            layer (Callable): Network layer of the experts.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
            **layer_kwargs: Additional keyword arguments for the layer class.
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize

        # Initialize phi and normalization scaling factor
        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        # Initialize phi using LeCun normal initialization
        # https://github.com/google-research/vmoe/blob/662341d007650d5bbb7c6a2bef7f3c759a20cc7e/vmoe/projects/soft_moe/router.py#L49C1-L49C1
        nn.init.normal_(self.phi, mean=0, std=1 / dim**0.5)

        # Create a list of expert networks
        self.experts = nn.ModuleList(
            [layer(**layer_kwargs) for _ in range(num_experts)]
        )
        self.additional_params = nn.Parameter(torch.ones(1, 1, 192))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
            x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
            len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi  # torch.Size([192, 128, 5])

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)  # torch.Size([256, 197, 128, 5])
        d = softmax(logits, dim=1)  # torch.Size([256, 197, 128, 5])
        c = softmax(logits, dim=(2, 3))  # torch.Size([256, 197, 128, 5])

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)  # torch.Size([256, 128, 5, 192])

        # Apply expert to corresponding slots
        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        )  # torch.Size([256, 128, 5, 192])

        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c)  # torch.Size([256, 197, 192])
        y = y * x
        expand_dim = self.additional_params.expand(y.shape[0], y.shape[1], -1)  # [256, 197, 192]
        y = y * expand_dim
        
        similarity_matrix = torch.matmul(x, x.transpose(1, 2))  # [batch_size, seq_length, seq_length]
        # Use the lower triangular part of the similarity matrix
        similarity_matrix = torch.tril(similarity_matrix)
        # Step 3: Normalize similarities using softmax
        normalized_similarity = F.softmax(similarity_matrix, dim=-1)  # [batch_size, seq_length, seq_length]
        y = torch.matmul(normalized_similarity, y)  # Out-of-place update
        return y

