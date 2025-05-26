from typing import Optional

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.attention.backends.utils import PAD_SLOT_ID

def causal_conv1d_fn_skip_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    skip_initial: int = 0,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    A drop-in replacement for causal_conv1d_fn that skips convolution
    on the first skip_initial tokens of each chunk.

    skip_initial: int
        Number of initial positions in each sequence chunk to leave
        un-convolved (i.e. only bias+activation or identity).
    All other args are identical to causal_conv1d_fn.
    """
    # must be >= 0
    if skip_initial < 0:
        raise ValueError("skip_initial must be >= 0")

    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    batch, dim, seqlen = x.shape

    # if nothing to skip, revert to original implementation
    if skip_initial == 0 or seqlen <= skip_initial:
        ops.causal_conv1d_fwd(
            x, weight, bias, conv_states,
            query_start_loc, cache_indices,
            has_initial_state,
            activation in ["silu", "swish"],
            pad_slot_id
        )
        return x

    # separate prefix and suffix
    prefix = x[:, :, :skip_initial]
    suffix = x[:, :, skip_initial:]

    # process the suffix normally, but
    # we need to adjust query_start_loc so that
    # the kernel thinks the suffix starts at position 0
    if query_start_loc is not None:
        # subtract skip_initial from all start locs (but clamp to >=0)
        # since each chunk’s start loc in the global x was e.g. [0, L1, L1+L2,...],
        # we need the per-chunk offset within this suffix
        qsl = query_start_loc - skip_initial
        qsl = torch.clamp(qsl, min=0)
    else:
        qsl = None

    # cache_indices, has_initial_state, and conv_states are unchanged

    ops.causal_conv1d_fwd(
        suffix, weight, bias, conv_states,
        qsl, cache_indices, has_initial_state,
        activation in ["silu", "swish"],
        pad_slot_id
    )

    # stitch back
    x[:, :, skip_initial:] = suffix
    # prefix stays as-is (identity)
    if bias is not None or activation:
        if bias is not None:
            prefix = prefix + bias.view(1, -1, 1)
        if activation == "silu":
            prefix = torch.nn.functional.silu(prefix)
        elif activation == "swish":
            prefix = prefix * torch.sigmoid(prefix)
        x[:, :, :skip_initial] = prefix

    return x


def causal_conv1d_update_skip_conv(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    *,
    skip_initial: int = 0,
    activation: Optional[str] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    conv_state_indices: Optional[torch.Tensor] = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    Like causal_conv1d_update, but skips pushing the first skip_initial
    tokens into the state / convolution.
    """
    if skip_initial < 0:
        raise ValueError("skip_initial must be >= 0")
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    # original logic
    if skip_initial == 0 or cache_seqlens is None:
        return ops.causal_conv1d_update(
            x if x.dim() > 2 else x.unsqueeze(-1),
            conv_state,
            weight,
            bias,
            activation in ["silu", "swish"],
            cache_seqlens,
            conv_state_indices,
            pad_slot_id,
        )

    # which batch-positions are still in the initial skip range
    skip_mask = cache_seqlens < skip_initial  # (batch,)

    # nothing to do if we need to skip everything
    if skip_mask.all():
        return x

    # otherwise, we need to call the original update on the non-skipped subset
    keep_indices = torch.nonzero(~skip_mask, as_tuple=False).view(-1)

    # slice out their x, conv_state, cache_seqlens (and conv_state_indices)
    x_keep = x[keep_indices]
    state_keep = conv_state[keep_indices] if conv_state_indices is None else conv_state
    cs_keep = cache_seqlens[keep_indices]
    idx_keep = None if conv_state_indices is None else conv_state_indices[keep_indices]

    # run the normal update on those
    updated = ops.causal_conv1d_update(
        x_keep.unsqueeze(-1) if x_keep.dim()==2 else x_keep,
        state_keep,
        weight,
        bias,
        activation in ["silu", "swish"],
        cs_keep,
        idx_keep,
        pad_slot_id,
    )

    # scatter the updated state back
    if conv_state_indices is None:
        conv_state[keep_indices] = state_keep
    return x
