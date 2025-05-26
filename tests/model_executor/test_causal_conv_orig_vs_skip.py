import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.mamba.ops import (
  causal_conv1d_fn, causal_conv1d_update,
  causal_conv1d_fn_skip_conv, causal_conv1d_update_skip_conv,
)


@pytest.mark.parametrize("batch,dim,seqlen,width", [
    (1,  3, 10, 3),
    (2,  4,  5, 5),
    (4, 16, 20, 7),
])
@pytest.mark.parametrize("activation", [None, "silu", "swish"])
@pytest.mark.parametrize("skip_initial", [0, 1, 3, 15])
def test_fn_against_original(batch, dim, seqlen, width, activation, skip_initial):
    x0 = torch.randn(batch, dim, seqlen, dtype=torch.float32)
    weight = torch.randn(dim, width, dtype=torch.float32)
    bias = torch.randn(dim, dtype=torch.float32)

    # original run
    x_orig = x0.clone()
    out_orig = causal_conv1d_fn(
        x_orig, weight, bias,
        query_start_loc=None, cache_indices=None,
        has_initial_state=None, conv_states=None,
        activation=activation,
    )

    # wrapper with skip_initial=0 must exactly match
    if skip_initial == 0:
        x_wrap0 = x0.clone()
        out_wrap0 = causal_conv1d_fn_skip_conv(
            x_wrap0, weight, bias,
            skip_initial=0,
            activation=activation,
        )
        torch.testing.assert_allclose(out_wrap0, out_orig, atol=1e-5, rtol=1e-4)

    # for skip_initial >0, compare suffix to original on sliced inputs
    if 0 < skip_initial < seqlen:
        K = skip_initial
        # run original on suffix only: slice input & adjust pad
        x_slice = x0[:, :, K:].clone()
        # emulate calling the kernel on only that suffix
        out_slice = causal_conv1d_fn(
            x_slice, weight, bias,
            activation=activation,
        )
        # now call full wrapper
        x_full = x0.clone()
        out_full = causal_conv1d_fn_skip_conv(
            x_full, weight, bias,
            skip_initial=K,
            activation=activation,
        )
        torch.testing.assert_allclose(
            out_full[:, :, K:], out_slice,
            atol=1e-5, rtol=1e-4
        )

    # skip_initial >= seqlen: only bias+act, same as original on an all-pad input
    if skip_initial >= seqlen:
        x_full = x0.clone()
        out_full = causal_conv1d_fn_skip_conv(
            x_full, weight, bias,
            skip_initial=skip_initial,
            activation=activation,
        )
        expected = x0 + bias.view(1, dim, 1)
        if activation == "silu":
            expected = F.silu(expected)
        elif activation == "swish":
            expected = expected * torch.sigmoid(expected)
        torch.testing.assert_allclose(out_full, expected, atol=1e-5, rtol=1e-4)


# state-update tests

@pytest.mark.parametrize("batch,dim,width", [
    (1, 4, 3),
    (2, 8, 5),
])
@pytest.mark.parametrize("activation", [None, "silu", "swish"])
@pytest.mark.parametrize("skip_initial", [0, 2, 5])
def test_update_vs_original(batch, dim, width, activation, skip_initial):
    state_len = width - 1
    # two separate state buffers
    state_orig = torch.zeros(batch, dim, state_len)
    state_wrap = state_orig.clone()

    weight = torch.randn(dim, width)
    bias = torch.randn(dim)
    cache_seqlens = torch.zeros(batch, dtype=torch.int32)

    # single‐step update
    x_t = torch.randn(batch, dim)

    # original update
    causal_conv1d_update(
        x_t, state_orig, weight, bias,
        activation=activation,
        cache_seqlens=cache_seqlens,
        conv_state_indices=None,
    )

    # wrapper with skip_initial=0 muat match original
    causal_conv1d_update_skip_conv(
        x_t, state_wrap, weight, bias,
        skip_initial=0,
        activation=activation,
        cache_seqlens=cache_seqlens,
        conv_state_indices=None,
    )
    torch.testing.assert_allclose(state_wrap, state_orig, atol=1e-6, rtol=1e-5)

    # f skip_initial > 0 and cache_seqlens < skip_initial,
    # state_wrap must remain zero
    if skip_initial > 0:
        causal_conv1d_update_skip_conv(
            x_t, state_wrap, weight, bias,
            skip_initial=skip_initial,
            activation=activation,
            cache_seqlens=cache_seqlens,
            conv_state_indices=None,
        )
        assert torch.allclose(state_wrap, torch.zeros_like(state_wrap))
