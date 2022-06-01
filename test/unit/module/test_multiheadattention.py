import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional

from megengine import tensor
from megengine.functional import full
from xopr.module import MultiHeadAttention


@pytest.mark.skipif(sys.version_info < (3, 6), reason="requires python3.6 or higher")
def test_multiheadattention_example():
    # init query key and value
    features = np.arange(0, 24)
    features_temp = np.where(
        features < 20, features, np.zeros_like(features, dtype=np.float32)
    )
    features = tensor(features_temp.reshape(2, 3, 4), dtype="float32")
    features_torch = torch.tensor(features_temp.reshape(2, 3, 4)).float()
    key_test_diff_dim_torch = torch.tensor(np.arange(0, 48).reshape(2, 3, 8)).float()
    value_test_diff_dim_torch = key_test_diff_dim_torch
    key_test_diff_dim_mge = tensor(np.arange(0, 48).reshape(2, 3, 8), dtype="float32")
    value_test_diff_dim_mge = key_test_diff_dim_mge
    query = features
    key = features
    value = features
    attn_mask_mge = tensor(
        np.array(
            [
                [[True, True, True], [True, True, True], [True, True, True]],
                [[True, False, True], [True, False, True], [True, False, True]],
                [[True, False, True], [True, False, True], [True, False, True]],
                [[True, False, True], [True, False, True], [True, False, True]],
            ],
            dtype=np.bool,
        )
    )
    attn_mask_torch = torch.tensor(
        np.array(
            [
                [[True, True, True], [True, True, True], [True, True, True]],
                [[True, False, True], [True, False, True], [True, False, True]],
                [[True, False, True], [True, False, True], [True, False, True]],
                [[True, False, True], [True, False, True], [True, False, True]],
            ],
            dtype=np.bool,
        )
    )
    key_padding_mask_mge = tensor(
        np.array([[False, True, True], [True, False, True]]), dtype=bool
    )
    key_padding_mask_torch = torch.tensor(
        np.array([[False, True, True], [True, False, True]], dtype=np.bool)
    )

    # 1.test query = key = value
    # nn.multiheadattention
    multihead_attn_same = nn.MultiheadAttention(
        4,
        2,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=True,
        device=None,
    )
    multihead_attn_same.in_proj_weight.data.fill_(0.01)  # init
    multihead_attn_same.in_proj_bias.data.fill_(0.00)  # init
    multihead_attn_same.out_proj.weight.data.fill_(0.01)  # init
    attn_output, attn_output_weights = multihead_attn_same(
        features_torch, features_torch, features_torch, need_weights=None
    )

    # my multiheadattention
    attention_same_test = MultiHeadAttention(
        4, 2, dropout=0.0, bias=True, batch_first=True
    )
    # init weight and bias
    attention_same_test.query_linear.weight = full(
        attention_same_test.query_linear.weight.shape, 0.01
    )
    attention_same_test.key_linear.weight = full(
        attention_same_test.key_linear.weight.shape, 0.01
    )
    attention_same_test.value_linear.weight = full(
        attention_same_test.value_linear.weight.shape, 0.01
    )
    attention_same_test.out_linear.weight = full(
        attention_same_test.out_linear.weight.shape, 0.01
    )
    attention_same_test.query_linear.bias = (
        attention_same_test.key_linear.bias
    ) = (
        attention_same_test.value_linear.bias
    ) = attention_same_test.out_linear.bias = None

    my_attn_output, my_attn_output_weights = attention_same_test(
        query, key, value, need_weights=None, attn_mask=None
    )

    # compare nn.multiheadattention and my multiheadattention
    np.testing.assert_allclose(
        attn_output.detach().numpy(), my_attn_output.detach().numpy(), atol=1e-6
    )

    # 2. test query != key and value
    # nn.multiheadattention
    multihead_attn_diff = nn.MultiheadAttention(
        4,
        2,
        dropout=0.0,
        bias=True,
        add_bias_kv=True,
        add_zero_attn=True,
        kdim=8,
        vdim=8,
        batch_first=True,
        device=None,
    )
    multihead_attn_diff.q_proj_weight.data.fill_(0.01)  # init
    multihead_attn_diff.k_proj_weight.data.fill_(0.01)  # init
    multihead_attn_diff.v_proj_weight.data.fill_(0.01)  # init
    multihead_attn_diff.out_proj.weight.data.fill_(0.01)  # init
    multihead_attn_diff.bias_k.data.fill_(0.0)  # init
    multihead_attn_diff.bias_v.data.fill_(0.0)  # init
    multihead_attn_diff.in_proj_bias.data.fill_(0.0)  # init
    multihead_attn_diff.out_proj.bias.data.fill_(0.0)  # init
    attn_output, attn_output_weights = multihead_attn_diff(
        features_torch,
        key_test_diff_dim_torch,
        value_test_diff_dim_torch,
        key_padding_mask=key_padding_mask_torch,
        need_weights=True,
        attn_mask=attn_mask_torch,
    )

    # my multiheadattention
    attention_diff_test = MultiHeadAttention(
        4,
        2,
        dropout=0.0,
        bias=True,
        add_bias_kv=True,
        add_zero_attn=True,
        kdim=8,
        vdim=8,
        batch_first=True,
    )
    # init weight and bias
    attention_diff_test.query_linear.weight = full(
        attention_diff_test.query_linear.weight.shape, 0.01
    )
    attention_diff_test.key_linear.weight = full(
        attention_diff_test.key_linear.weight.shape, 0.01
    )
    attention_diff_test.value_linear.weight = full(
        attention_diff_test.value_linear.weight.shape, 0.01
    )
    attention_diff_test.out_linear.weight = full(
        attention_diff_test.out_linear.weight.shape, 0.01
    )
    attention_diff_test.query_linear.bias = (
        attention_diff_test.key_linear.bias
    ) = (
        attention_diff_test.value_linear.bias
    ) = attention_diff_test.out_linear.bias = None
    attention_diff_test.bias_k.reset_zero()
    attention_diff_test.bias_v.reset_zero()
    # return result
    my_attn_output, my_attn_output_weights = attention_diff_test(
        query,
        key_test_diff_dim_mge,
        value_test_diff_dim_mge,
        key_padding_mask=key_padding_mask_mge,
        need_weights=True,
        attn_mask=attn_mask_mge,
    )

    # compare nn.multiheadattention and my multiheadattention
    np.testing.assert_allclose(
        attn_output.detach().numpy(), my_attn_output.detach().numpy(), atol=1e-7
    )
    np.testing.assert_allclose(
        attn_output_weights.detach().numpy(),
        my_attn_output_weights.detach().numpy(),
        atol=1e-7,
    )