from functools import lru_cache
from typing import NamedTuple, Optional, Sequence, Tuple, Union

from megengine import Tensor
from megengine.functional import (
    linear,
    repeat,
    concat,
    full,
    zeros,
    broadcast_to,
    logical_or,
    where,
    matmul,
    softmax,
    dropout,
)

__all__ = [
    "multiheadattention",
]


def multiheadattention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    head_dim: int,
    num_heads: int,
    query_weight: Tensor,
    query_bias: Optional[Tensor],
    key_weight: Tensor,
    key_bias: Optional[Tensor],
    value_weight: Tensor,
    value_bias: Optional[Tensor],
    attn_output_weight: Tensor,
    attn_output_bias: Optional[Tensor],
    dropout_p: float,
    bias_k: Tensor,
    bias_v: Tensor,
    add_zero_attn: bool = False,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    compute_mode: str = "default",
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Args:
        query, key, value(Tensor): map a query and a set of key-value pairs to an output.See "Attention Is All You Need" for more details.
        head_dim(int): total dimension of every head.
        num_heads(int): parallel attention heads.
        query_weight, query_bias, key_weight, key_bias, value_weight, value_bias(Tensor): input projection weight and bias.
        attn_output_weight, attn_output_bias(Tensor):output projection weight and bias.
        dropout_p(float): probability of an element to be zeroed.
        bias_k, bias_v(Tensor): bias of the key and value sequences to be added at dim=0.
        add_zero_attn(bool): add a new batch of zeros to the key and value sequences at dim=1.
        key_padding_mask(Tensor): if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -1e9.
        need_weights(bool): output attn_output_weights.
        attn_mask(Tensor): 3D mask that prevents attention to certain positions.

    Shape:
        Inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
            the embedding dimension.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
            the embedding dimension.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
            - attn_mask: 3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
            S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
            positions.

        Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
            E is the embedding dimension.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
            L is the target sequence length, S is the source sequence length.
    """
    tgt_len = query.shape[0]
    bsz = query.shape[1]

    # 1) Do all the linear projections in batch
    query = linear(query, query_weight, query_bias, compute_mode)
    key = linear(key, key_weight, key_bias, compute_mode)
    value = linear(value, value_weight, value_bias, compute_mode)
    # add bias along batch dimension
    if bias_k is not None and bias_v is not None:
        bias_k_temp = repeat(bias_k, bsz, axis=1)
        bias_v_temp = repeat(bias_v, bsz, axis=1)
        key = concat([key, bias_k_temp])
        value = concat([value, bias_v_temp])
        if attn_mask is not None:
            attn_mask_temp3 = full(
                (attn_mask.shape[0], attn_mask.shape[1], 1), False, dtype=bool
            )
            attn_mask = concat([attn_mask, attn_mask_temp3], axis=2)
        if key_padding_mask is not None:
            key_padding_mask_temp = full(
                (key_padding_mask.shape[0], 1), False, dtype=bool
            )
            key_padding_mask = concat([key_padding_mask, key_padding_mask_temp], axis=1)

    query = query.reshape(-1, bsz * num_heads, head_dim).transpose(1, 0, 2)
    key = key.reshape(-1, bsz * num_heads, head_dim).transpose(1, 0, 2)
    value = value.reshape(-1, bsz * num_heads, head_dim).transpose(1, 0, 2)
    # add zero attention along batch dimension
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        key = concat(
            [key, zeros(zero_attn_shape, dtype=key.dtype)],
            axis=1,
            device=key.device,
        )
        value = concat(
            [value, zeros(zero_attn_shape, dtype=value.dtype)],
            axis=1,
            device=value.device,
        )
        if attn_mask is not None:
            attn_mask_temp3 = full(
                (attn_mask.shape[0], attn_mask.shape[1], 1), False, dtype=bool
            )
            attn_mask = concat([attn_mask, attn_mask_temp3], axis=2)
        if key_padding_mask is not None:
            key_padding_mask_temp = full(
                (key_padding_mask.shape[0], 1), False, dtype=bool
            )
            key_padding_mask = concat([key_padding_mask, key_padding_mask_temp], axis=1)
    # update source sequence length after adjustments
    src_len = key.shape[1]

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape[0] == bsz
        assert key_padding_mask.shape[1] == src_len
        key_padding_mask = key_padding_mask.reshape(bsz, 1, 1, src_len)
        key_padding_mask = broadcast_to(
            key_padding_mask, (bsz, num_heads, 1, src_len)
        ).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = Tensor(logical_or(attn_mask, key_padding_mask))

    # convert mask to float
    if attn_mask is not None:
        new_attn_mask_temp1 = full(attn_mask.shape, 0.0)
        new_attn_mask_temp2 = full(attn_mask.shape, -1e9)
        new_attn_mask = where(attn_mask, new_attn_mask_temp2, new_attn_mask_temp1)
        attn_mask = new_attn_mask

    # 2) Apply attention on all the projected vectors in batch.
    attn_output_weights = matmul(
        query, key.transpose(0, 2, 1), compute_mode=compute_mode
    ) / (head_dim ** 0.5)
    if attn_mask is not None:
        attn_output_weights = attn_output_weights + attn_mask

    attn_output_weights = attn_output_weights.reshape(bsz * num_heads, tgt_len, src_len)
    attn_output_weights = softmax(attn_output_weights, axis=-1)
    if dropout_p > 0.0:
        attn_output_weights = dropout(attn_output_weights, dropout_p)
    attn_output = matmul(attn_output_weights, value, compute_mode=compute_mode)

    # 3) "Concat" using a reshape and apply a final linear.
    attn_output = attn_output.transpose(1, 0, 2).reshape(
        tgt_len, bsz, num_heads * head_dim
    )
    attn_output_weights = (
        attn_output_weights.reshape(bsz, num_heads, tgt_len, src_len).sum(axis=1)
        / num_heads
    )
    attn_output = linear(
        attn_output, attn_output_weight, attn_output_bias, compute_mode
    )
    if need_weights:
        return attn_output, attn_output_weights
    else:
        return attn_output, None
