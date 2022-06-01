import math
from abc import abstractmethod
from typing import Optional, Tuple

from megengine import tensor

from megengine.functional import zeros
from ..functional.nn import multiheadattention
from megengine.tensor import Parameter
from megengine.module import init, Linear, Module


class MultiHeadAttention(Module):
    """
    Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O

    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.

    Args:
        embed_dim(int): total dimension of the model.
        num_heads(int): parallel attention heads.
        dropout(float): a Dropout layer on attn_weight. Default: 0.0.
        bias(bool): add bias as module parameter. Default: True.
        add_zero_attn(bool): add a new batch of zeros to the key and value sequences at axis=1.
        add_bias_kv(bool): add bias to the key and value sequences at axis=0.
        kdim(int): total number of features in key. Default: None.
        vdim(int): total number of features in value. Default: None.
        batch_first(bool): If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        compute_mode(str): Support mixed precision acceleration in linear.

        Note: if attr:`kdim` and :attr:`vdim` are None, they will be set
        to :attr:`embed_dim` such that query, key, and value have the same
        number of features.

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
    Examples::

            import numpy as np

            import megengine as mge
            from megengine.functional import full
            from megengine.module import *


            # init query ,key and value
            features = np.arange(0, 24)
            features_temp = np.where(
                features < 20, features, np.zeros_like(features, dtype=np.float32)
            )
            features = mge.tensor(features_temp.reshape(2, 3, 4), dtype="float32")
            query = key = value = features

            my_attn = MultiHeadAttention(4, 2, dropout=0.0, bias=True, batch_first=True)
            # init weight and bias
            my_attn.query_linear.weight = full(my_attn.query_linear.weight.shape, 0.01)
            my_attn.key_linear.weight = full(my_attn.key_linear.weight.shape, 0.01)
            my_attn.value_linear.weight = full(my_attn.value_linear.weight.shape, 0.01)
            my_attn.out_linear.weight = full(my_attn.out_linear.weight.shape, 0.01)
            my_attn.query_linear.bias = (
                my_attn.key_linear.bias
            ) = my_attn.value_linear.bias = my_attn.out_linear.bias = None
            # return result
            my_attn_output, my_attn_output_weights = my_attn(query, key, value,)
            print("my_attn_output=", my_attn_output)
            print("my_attn_output_weights=", my_attn_output_weights)

                
        Output:

            my_attn_output= Tensor([[[0.0089 0.0089 0.0089 0.0089]
            [0.009  0.009  0.009  0.009 ]
            [0.0092 0.0092 0.0092 0.0092]]

           [[0.0191 0.0191 0.0191 0.0191]
            [0.0197 0.0197 0.0197 0.0197]
            [0.0165 0.0165 0.0165 0.0165]]], device=xpux:0)

            my_attn_output_weights= Tensor([[[0.3288 0.3333 0.3379]
            [0.3169 0.3331 0.3501]
            [0.3051 0.3325 0.3624]]

           [[0.3582 0.4047 0.2371]
            [0.3626 0.4249 0.2125]
            [0.3333 0.3333 0.3333]]], device=xpux:0)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        kdim: int = None,
        vdim: int = None,
        bias: bool = True,
        add_zero_attn: bool = False,
        add_bias_kv: bool = False,
        batch_first: bool = False,
        compute_mode: str = "default",
    ):
        # init parameter
        super(MultiHeadAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.query_linear = Linear(embed_dim, embed_dim, bias, compute_mode)
        self.key_linear = Linear(self.kdim, embed_dim, bias, compute_mode)
        self.value_linear = Linear(self.vdim, embed_dim, bias, compute_mode)
        self.out_linear = Linear(embed_dim, embed_dim, bias, compute_mode)
        self.add_zero_attn = add_zero_attn
        self.dropout = dropout
        self.batch_first = batch_first
        if add_bias_kv:
            self.add_bias_kv = add_bias_kv
            self.bias_k = Parameter(zeros((1, 1, embed_dim), dtype="float32"))
            self.bias_v = Parameter(zeros((1, 1, embed_dim), dtype="float32"))
        else:
            self.bias_k = self.bias_v = None
        self.compute_mode = compute_mode
        self._reset_parameters()

    def _reset_parameters(self):
        if self.bias_k is not None:
            init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            init.xavier_normal_(self.bias_v)

    def forward(
        self,
        query: tensor,
        key: tensor,
        value: tensor,
        key_padding_mask: Optional[tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[tensor] = None,
    ) -> Tuple[tensor, Optional[tensor]]:
        """
            Args:
                query, key, value(tensor): map a query and a set of key-value pairs to an output.
                    See "Attention Is All You Need" for more details.
                key_padding_mask(tensor): if provided, specified padding elements in the key will
                    be ignored by the attention.
                need_weights(bool): output attn_output_weights.
                attn_mask(bool): 3D mask that prevents attention to certain positions. A 3D mask allows to specify a different mask for the entries of each batch.
        
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
        if self.batch_first:
            query, key, value = [x.transpose(1, 0, 2) for x in (query, key, value)]

        attn, attn_weight = multiheadattention(
            query,
            key,
            value,
            self.head_dim,
            self.num_heads,
            self.query_linear.weight,
            self.query_linear.bias,
            self.key_linear.weight,
            self.key_linear.bias,
            self.value_linear.weight,
            self.value_linear.bias,
            self.out_linear.weight,
            self.out_linear.bias,
            self.dropout,
            self.bias_k,
            self.bias_v,
            self.add_zero_attn,
            key_padding_mask,
            need_weights,
            attn_mask,
            self.compute_mode,
        )

        if self.batch_first:
            return attn.transpose(1, 0, 2), attn_weight
        else:
            return attn, attn_weight
