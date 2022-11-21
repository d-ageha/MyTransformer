
import torch
import math
import torch.nn as nn
from typing import Optional


class Attention(nn.Module):
    """Attention mechanism"""

    def __init__(self, is_masked: bool = False, maxq_len: Optional[int] = None, maxk_len: Optional[int] = None) -> None:
        super().__init__()
        if is_masked:
            if maxq_len is None or maxk_len is None:
                raise Exception(
                    "Attention: you must set maxq_len and maxk_len when is_masked=True")
            mask = torch.ones(maxq_len, maxk_len)
            mask = (mask*(-float("inf")))+mask
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        scale = math.sqrt(query.size()[1])
        h = torch.div(torch.einsum("bhik,bhjk->bhij", query, key),scale)
        weight = nn.functional.softmax(h, dim=0)
        if self.mask is not None:
            weight = torch.einsum("bhiv,iv->bhiv", weight, self.mask)

        return torch.einsum("bhij,bhjv->bhiv", weight, value)


class MultiheadAttention(nn.Module):
    """Multihead Attention layer"""

    def __init__(self, head_num: int, q_dim: int, k_dim: int, v_dim: int,
                 is_masked: bool = False, maxq_len: Optional[int] = None, maxk_len: Optional[int] = None) -> None:
        super().__init__()
        self.kh_dim = max((int)(k_dim/head_num), 1)
        self.vh_dim = max((int)(k_dim/head_num), 1)

        if is_masked and (maxq_len is None or maxk_len is None):
            raise Exception("set dimensions of possible inputs")

        self.head_num = head_num
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.q_w = nn.Parameter(torch.rand(head_num, q_dim, self.kh_dim,
                              requires_grad=True))
        self.k_w = nn.Parameter(torch.rand(head_num, k_dim, self.kh_dim,
                              requires_grad=True))
        self.v_w = nn.Parameter(torch.rand(head_num, v_dim, self.vh_dim,
                              requires_grad=True))
        self.o_w = nn.Parameter(torch.rand(head_num * self.vh_dim, q_dim,
                              requires_grad=True))
        self.att = Attention(is_masked, maxq_len, maxk_len)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """ forward
            query: i x q tensor
            key: j x k tensor
            value: j x v tensor
            i = length of query
            j = length of key/value
        """
        """
            s: kh_dim
            t: vh_dim
        """
        weighted_q = torch.einsum("biq,hqs->bhis", query, self.q_w)
        weighted_k = torch.einsum("bjk,hks->bhjs", key, self.k_w)
        weighted_v = torch.einsum("bjv,hvt->bhjt", value, self.v_w)

        heads = self.att.forward(weighted_q, weighted_k, weighted_v)
        # heads.shape: [batch_size, head_num, query[0].shape, vh_dim]
        return torch.einsum("bit,to->bio", heads.view(heads.shape[0], -1, self.head_num*self.vh_dim), self.o_w)
