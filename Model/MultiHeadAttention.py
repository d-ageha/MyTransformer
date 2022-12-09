import torch
import math
import torch.nn as nn
from typing import Optional


class Attention(nn.Module):
    """Attention mechanism"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor]):
        # mask: i x j tensor
        scale = math.sqrt(query.size()[1])
        h = torch.div(torch.einsum("bhik,bhjk->bhij", query, key), scale)
        if mask is not None:
            h.masked_fill(mask, float(-math.inf))
        weight = nn.functional.softmax(h, dim=0)
        return torch.einsum("bhij,bhjv->bhiv", weight, value)


class MultiheadAttention(nn.Module):
    """Multihead Attention layer"""

    def __init__(self, head_num: int, q_dim: int, k_dim: int, v_dim: int,
                 is_masked: bool = False) -> None:
        super().__init__()
        self.qh_dim = max((int)(k_dim / head_num), 1)
        self.kh_dim = max((int)(k_dim / head_num), 1)

        self.head_num = head_num
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.q_w = nn.Parameter(torch.rand(head_num, q_dim, self.kh_dim,
                                           requires_grad=True))
        self.k_w = nn.Parameter(torch.rand(head_num, k_dim, self.kh_dim,
                                           requires_grad=True))
        self.v_w = nn.Parameter(torch.rand(head_num, v_dim, self.kh_dim,
                                           requires_grad=True))
        self.o_w = nn.Parameter(torch.rand(head_num * self.kh_dim, q_dim,
                                           requires_grad=True))
        self.att = Attention()
        if is_masked:
            mask = torch.ones(q_dim, q_dim)
            mask = torch.triu(mask, 1)
            self.register_buffer("mask", mask)
        else:
            self.mask = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        """ forward
            query: b x i x q tensor
            key: b x j x k tensor
            value: b x j x v tensor
            pad_mask: b x i

            i = length of query sequence
            j = length of key/value sequence
        """
        """
            s: q/head_num
            t: k/head_num
        """

        # if pad_mask is passed, I assume i = j  (self attention)
        mask = self.mask
        if pad_mask is not None:
            if mask is not None:
                mask = (mask == 1) | (pad_mask == 1)
            else:
                mask = pad_mask == 1

        weighted_q = torch.einsum("biq,hqs->bhis", query, self.q_w)
        weighted_k = torch.einsum("bjk,hks->bhjs", key, self.k_w)
        weighted_v = torch.einsum("bjv,hvt->bhjt", value, self.v_w)

        heads = self.att.forward(weighted_q, weighted_k, weighted_v, mask)
        # heads.shape: [batch_size, head_num, query[0].shape, vh_dim]
        heads = heads.view(heads.shape[0], -1, self.head_num * self.kh_dim)
        return torch.einsum("bit,to->bio", heads, self.o_w)
