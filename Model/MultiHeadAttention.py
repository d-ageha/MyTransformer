import torch
import math
import torch.nn as nn
from typing import Optional


class Attention(nn.Module):
    """Attention mechanism"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor]):
        # mask: b x i x j tensor
        scale = math.sqrt(query.size()[1])
        h = torch.div(torch.einsum("bhik,bhjk->bhij", query, key), scale)
        if mask is not None:
            # ...what?
            h = h.transpose(0, 1).masked_fill(mask, -float("Inf")).nan_to_num().transpose(0, 1)
        weight = nn.functional.softmax(h, dim=-1)
        return torch.einsum("bhij,bhjv->bhiv", weight, value)


class MultiheadAttention(nn.Module):
    """Multihead Attention layer"""

    def __init__(self, head_num: int, q_length: int, k_length: int, q_dim: int, k_dim: int, v_dim: int,
                 is_masked: bool = False) -> None:
        super().__init__()
        self.qh_dim = q_dim // head_num
        self.kh_dim = k_dim // head_num

        self.head_num = head_num
        self.q_length = q_length
        self.k_length = k_length
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.q_linear = nn.Linear(q_dim, q_dim)
        self.k_linear = nn.Linear(k_dim, k_dim)
        self.v_linear = nn.Linear(v_dim, v_dim)
        self.o_linear = nn.Linear(self.kh_dim * head_num, v_dim)
        self.att = Attention()
        if is_masked:
            mask = torch.ones(q_length, k_length)
            mask = torch.tril(mask)
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
            pad_mask = pad_mask == 0
            pad_mask = pad_mask.repeat(1, 1, self.q_length).view(-1, self.q_length, self.q_length)
            if mask is not None:
                mask = pad_mask * mask
                mask = mask == 0
            else:
                mask = pad_mask

        query = self.q_linear(query).view(query.shape[0], self.head_num, self.q_length, self.qh_dim)
        key = self.k_linear(key).view(key.shape[0], self.head_num, self.k_length, self.kh_dim)
        value = self.v_linear(value).view(value.shape[0], self.head_num, self.k_length, self.kh_dim)
        heads = self.att.forward(query, key, value, mask)
        # heads.shape: [batch_size, head_num, query[0].shape, vh_dim]
        heads = heads.view(heads.shape[0], -1, self.head_num * self.kh_dim)
        return self.o_linear(heads)
