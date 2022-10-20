import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()


def Attention(query, key, value, is_masked=false):
    """Attention mechanism"""
    weight = torch.softmax(
        torch.einsum("ihk,jhk->ihj", query, key)/torch.sqrt(query.size()[1]))
    if is_masked:
        pass
    return torch.einsum("ihj,jhv->ihv", weight, value)

class MultiheadAttention(nn.Module):
    """Multihead Attention layer"""

    def __init__(self, h_num, model_dim, k_dim=None, v_dim=None) -> None:
        super().__init__()
        if k_dim == None:
            k_dim = model_dim/h_num
        if v_dim == None:
            v_dim = model_dim/h_num
        self.h_num = h_num
        self.model_dim = model_dim
        self.v_dim = v_dim
        self.q_w = torch.tensor(torch.random(model_dim, k_dim, h_num),
                                requires_grad=True)
        self.k_w = torch.tensor(torch.random(model_dim, k_dim, h_num),
                                requires_grad=True)
        self.v_w = torch.tensor(torch.random(model_dim, v_dim, h_num),
                                requires_grad=True)
        self.o_w = torch.tensor(torch.random(h_num * v_dim, model_dim, h_num),
                                requires_grad=True)

    def forward(self, query, key, value):
        """ forward 
            query: i x d_k tensor
            key: j x d_k tensor
            value: j x d_v tensor
        """
        weighted_q = torch.einsum("im,mhk->ihk", query, self.q_w)
        weighted_k = torch.einsum("jm,mhk->jhk", key, self.k_w)
        weighted_v = torch.einsum("jm,mhv->jhv", value, self.v_w)
        heads = Attention(weighted_q, weighted_k, weighted_v)
        return torch.einsum("it,to->io", heads.view(-1, self.h_num*self.v_dim), self.o_w)


class FeedForward(nn.Module):
    """Feed Forward Layer

        One layer of fully connected neural network with relu activation,
        followed by the same network structure without activation.
    """

    def __init__(self, in_dim, mid_dim, out_dim) -> None:
        super().__init__()
        self.first_layer = nn.Linear(in_dim, mid_dim)
        self.second_layer = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        return self.second_layer(torch.relu(self.first_layer(x)))


class EncoderLayer(nn.Module):
    """ A single layer of Transformer's encoder part. """

    def __init__(self, h_num, model_dim) -> None:
        super().__init__()
        self.ff_layer = FeedForward(model_dim, model_dim*4, model_dim)
        self.att_layer = MultiheadAttention(h_num, model_dim)

    def forward(self, x):
        h = self.att_layer(x)+x
        h = nn.LayerNorm(h)
        h = self.ff_layer(h)+x
        return nn.LayerNorm(h)


class DecoderLayer(nn.Module):
    """ A single layer of Transformer's decoder part. """

    def __init__(self, h_num, model_dim) -> None:
        super().__init__()


A = torch.tensor(
    [[-1, 1, 0],
     [1, 3, 1]])

B = torch.tensor(
    [[[0, 1, 0], [1, 1, 0]],
     [[-1, 2, 1], [0, 0, 0]],
     [[3, 2, 1], [-1, 2, 3]],
     [[3, 2, 1], [-1, 2, 3]]])

print(torch.einsum("ij,klj->ikl", A, B))
print(B.view(-1, 6))

test = FeedForward(10, 20, 10)
print(test.forward(torch.rand(10, 10)))
