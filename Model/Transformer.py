from math import cos, sin
from typing import Optional
import torch
import torch.nn as nn
from MultiHeadAttention import MultiheadAttention


class Transformer(nn.Module):

    def __init__(self, dec_num: int, enc_num: int,
                 head_num: int, model_dim: int, max_seq_len: int, drop_rate: float) -> None:
        super().__init__()
        self.decoders = [DecoderLayer(head_num, max_seq_len, max_seq_len, model_dim, drop_rate)
                         for x in range(dec_num)]
        self.encoders = [EncoderLayer(head_num, max_seq_len, model_dim, drop_rate)
                         for x in range(enc_num)]
        self.linear = torch.nn.Linear(model_dim, model_dim)

        for i, decoder in enumerate(self.decoders):
            self.register_module("decoder" + str(i), decoder)
        for i, encoder in enumerate(self.encoders):
            self.register_module("encoder" + str(i), encoder)

        pos_enc = [[sin(x / pow(10000, y / model_dim)) if y % 2 == 0 else cos(x / pow(10000, (y - 1) / model_dim))
                    for y in range(model_dim)] for x in range(max_seq_len)]
        self.register_buffer("pos_enc", torch.tensor(pos_enc))

    def forward(self, x: torch.Tensor, y: torch.Tensor, x_pad_mask: Optional[torch.Tensor] = None, y_pad_mask: Optional[torch.Tensor] = None):
        enc = x + self.pos_enc
        for encoder in self.encoders:
            enc = encoder.forward(enc, x_pad_mask)
        dec = y + self.pos_enc
        for decoder in self.decoders:
            dec = decoder.forward(dec, enc, y_pad_mask, x_pad_mask)
        return dec


class FeedForward(nn.Module):
    """Feed Forward Layer

        One layer of fully connected neural network with relu activation,
        followed by the same network structure without activation.
    """

    def __init__(self, in_dim: int, mid_dim: int, out_dim: int) -> None:
        super().__init__()
        self.first_layer = nn.Linear(in_dim, mid_dim)
        self.second_layer = nn.Linear(mid_dim, out_dim)

    def forward(self, x: torch.Tensor):
        return self.second_layer(torch.relu(self.first_layer(x)))


class EncoderLayer(nn.Module):
    """ A single layer of Transformer's encoder part. """

    def __init__(self, h_num: int, max_in_length: int, model_dim: int, drop_rate: float) -> None:
        super().__init__()
        self.ff_layer = FeedForward(model_dim, model_dim * 4, model_dim)
        self.att_layer = MultiheadAttention(
            h_num, max_in_length, max_in_length, model_dim, model_dim, model_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.lnorm1 = nn.LayerNorm(model_dim)
        self.lnorm2 = nn.LayerNorm(model_dim)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None):
        h = self.att_layer(x, x, x, pad_mask)
        h = self.dropout1(h) + x
        h = self.lnorm1(h)
        h2 = self.ff_layer(h)
        h2 = self.dropout2(h2) + h
        return self.lnorm2(h2)


class DecoderLayer(nn.Module):
    """ A single layer of Transformer's decoder part. """

    def __init__(self, h_num: int, max_in_length: int, max_out_length: int, model_dim: int, drop_rate: float) -> None:
        super().__init__()
        self.ff_layer = FeedForward(model_dim, model_dim * 4, model_dim)
        self.maksed_att_layer = MultiheadAttention(
            h_num, max_in_length, max_in_length, model_dim, model_dim, model_dim, True)
        self.att_layer = MultiheadAttention(
            h_num, max_in_length, max_out_length, model_dim, model_dim, model_dim)
        self.dropout1 = nn.Dropout(drop_rate)
        self.dropout2 = nn.Dropout(drop_rate)
        self.dropout3 = nn.Dropout(drop_rate)
        self.lnorm1 = nn.LayerNorm(model_dim)
        self.lnorm2 = nn.LayerNorm(model_dim)
        self.lnorm3 = nn.LayerNorm(model_dim)

    def forward(self, x, enc, x_pad_mask: Optional[torch.Tensor] = None, enc_pad_mask: Optional[torch.Tensor] = None):
        h = self.maksed_att_layer(x, x, x, x_pad_mask)
        h = self.dropout1(h) + x
        h = self.lnorm1(h)
        h2 = self.att_layer(h, enc, enc, enc_pad_mask)
        h2 = self.dropout2(h2) + h
        h2 = self.lnorm1(h2)
        h3 = self.ff_layer(h2)
        h3 = self.dropout3(h3) + h2
        return self.lnorm1(h3)
