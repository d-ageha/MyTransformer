from math import cos, sin
import torch
import torch.nn as nn
from MultiHeadAttention import MultiheadAttention


class Transformer(nn.Module):

    def __init__(self, dec_num: int, enc_num: int,
                 head_num: int, model_dim: int, max_seq_len: int, drop_rate: float) -> None:
        super().__init__()
        self.decoders = [DecoderLayer(head_num, model_dim, drop_rate)
                         for x in range(dec_num)]
        self.encoders = [EncoderLayer(head_num, model_dim, drop_rate)
                         for x in range(enc_num)]
        self.linear = torch.nn.Linear(model_dim, model_dim)

        for i, decoder in enumerate(self.decoders):
            self.register_module("decoder"+str(i), decoder)
        for i, encoder in enumerate(self.encoders):
            self.register_module("encoder"+str(i), encoder)

        pos_enc = [[sin(x/pow(10000, y/model_dim)) if y % 2 == 0 else cos(x/pow(10000, (y-1)/model_dim))
                    for y in range(model_dim)] for x in range(max_seq_len)]
        self.register_buffer("pos_enc", torch.tensor(pos_enc))

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        enc = x + (torch.Tensor)(self.pos_enc)[:x.shape[1],
                                               :x.shape[2]].expand(x.shape[0], -1, -1)
        for encoder in self.encoders:
            enc = encoder.forward(enc)
        dec = y + (torch.Tensor)(self.pos_enc)[:y.shape[1],
                                               :y.shape[2]].expand(y.shape[0], -1, -1)
        for decoder in self.decoders:
            dec = decoder.forward(dec, enc)
        dec = self.linear(dec)
        return torch.nn.functional.softmax(dec, 2)


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

    def __init__(self, h_num: int, model_dim: int, drop_rate: float) -> None:
        super().__init__()
        self.ff_layer = FeedForward(model_dim, model_dim*4, model_dim)
        self.att_layer = MultiheadAttention(
            h_num, model_dim, model_dim, model_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x: torch.Tensor):
        h = self.att_layer(x, x, x)
        h = self.dropout(h)+x
        h = nn.functional.layer_norm(h, h.size())
        h2 = self.ff_layer(h)
        h2 = self.dropout(h2)+h
        return nn.functional.layer_norm(h2, h2.size())


class DecoderLayer(nn.Module):
    """ A single layer of Transformer's decoder part. """

    def __init__(self, h_num: int, model_dim: int, drop_rate: float) -> None:
        super().__init__()
        self.ff_layer = FeedForward(model_dim, model_dim*4, model_dim)
        self.maksed_att_layer = MultiheadAttention(
            h_num, model_dim, model_dim, model_dim)
        self.att_layer = MultiheadAttention(
            h_num, model_dim, model_dim, model_dim)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, enc):
        h = self.maksed_att_layer(x, x, x)
        h = self.dropout(h)+x
        h = nn.functional.layer_norm(h, h.size())
        h2 = self.att_layer(h, enc, enc)
        h2 = self.dropout(h2)+h
        h2 = nn.functional.layer_norm(h2, h2.size())
        h3 = self.ff_layer(h2)
        h3 = self.dropout(h3)+h2
        return nn.functional.layer_norm(h3, h3.size())


if __name__ == "__main__":
    from transformers import AutoTokenizer
    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ja_tokenizer = AutoTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese")
    en_tokenizer.add_special_tokens({ "eos_token":"[EOS]"})
    print(en_tokenizer.all_special_tokens)
    print(ja_tokenizer.all_special_tokens)
    en_embedding = torch.nn.Embedding(
        len(en_tokenizer), 64, en_tokenizer.pad_token_id)
    ja_embedding = torch.nn.Embedding(
        len(ja_tokenizer), 64, ja_tokenizer.pad_token_id)

    en = "I want to do some stupid things in my life"
    ja = "なにか馬鹿みたいなことでもしたいなぁ"

    print(en_tokenizer(en), ja_tokenizer(ja))
    en_tokens = torch.tensor(en_tokenizer(en)["input_ids"])
    en_emb = en_embedding(en_tokens)
    ja_tokens = torch.tensor(en_tokenizer(ja)["input_ids"])
    ja_emb = ja_embedding(ja_tokens)
    print(en_tokens, en_emb)
    out = Transformer(6, 6, 8, 64, 30, 0.1).forward(
        en_emb.expand(1, -1, -1), ja_emb.expand(1, -1, -1))

    ja_out = torch.norm(
        ja_embedding.weight.data.expand(out.shape[1], -1, -1).transpose(0, 1)-out, dim=1)
    ja_token = torch.argmin(ja_out, dim=0)

    ja_tes = torch.norm(
        ja_embedding.weight.data.expand(ja_emb.shape[0], -1, -1).transpose(0, 1)-ja_emb, dim=1)
    ja_teso = torch.argmin(ja_tes, dim=0)
    print(ja_token.shape)
    print(ja_tokenizer.decode(ja_token))
    print(ja_tokenizer.decode(torch.argmin(torch.norm(ja_embedding.weight.data-ja_emb[2],dim=1))))
