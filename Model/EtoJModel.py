import torch
import sys
import os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "Model"))
from Transformer import Transformer


class EtoJModel(torch.nn.Module):
    def __init__(
        self,
        model_dim: int,
        en_pad_idx: int,
        ja_pad_idx: int,
        max_seq_len: int,
        en_embs: int,
        ja_embs: int,
        use_mine=True,
    ) -> None:
        super().__init__()
        self.use_mine = use_mine
        print(use_mine)
        if use_mine:
            print("Transformer Implementation: my model")
            self.transformer = Transformer(6, 6, 8, model_dim, max_seq_len, 0.1)
        else:
            print("Transformer Implementation: not mine")
            self.transformer = torch.nn.Transformer(model_dim, 8, batch_first=True)
        self.en_emb = torch.nn.Embedding(en_embs, model_dim, padding_idx=en_pad_idx)
        self.ja_emb = torch.nn.Embedding(ja_embs, model_dim, padding_idx=ja_pad_idx)
        self.linear = torch.nn.Linear(model_dim, ja_embs)
        self.max_seq_len = max_seq_len

    def forward(self, x, y, x_pad_mask, y_pad_mask):
        x = self.en_emb(x)
        y = self.ja_emb(y)
        if self.use_mine:
            out = self.transformer(x, y, x_pad_mask, y_pad_mask)
        else:
            out = self.transformer(
                x,
                y,
                src_key_padding_mask=x_pad_mask == 0,
                tgt_key_padding_mask=y_pad_mask == 0,
            )
        out = self.linear(out)
        return out

    def en_embed(self, x):
        return self.en_emb(x)

    def ja_embed(self, y):
        return self.ja_emb(y)

    def ja_decode(self, y):
        return torch.argmax(y, 2)

    def translate(self, x, sos_id, eos_id, ja_pad_id, pad_mask, device):
        result = torch.tensor([sos_id]).unsqueeze(0)
        x = x.to(device)
        pad_mask = pad_mask.to(device)
        for i in range(self.max_seq_len - 1):
            y_pads = torch.tensor([[ja_pad_id for j in range(self.max_seq_len - (i + 1))]])
            y = torch.cat((result, y_pads), dim=1)
            y_pad_mask = [1 for j in range(i + 1)] + [0 for j in range(self.max_seq_len - (i + 1))]
            y_pad_mask = torch.tensor(y_pad_mask, dtype=torch.int64).unsqueeze(0)
            if not self.use_mine:
                pad_mask = pad_mask == 0
                y_pad_mask = y_pad_mask == 0

            y = y.to(device)
            y_pad_mask = y_pad_mask.to(device)
            y = self.forward(x, y, pad_mask, y_pad_mask)
            next_token = torch.tensor([[self.ja_decode(y)[0, i]]])
            result = torch.cat((result, next_token), dim=1)
            if result[0, -1] == eos_id:
                break
        return result
