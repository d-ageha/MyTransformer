from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'Model'))
from Transformer import Transformer
from JESC_DataSet import JESC_DataSet


class EtoJModel(torch.nn.Module):
    def __init__(self, model_dim: int, en_pad_idx: int, ja_pad_idx: int, max_seq_len: int, en_embs: int, ja_embs: int) -> None:
        super().__init__()
        self.transformer = Transformer(6, 6, 8, model_dim, max_seq_len, 0.1)
        #self.transformer = torch.nn.Transformer(model_dim, 8, batch_first=True)
        self.en_emb = torch.nn.Embedding(en_embs, model_dim, padding_idx=en_pad_idx)
        self.ja_emb = torch.nn.Embedding(ja_embs, model_dim, padding_idx=ja_pad_idx)
        self.linear = torch.nn.Linear(model_dim, ja_embs)
        self.max_seq_len = max_seq_len

    def forward(self, x, y, x_pad_mask, y_pad_mask):
        x = self.en_emb(x)
        y = self.ja_emb(y)
        out = self.transformer(x, y, x_pad_mask, y_pad_mask)
        #out = self.transformer(x, y, src_key_padding_mask=x_pad_mask == 0, tgt_key_padding_mask=y_pad_mask == 0)
        out = self.linear(out)
        return out

    def en_embed(self, x):
        return self.en_emb(x)

    def ja_embed(self, y):
        return self.ja_emb(y)

    def ja_decode(self, y):
        return torch.argmax(y, 2)

    def translate(self, x, sos_id, eos_id):
        x = [self.en_emb(x)]
        y = [[sos_id]]
        y = self.ja_emb(y)
        result = ""
        for i in range(self.max_seq_len):
            y = self.transformer(x, y)
            result = self.ja_decode(y)
            if result[0, -1] == eos_id:
                break
        return result

    def load_model(self, path):
        self.transformer = torch.load(path)


def train(train: str, val: str, dim=256, epoch=10, batch=1, lr=0.01):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print(device)
    max_length = 70
    train_dataset = JESC_DataSet(train, max_length)
    val_dataset = JESC_DataSet(val, max_length)
    train_dataloader = DataLoader(train_dataset, batch, True)
    val_dataloader = DataLoader(val_dataset, 1, False)

    en_pad_id = train_dataset.en_tokenizer.pad_token_id or 0
    ja_pad_id = train_dataset.ja_tokenizer.pad_token_id or 0
    model = EtoJModel(dim, en_pad_id, ja_pad_id, max_length, len(
        train_dataset.en_tokenizer), len(train_dataset.ja_tokenizer))
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ja_pad_id, label_smoothing=0.1)
    for e in range(epoch):
        model.train(True)
        train_loss = 0.0
        valid_loss = 0.0
        t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in t:
            en, ja = data["en"], data["ja"]
            en_tokens = torch.stack(en["input_ids"]).transpose(0, 1).to(device)
            ja_tokens = torch.stack(ja["input_ids"]).transpose(0, 1).to(device)
            en_masks, ja_masks = torch.stack(en["attention_mask"]).transpose(
                0, 1).to(device), torch.stack(ja["attention_mask"]).transpose(0, 1).to(device)

            if en_tokens.dim() == 1:
                en_tokens, ja_tokens = en_tokens.unsqueeze(0), ja_tokens.unsqueeze(0)
                en_masks, ja_masks = en_masks.unsqueeze(0), ja_masks.unsqueeze(0)

            optim.zero_grad()
            out = model(en_tokens, ja_tokens, en_masks, ja_masks)
            loss = loss_fn(out.transpose(1, 2)[:, :, :-1], ja_tokens[:, 1:])
            train_loss += float(loss)
            loss.backward()
            optim.step()
            with torch.no_grad():
                t.set_postfix_str("Epoch: {} loss={}".format(e, loss))
            if i % 100 == 0:
                out_tokens = model.ja_decode(out).to(device)
                out_tokens = out_tokens.masked_fill(ja_masks == 0, ja_pad_id)
                print(out_tokens[0])
                print(train_dataset.ja_tokenizer.decode(out_tokens[0]))
                print(train_dataset.ja_tokenizer.decode(ja_tokens[0]))
                torch.save(model.state_dict(), "output/JESC_Transformer_Model")

        train_loss /= train_dataloader.__len__()
        model.train(False)

        for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            en, ja = data["en"]["input_ids"].to(device), data["ja"]["input_ids"].to(device)
            out = model(en)
            loss = loss_fn(out, ja)
            valid_loss += float(loss)
        valid_loss /= val_dataloader.__len__()
        print("Train Loss:{}, Valid Loss:{}".format(train_loss, valid_loss))

    return model


if __name__ == "__main__":
    print(torch.__version__)
    model = train("dataset/train_p", "dataset/dev_p", 128, 10, 5, lr=1)
