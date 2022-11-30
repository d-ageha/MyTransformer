from torch.utils.data.dataloader import DataLoader
import torch
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'Model'))
from Transformer import Transformer
from JESC_DataSet import JESC_DataSet


class EtoJModel(torch.nn.Module):
    def __init__(self, model_dim: int, max_seq_len: int, en_embs: int, ja_embs: int) -> None:
        super().__init__()
        self.transformer = Transformer(6, 6, 8, model_dim, max_seq_len, 0.1)
        self.en_emb = torch.nn.Embedding(en_embs, model_dim)
        self.ja_emb = torch.nn.Embedding(ja_embs, model_dim)

    def forward(self, x, y):
        x = self.en_emb(x)
        y = self.ja_emb(y)
        #y = y[:, :-1, :]
        return self.transformer(x, y)

    def en_embed(self, x):
        return self.en_emb(x)

    def ja_embed(self, y):
        return self.ja_emb(y)

    def en_decode(self, x):
        weight = self.en_emb.weight.data.expand(
            x.shape[0], -1, -1).transpose(0, 1)
        return torch.argmin(torch.norm(weight - x, dim=2), dim=0)

    def ja_decode(self, y):
        weight = self.ja_emb.weight.data.expand(
            y.shape[0], -1, -1).transpose(0, 1)
        return torch.argmin(torch.norm(weight - y, dim=2), dim=0)


def train(train: str, val: str, dim=256, epoch=10, batch=1, lr=0.01):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print(device)
    max_length = 100
    train_dataset = JESC_DataSet(train, max_length)
    val_dataset = JESC_DataSet(val, max_length)
    train_dataloader = DataLoader(train_dataset, batch, True)
    val_dataloader = DataLoader(val_dataset, 1, False)

    model = EtoJModel(dim, max_length, len(train_dataset.en_tokenizer),
                      len(train_dataset.ja_tokenizer))
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(epoch):
        model.train(True)
        train_loss = 0.0
        valid_loss = 0.0
        t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in t:
            optim.zero_grad()
            en, ja = data["en"].to(device), data["ja"].to(device)
            # print(train_dataset.ja_tokenizer.decode(ja[0]), train_dataset.ja_tokenizer.decode(
            #    model.ja_decode(model.ja_embed(ja[0]))))
            out = model(en, ja)
            loss = loss_fn(out, model.ja_embed(ja))
            train_loss += float(loss)
            loss.backward()
            torch.no_grad()
            print(train_dataset.ja_tokenizer.decode(model.ja_decode(out[0])))
            with torch.no_grad():
                t.set_postfix_str("Epoch: {} loss={}".format(e, loss))
            if i % 100 == 0:
                torch.save(model.state_dict(), "output/JESC_Transformer_Model")

        train_loss /= train_dataloader.__len__()
        model.train(False)

        for i, data in tqdm(enumerate(val_dataloader), total=len(val_dataloader)):
            en, ja = data["en"].to(device), data["ja"].to(device)
            out = model(en)
            loss = loss_fn(out, ja)
            valid_loss += float(loss)
        valid_loss /= val_dataloader.__len__()
        print("Train Loss:{}, Valid Loss:{}".format(train_loss, valid_loss))

    return model


if __name__ == "__main__":
    print(torch.__version__)
    model = train("dataset/train_p", "dataset/dev_p", 100, 10)
