from JESC_DataSet import JESC_DataSet
import sys
import os
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), 'Model'))
from Transformer import Transformer



class EtoJModel(torch.nn.Module):
    def __init__(self, model_dim: int, max_seq_len: int, en_embs: int, ja_embs: int) -> None:
        super().__init__()
        self.transformer = Transformer(6, 6, 8, model_dim, max_seq_len, 0.1)
        self.en_emb = torch.nn.Embedding(en_embs, model_dim)
        self.ja_emb = torch.nn.Embedding(ja_embs, model_dim)

    def forward(self, x):
        pass


def train(train: str, val: str,  epoch=10, batch=5, lr=0.01):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    print(device)
    train_dataloader = DataLoader(JESC_DataSet(train), batch, True)
    val_dataloader = DataLoader(JESC_DataSet(val), 1, False)

    model = Transformer(6, 6, 8, 728, 100, 0.1)
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
            out = model(en, ja)
            loss = loss_fn(out, ja)
            train_loss += float(loss)
            loss.backward()
            t.set_postfix_str("loss={}".format(loss))
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
    model = train("dataset/train_p", "dataset/dev_p", 10)
