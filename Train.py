from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
from Model.Transformer import Transformer
from MachineTranslation.JESC_DataSet import JESC_DataSet


def train(train: str, val: str,  epoch=10, batch=5, lr=0.01):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache() 
    print(device)
    train_dataloader = DataLoader(JESC_DataSet(train), batch, True)
    val_dataloader = DataLoader(JESC_DataSet(val), 1, False)

    model = Transformer(6, 6, 8, 728, 100, 0.1)
    model.to(device)    
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(epoch):
        model.train(True)
        train_loss = 0.0
        valid_loss = 0.0
        t = tqdm(enumerate(train_dataloader))
        for i, data in t:
            optim.zero_grad()
            en, ja = data["en"].to(device), data["ja"].to(device)
            out = model(en)
            loss = loss_fn(out, ja)
            train_loss += float(loss)
            loss.backward()
            t.set_postfix_str("loss={}".format(loss))
        train_loss /= train_dataloader.__len__()
        model.train(False)

        for i, data in tqdm(enumerate(val_dataloader)):
            en, ja = data["en"].to(device), data["ja"].to(device)
            out = model(en)
            loss = loss_fn(out, ja)
            valid_loss += float(loss)
        valid_loss /= val_dataloader.__len__()
        print("Train Loss:{}, Valid Loss:{}".format(train_loss,valid_loss))

    return model


if __name__ == "__main__":
    print(torch.__version__)
    model = train("dataset/train_p", "dataset/dev_p", 10)
    torch.save(model.state_dict(), "output/JESC_Transformer_Model")
