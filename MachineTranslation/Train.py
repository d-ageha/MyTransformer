from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "Model"))
from Transformer import Transformer
from JESC_DataSet import JESC_DataSet


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
        if use_mine:
            self.transformer = Transformer(6, 6, 8, model_dim, max_seq_len, 0.1)
        else:
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

    def translate(self, x, sos_id, eos_id, ja_pad_id, pad_mask):
        result = torch.tensor([sos_id]).unsqueeze(0)
        for i in range(self.max_seq_len - 1):
            y_pads = torch.tensor(
                [[ja_pad_id for j in range(self.max_seq_len - (i + 1))]]
            )
            y = torch.cat((result, y_pads), dim=1)
            y_pad_mask = [1 for j in range(i + 1)] + [
                0 for j in range(self.max_seq_len - (i + 1))
            ]
            y_pad_mask = torch.tensor(y_pad_mask, dtype=torch.int64).unsqueeze(0)
            if not self.use_mine:
                pad_mask = pad_mask == 0
                y_pad_mask = y_pad_mask == 0
            y = self.forward(x, y, pad_mask, y_pad_mask)
            next_token = torch.tensor([[self.ja_decode(y)[0, i]]])
            result = torch.cat((result, next_token), dim=1)
            if result[0, -1] == eos_id:
                break
        return result


def get_learning_rate(step: int, d_model: int, warmup: int):
    return (d_model**-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))


def train(
    train: str,
    val: str,
    dim=256,
    epoch=10,
    batch=1,
    warmup=4000,
    model_save_dir: str = "./output/",
    model_save_name: str = "model",
    model_load_filepath: str | None = None,
    use_mine: bool = True,
    previous_steps: int = 0,
):
    if not model_save_dir.endswith("/"):
        model_save_dir = model_save_dir + "/"
    print("output:" + model_save_dir + model_save_name)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    print(device)
    max_length = 130
    train_dataset = JESC_DataSet(train, max_length)
    val_dataset = JESC_DataSet(val, max_length)
    train_dataloader = DataLoader(train_dataset, batch, True)
    val_dataloader = DataLoader(val_dataset, 1, False)

    en_pad_id = train_dataset.en_tokenizer.pad_token_id or 0
    ja_pad_id = train_dataset.ja_tokenizer.pad_token_id or 0
    model = EtoJModel(
        dim,
        en_pad_id,
        ja_pad_id,
        max_length,
        len(train_dataset.en_tokenizer),
        len(train_dataset.ja_tokenizer),
        use_mine,
    )
    if model_load_filepath:
        model.load_state_dict(
            torch.load(model_load_filepath, map_location=torch.device(device))
        )
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ja_pad_id, label_smoothing=0.1)
    step = 0
    for e in range(epoch):
        model.train(True)
        train_loss = 0.0
        valid_loss = 0.0
        t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for i, data in t:
            step += 1
            if previous_steps > step:
                continue

            lr = get_learning_rate(step, dim, warmup)
            print("lr:{}".format(lr))
            for g in optim.param_groups:
                g["lr"] = lr

            en, ja = data["en"], data["ja"]
            en_tokens = torch.stack(en["input_ids"]).transpose(0, 1).to(device)
            ja_tokens = torch.stack(ja["input_ids"]).transpose(0, 1).to(device)
            en_masks, ja_masks = torch.stack(en["attention_mask"]).transpose(0, 1).to(
                device
            ), torch.stack(ja["attention_mask"]).transpose(0, 1).to(device)

            if en_tokens.dim() == 1:
                # it batch size=1, unsqueeze and tensorize(?) the tokens
                en_tokens, ja_tokens = en_tokens.unsqueeze(0), ja_tokens.unsqueeze(0)
                en_masks, ja_masks = en_masks.unsqueeze(0), ja_masks.unsqueeze(0)

            optim.zero_grad()
            out = model(en_tokens, ja_tokens, en_masks, ja_masks)
            # ignore the last token of output and the first token of the ground truth

            loss = loss_fn(out.transpose(1, 2)[:, :, :-1], ja_tokens[:, 1:])
            loss.backward()
            optim.step()
            model.train(False)
            with torch.no_grad():
                t.set_postfix_str("Epoch: {} loss={}".format(e, loss))
            if step != 0 and step % 100 == 0:
                out_tokens = model.ja_decode(out).to(device)
                out_tokens = out_tokens.masked_fill(ja_masks == 0, ja_pad_id)
                print(train_dataset.en_tokenizer.decode(en_tokens[0]))
                print(train_dataset.ja_tokenizer.decode(out_tokens[0]))
                print(train_dataset.ja_tokenizer.decode(ja_tokens[0]))
                torch.save(model.state_dict(), model_save_dir + model_save_name)
                log = open(model_save_dir + "last_saved_step.txt", "w")
                log.write(str(step))
                log.close()

    return model


if __name__ == "__main__":
    print(torch.__version__)
    if sys.argv.__len__() == 12:
        model = train(
            sys.argv[1],
            sys.argv[2],
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
            warmup=int(sys.argv[6])
            use_mine=bool(sys.argv[7]),
            model_save_dir=sys.argv[8],
            model_save_name=sys.argv[9],
            model_load_filepath=sys.argv[10],
            previous_steps=int(sys.argv[11]),
        )
    if sys.argv.__len__() == 10:
        model = train(
            sys.argv[1],
            sys.argv[2],
            int(sys.argv[3]),
            int(sys.argv[4]),
            int(sys.argv[5]),
            warmup=int(sys.argv[6]),
            use_mine=bool(sys.argv[7]),
            model_save_dir=sys.argv[8],
            model_save_name=sys.argv[9],
        )
    elif sys.argv.__len__() == 1:
        model = train("dataset/train_p", "dataset/dev_p", 128, 2, 5)
    else:
        print(
            sys.argv[0]
            + " train_dataset_path test_dataset_path use_mine model_save_dir model_save_name model_load_filepath previous_steps"
        )
