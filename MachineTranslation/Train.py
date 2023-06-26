from torch.utils.data.dataloader import DataLoader
import torch
import torch.nn.functional as F
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "Model"))
from TSV_DataSet import TSV_DataSet
from EtoJModel import EtoJModel


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
    train_dataset = TSV_DataSet(train, max_length)
    val_dataset = TSV_DataSet(val, max_length)
    train_dataloader = DataLoader(
        train_dataset,
        batch,
        True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(val_dataset, 1, False)
    print(
        "train_size:{} test_size{}".format(
            train_dataloader.__len__(), val_dataloader.__len__()
        )
    )

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
            for g in optim.param_groups:
                g["lr"] = lr

            en, ja = data["en"], data["ja"]
            en_tokens = torch.stack(en["input_ids"]).transpose(0, 1).to(device)
            ja_tokens = torch.stack(ja["input_ids"]).transpose(0, 1).to(device)

            en_masks, ja_masks = torch.stack(en["attention_mask"]).transpose(0, 1).to(
                device
            ), torch.stack(ja["attention_mask"]).transpose(0, 1).to(device)

            if en_tokens.dim() == 1:
                # if batch size=1, unsqueeze the tokens
                en_tokens, ja_tokens = en_tokens.unsqueeze(0), ja_tokens.unsqueeze(0)
                en_masks, ja_masks = en_masks.unsqueeze(0), ja_masks.unsqueeze(0)

            optim.zero_grad()
            out = model(en_tokens, ja_tokens, en_masks, ja_masks)

            # ignores the last token of output and the first token of the ground truth
            loss = loss_fn(out.transpose(1, 2)[:, :, :-1], ja_tokens[:, 1:])
            loss.backward()
            optim.step()
            model.train(False)
            with torch.no_grad():
                t.set_postfix_str("Epoch: {} loss={} lr={}".format(e, loss, lr))
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
            warmup=int(sys.argv[6]),
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
            + " train_dataset_path test_dataset_path dim epoch batch warmup use_mine model_save_dir model_save_name model_load_filepath previous_steps"
        )
