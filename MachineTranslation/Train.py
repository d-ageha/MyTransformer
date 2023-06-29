from torch.utils.data.dataloader import DataLoader
import torch
import torch.utils.data
import torch.nn.functional as F
from random import sample
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.join(os.path.dirname(sys.path[0]), "Model"))
from TSV_DataSet import TSV_DataSet
from EtoJModel import EtoJModel


def get_learning_rate(step: int, d_model: int, warmup: int):
    return (d_model**-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))


def prepare_data(data, device, batch):
    en, ja = data["en"], data["ja"]
    en_tokens = torch.stack(en["input_ids"]).transpose(0, 1).to(device)
    ja_tokens = torch.stack(ja["input_ids"]).transpose(0, 1).to(device)
    en_masks, ja_masks = torch.stack(en["attention_mask"]).transpose(0, 1).to(device), torch.stack(
        ja["attention_mask"]
    ).transpose(0, 1).to(device)

    return en_tokens, ja_tokens, en_masks, ja_masks


def train(
    dataset_filepath: str,
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

    dataset = TSV_DataSet(dataset_filepath, max_length)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [dataset.__len__() - 100, 100], generator=torch.Generator().manual_seed(1023)
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch,
        pin_memory=True,
    )
    val_dataloader = DataLoader(val_dataset, 1, False)

    print("train_size: {} test_size: {}".format(train_dataloader.__len__(), val_dataloader.__len__()))

    en_pad_id = dataset.en_tokenizer.pad_token_id or 0
    ja_pad_id = dataset.ja_tokenizer.pad_token_id or 0
    model = EtoJModel(
        dim,
        en_pad_id,
        ja_pad_id,
        max_length,
        len(dataset.en_tokenizer),
        len(dataset.ja_tokenizer),
        use_mine,
    )
    if model_load_filepath:
        model.load_state_dict(torch.load(model_load_filepath, map_location=torch.device(device)))
    model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1, betas=(0.9, 0.98))
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ja_pad_id, label_smoothing=0.1)
    step = 0
    previous_val_loss = float("inf")
    for e in range(epoch):
        model.train(True)
        t = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for _, data in t:
            step += 1
            if previous_steps > step:
                continue

            lr = get_learning_rate(step, dim, warmup)
            for g in optim.param_groups:
                g["lr"] = lr

            en_tokens, ja_tokens, en_masks, ja_masks = prepare_data(data, device, batch)

            optim.zero_grad()
            out = model(en_tokens, ja_tokens, en_masks, ja_masks)

            # ignores the last token of output and the first token of the ground truth
            loss = loss_fn(out.transpose(1, 2)[:, :, :-1], ja_tokens[:, 1:])
            loss.backward()
            optim.step()
            model.train(False)
            t.set_postfix_str("Epoch: {} loss={} lr={}".format(e, loss, lr))
            if step == 0 or step % 100 != 0:
                continue

            with torch.no_grad():
                val_loss = 0.0
                preview = sample(range(len(val_dataloader)), 10)
                for i, data in enumerate(val_dataloader):
                    en_tokens, ja_tokens, en_masks, ja_masks = prepare_data(data, device, batch)
                    out = model.forward(en_tokens, ja_tokens, en_masks, ja_tokens)
                    val_loss += loss_fn(out.transpose(1, 2)[:, :, :-1], ja_tokens[:, 1:])
                    if i in preview:
                        out_tokens = model.ja_decode(out).to(device)
                        out_tokens = out_tokens.masked_fill(ja_masks == 0, ja_pad_id)
                        print("")
                        print(dataset.en_tokenizer.decode(en_tokens[0], skip_special_tokens=True))
                        print(dataset.ja_tokenizer.decode(out_tokens[0], skip_special_tokens=True))
                        print(dataset.ja_tokenizer.decode(ja_tokens[0], skip_special_tokens=True))

                ex_sentence = "If you don't like bread, especially baguette, don't talk to me."
                en = dataset.en_tokenizer([ex_sentence], padding="max_length", max_length=130)
                en_tokens = torch.tensor(en["input_ids"]).to(device)
                en_pad_mask = torch.tensor(en["attention_mask"]).to(device)

                print("sample translation:{}".format(ex_sentence))
                print(
                    dataset.ja_tokenizer.decode(
                        model.translate(
                            en_tokens,
                            dataset.ja_tokenizer.cls_token_id,
                            dataset.ja_tokenizer.sep_token_id,
                            ja_pad_id,
                            en_pad_mask,
                            device,
                        )[0],
                        skip_special_tokens=True,
                    )
                )
                print("(val_loss:{}  previous best:{}).".format(val_loss, previous_val_loss))
                if val_loss.__float__() < previous_val_loss:
                    print("The loss is smaller than before. Saving the model.")
                    torch.save(model.state_dict(), model_save_dir + model_save_name)
                    log = open(model_save_dir + "last_saved_step.txt", "w")
                    log.write(str(step))
                    log.close()
                    previous_val_loss = val_loss
    return model


if __name__ == "__main__":
    print(torch.__version__)
    if sys.argv.__len__() == 11:
        model = train(
            sys.argv[1],
            int(sys.argv[2]),
            int(sys.argv[3]),
            int(sys.argv[4]),
            warmup=int(sys.argv[5]),
            use_mine=sys.argv[6] == "True",
            model_save_dir=sys.argv[7],
            model_save_name=sys.argv[8],
            model_load_filepath=sys.argv[9],
            previous_steps=int(sys.argv[10]),
        )
    if sys.argv.__len__() == 9:
        model = train(
            sys.argv[1],
            int(sys.argv[2]),
            int(sys.argv[3]),
            int(sys.argv[4]),
            warmup=int(sys.argv[5]),
            use_mine=sys.argv[6] == "True",
            model_save_dir=sys.argv[7],
            model_save_name=sys.argv[8],
        )
    elif sys.argv.__len__() == 1:
        model = train("dataset/train_p", 128, 2, 5)
    else:
        print(
            sys.argv[0]
            + " dataset_path dim epoch batch warmup use_mine model_save_dir model_save_name model_load_filepath previous_steps"
        )
