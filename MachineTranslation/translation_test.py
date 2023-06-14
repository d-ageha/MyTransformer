import sys
import torch
from Train import EtoJModel
from JESC_DataSet import JESC_DataSet

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: {} dim filename mode".format(sys.argv[0]))
        exit()
    dim = int(sys.argv[1])
    filename = sys.argv[2]
    use_mine = sys.argv[3] == "mine"
    dataset = JESC_DataSet("dataset/dev_p", 130)
    en_pad_id = dataset.en_tokenizer.pad_token_id or 0
    ja_pad_id = dataset.ja_tokenizer.pad_token_id or 0
    model = EtoJModel(dim, en_pad_id, ja_pad_id, 130, len(
        dataset.en_tokenizer), len(dataset.ja_tokenizer), use_mine=use_mine)
    model.load_state_dict(torch.load(filename))
    while True:
        en = input()
        en = dataset.en_tokenizer([en], padding="max_length", max_length=130)
        en_tokens_list = en["input_ids"]
        en_tokens = torch.tensor(en["input_ids"])
        en_pad_mask = torch.tensor(en["attention_mask"])
        print(en_tokens)
        res = model.translate(en_tokens, dataset.ja_tokenizer.cls_token_id,
                              dataset.ja_tokenizer.sep_token_id, ja_pad_id, en_pad_mask)
        print(res)
        print(dataset.en_tokenizer.decode(en_tokens[0]))
        print(dataset.ja_tokenizer.decode(res[0]))
