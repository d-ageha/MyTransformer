import sys
import torch
from transformers import AutoTokenizer
from Train import EtoJModel

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("usage: {} dim filename use_mine(bool)".format(sys.argv[0]))
        exit()
    dim = int(sys.argv[1])
    filename = sys.argv[2]
    use_mine = sys.argv[3] == "True"
    en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    ja_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    en_pad_id = en_tokenizer.pad_token_id or 0
    ja_pad_id = ja_tokenizer.pad_token_id or 0

    model = EtoJModel(
        dim,
        en_pad_id,
        ja_pad_id,
        130,
        len(en_tokenizer),
        len(ja_tokenizer),
        use_mine=use_mine,
    )
    model.load_state_dict(torch.load(filename))
    while True:
        en = input()
        en = en_tokenizer([en], padding="max_length", max_length=130)
        en_tokens_list = en["input_ids"]
        en_tokens = torch.tensor(en["input_ids"])
        en_pad_mask = torch.tensor(en["attention_mask"])

        print(en_tokens)
        res = model.translate(
            en_tokens,
            ja_tokenizer.cls_token_id,
            ja_tokenizer.sep_token_id,
            ja_pad_id,
            en_pad_mask,
        )
        print(res)
        print(en_tokenizer.decode(en_tokens[0]))
        print(ja_tokenizer.decode(res[0]))
