import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class TSV_DataSet(Dataset):
    def __init__(self, csv: str, max_length: int):
        self.data = pd.read_csv(csv, header=None, sep="\t", on_bad_lines="warn", engine="python")
        self.max_length = max_length
        self.en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.ja_tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        en = self.en_tokenizer(
            self.data.iloc[idx][0], padding="max_length", max_length=self.max_length, truncation=True
        )
        ja = self.ja_tokenizer(
            self.data.iloc[idx][1], padding="max_length", max_length=self.max_length, truncation=True
        )
        return {"en": en, "ja": ja}
