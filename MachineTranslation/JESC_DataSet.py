import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset


class JESC_DataSet(Dataset):
    def __init__(self, csv: str, max_length: int):
        self.data = pd.read_csv(csv, sep="\t")
        self.max_length = max_length
        self.en_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.ja_tokenizer = AutoTokenizer.from_pretrained(
            "cl-tohoku/bert-base-japanese"
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        en = self.en_tokenizer(
            self.data.iloc[idx][0], padding="max_length", max_length=self.max_length
        )

        ja = self.ja_tokenizer(
            self.data.iloc[idx][1], padding="max_length", max_length=self.max_length
        )
        return {"en": en, "ja": ja}
