from pathlib import Path
import random
import pandas as pd
from torch.utils.data import Dataset


def load_parallel(en_path, nl_path):
    en = Path(en_path).read_text(encoding="utf-8").splitlines()
    nl = Path(nl_path).read_text(encoding="utf-8").splitlines()
    assert len(en) == len(nl)
    return list(zip(en, nl))


def filter_pairs(pairs, max_len=128, min_len=3):
    out = []
    for en, nl in pairs:
        if not en.strip() or not nl.strip():
            continue
        if len(en.split()) < min_len or len(nl.split()) < min_len:
            continue
        if len(en.split()) > max_len or len(nl.split()) > max_len:
            continue
        out.append((en, nl))
    return out


def subsample_and_split(pairs, total_size, val_size, seed):
    random.seed(seed)
    random.shuffle(pairs)
    pairs = pairs[:total_size]
    return pairs[:-val_size], pairs[-val_size:]


def save_pairs(pairs, out_en, out_nl):
    Path(out_en).write_text("\n".join(p[0] for p in pairs), encoding="utf-8")
    Path(out_nl).write_text("\n".join(p[1] for p in pairs), encoding="utf-8")


class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer, max_len):
        self.src = Path(src_file).read_text(encoding="utf-8").splitlines()
        self.tgt = Path(tgt_file).read_text(encoding="utf-8").splitlines()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        src = self.tokenizer(
            self.src[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        tgt = self.tokenizer(
            self.tgt[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )

        labels = tgt["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


def load_flores(flores_dir):
    en = pd.read_parquet(f"{flores_dir}/eng_Latn.parquet")["text"].tolist()
    nl = pd.read_parquet(f"{flores_dir}/nld_Latn.parquet")["text"].tolist()
    return en, nl


def load_challenge_excel(path):
    df = pd.read_excel(path)
    return df["English Source"].tolist(), df["Reference Translation"].tolist()