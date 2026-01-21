from pathlib import Path
import random
import pandas as pd
from torch.utils.data import Dataset


def load_parallel(en_path, nl_path):
    en = Path(en_path).read_text(encoding="utf-8").splitlines()
    nl = Path(nl_path).read_text(encoding="utf-8").splitlines()
    assert len(en) == len(nl)
    return list(zip(en, nl))


def filter_pairs(pairs, max_len, min_len):
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
    def __init__(self, src, tgt, tokenizer, max_len):
        self.src = Path(src).read_text(encoding="utf-8").splitlines()
        self.tgt = Path(tgt).read_text(encoding="utf-8").splitlines()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        src = self.tok(self.src[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")
        tgt = self.tok(self.tgt[i], truncation=True, padding="max_length",
                       max_length=self.max_len, return_tensors="pt")

        labels = tgt["input_ids"]
        labels[labels == self.tok.pad_token_id] = -100

        return {
            "input_ids": src["input_ids"].squeeze(),
            "attention_mask": src["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


class DecoderOnlyTranslationDataset(Dataset):
    def __init__(self, src, tgt, tokenizer, max_len):
        self.src = Path(src).read_text(encoding="utf-8").splitlines()
        self.tgt = Path(tgt).read_text(encoding="utf-8").splitlines()
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        prompt = (
            "Translate the following English software UI text into Dutch.\n\n"
            "Rules:\n"
            "- Preserve placeholders like {1}, {2}\n"
            "- Preserve symbols and units\n"
            "- Do not add or remove information\n\n"
            f"English:\n{self.src[i]}\n\nDutch:\n"
        )

        full = prompt + self.tgt[i]
        enc = self.tok(full, truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")

        labels = enc["input_ids"].clone()
        prompt_len = len(self.tok(prompt)["input_ids"])
        labels[:, :prompt_len] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
        }


def load_flores(dir):
    en = pd.read_parquet(f"{dir}/eng_Latn.parquet")["text"].tolist()
    nl = pd.read_parquet(f"{dir}/nld_Latn.parquet")["text"].tolist()
    return en, nl


def load_challenge_excel(path):
    df = pd.read_excel(path)
    return df["English Source"].tolist(), df["Reference Translation"].tolist()
